//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#ifdef PROXSUITE_WITH_SERIALIZATION
#include "serialization.hpp"

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

DOCTEST_TEST_CASE("test serialization of qp model, results and settings")
{
  std::cout << "--- serialization ---" << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(0);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u);
  qp.solve();

  generic_test(qp.model, TEST_SERIALIZATION_FOLDER "/qp_model");
  generic_test(qp.settings, TEST_SERIALIZATION_FOLDER "/qp_settings");
  generic_test(qp.results, TEST_SERIALIZATION_FOLDER "/qp_results");

  generic_test(qp, TEST_SERIALIZATION_FOLDER "/qp_wrapper");
}

DOCTEST_TEST_CASE(
  "test serialization of eigen matrices with different storage orders")
{
  Eigen::Matrix<float, 2, 2, Eigen::RowMajor> row_matrix;
  Eigen::Matrix<float, 2, 2, Eigen::RowMajor> row_matrix_loaded;
  Eigen::Matrix<float, 2, 2, Eigen::ColMajor> col_matrix_loaded;

  row_matrix << 1, 2, 3, 4;

  proxsuite::serialization::saveToJSON(row_matrix, "row_matrix");
  proxsuite::serialization::loadFromJSON(row_matrix_loaded, "row_matrix");
  proxsuite::serialization::loadFromJSON(col_matrix_loaded, "row_matrix");

  DOCTEST_CHECK(row_matrix_loaded == row_matrix);
  DOCTEST_CHECK(col_matrix_loaded == row_matrix);
}
#endif