//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <proxsuite/serialization/model.hpp>
#include <proxsuite/serialization/results.hpp>
#include <proxsuite/serialization/settings.hpp>
#include <proxsuite/serialization/ruiz.hpp>
#include <proxsuite/serialization/wrapper.hpp>

template<typename object>
struct init;

template<typename T>
struct init<proxsuite::proxqp::dense::Model<T>>
{
  typedef proxsuite::proxqp::dense::Model<T> Model;

  static Model run()
  {
    Model model(1, 0, 0);
    return model;
  }
};

template<typename T>
struct init<proxsuite::proxqp::Results<T>>
{
  typedef proxsuite::proxqp::Results<T> Results;

  static Results run()
  {
    Results results;
    return results;
  }
};

template<typename T>
struct init<proxsuite::proxqp::Settings<T>>
{
  typedef proxsuite::proxqp::Settings<T> Settings;

  static Settings run()
  {
    Settings settings;
    return settings;
  }
};
template<typename T>
struct init<proxsuite::proxqp::dense::QP<T>>
{
  typedef proxsuite::proxqp::dense::QP<T> QP;

  static QP run()
  {
    QP qp(1, 0, 0);
    return qp;
  }
};

template<typename T>
void
generic_test(const T& object, const std::string& filename)
{
  using namespace proxsuite::serialization;

  // Load and save as XML
  const std::string xml_filename = filename + ".xml";
  saveToXML(object, xml_filename);

  {
    T object_loaded = init<T>::run();
    loadFromXML(object_loaded, xml_filename);

    // Check
    DOCTEST_CHECK(object_loaded == object);
  }

  // Load and save as json
  const std::string json_filename = filename + ".json";
  saveToJSON(object, json_filename);

  {
    T object_loaded = init<T>::run();
    loadFromJSON(object_loaded, json_filename);

    // Check
    DOCTEST_CHECK(object_loaded == object);
  }

  // Load and save as binary
  const std::string bin_filename = filename + ".bin";
  saveToBinary(object, bin_filename);

  {
    T object_loaded = init<T>::run();
    loadFromBinary(object_loaded, bin_filename);

    // Check
    DOCTEST_CHECK(object_loaded == object);
  }
}

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

DOCTEST_TEST_CASE("test serialization of qp model, results and settings")
{
  std::cout << "--- serialization ---" << std::endl;
  double sparsity_factor = 0.15;
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