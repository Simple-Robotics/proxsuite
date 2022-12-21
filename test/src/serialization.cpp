//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#ifdef PROXSUITE_WITH_SERIALIZATION
#include <proxsuite/proxqp/serialization/archive.hpp>
#include <proxsuite/proxqp/serialization/eigen.hpp>
#include <proxsuite/proxqp/serialization/model.hpp>
#include <proxsuite/proxqp/serialization/results.hpp>
#include <proxsuite/proxqp/serialization/settings.hpp>

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

DOCTEST_TEST_CASE("ProxQP::test serialization")
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

  proxsuite::serialization::saveToBinary(qp.model, "qp_model.bin");
  proxsuite::serialization::saveToJSON(qp.results, "results.json");
  proxsuite::serialization::saveToXML(qp.results, "results.xml");
  proxsuite::serialization::loadFromBinary(qp.model, "qp_model.bin");

  proxsuite::serialization::saveToJSON(qp.model, "qp_model.json");
  proxsuite::serialization::loadFromJSON(qp.model, "qp_model.json");
  proxsuite::serialization::loadFromJSON(qp.results, "results.json");
  proxsuite::serialization::loadFromXML(qp.results, "results.xml");
}
#endif