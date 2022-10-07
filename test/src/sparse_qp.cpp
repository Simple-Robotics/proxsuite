//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <doctest.hpp>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

using namespace proxsuite;
using T = double;
using I = proxqp::utils::c_int;
using namespace linalg::sparse::tags;
/*
TEST_CASE("random ruiz") {

        for (auto const& dims : {
                                         linalg::veg::tuplify(2, 0,
2), linalg::veg::tuplify(50, 0, 0),
                                         linalg::veg::tuplify(50, 25,
0), linalg::veg::tuplify(50, 0, 25),
                                         linalg::veg::tuplify(50, 10,
25),
                         }) {
                VEG_BIND(auto const&, (n, n_eq, n_in), dims);

                double p = 1.0;

                auto H = proxqp::utils::rand::sparse_positive_definite_rand(n,
T(10.0), p); auto g = proxqp::utils::rand::vector_rand<T>(n); auto AT =
proxqp::utils::rand::sparse_matrix_rand<T>(n, n_eq, p); auto b =
proxqp::utils::rand::vector_rand<T>(n_eq); auto CT =
proxqp::utils::rand::sparse_matrix_rand<T>(n, n_in, p); auto l =
proxqp::utils::rand::vector_rand<T>(n_in); auto u = (l.array() +
1).matrix().eval();

                {
                        sparse::qpView<T, I> qp = {
                                        {linalg::sparse::from_eigen,
H}, {linalg::sparse::from_eigen, g},
                                        {linalg::sparse::from_eigen,
AT}, {linalg::sparse::from_eigen, b},
                                        {linalg::sparse::from_eigen,
CT}, {linalg::sparse::from_eigen, l},
                                        {linalg::sparse::from_eigen,
u},
                        };

                        proxqp::sparse::preconditioner::RuizEquilibration<T, I>
ruiz{ n, n_eq + n_in, 1e-3, 10, proxqp::sparse::preconditioner::Symmetry::UPPER,
                        };

                        sparse::Workspace<T, I> work;
                        Settings<T> settings;
                        Results<T> results;
                        sparse::Data<T, I> data;
                        sparse::qp_setup(qp, results, data, work, ruiz);

                        auto& x = results.x;
                        auto& y = results.y;
                        auto& z = results.z;

                        x.setZero();
                        y.setZero();
                        z.setZero();

                        sparse::qp_solve(results, data, settings, work, ruiz);
                        CHECK(
                                        proxqp::dense::infty_norm(
                                                        H.selfadjointView<Eigen::Upper>()
* x + g + AT * y + CT * z) <= 1e-9);
                        CHECK(proxqp::dense::infty_norm(AT.transpose() * x - b)
<= 1e-9); if (n_in > 0) { CHECK((CT.transpose() * x - l).minCoeff() > -1e-9);
                                CHECK((CT.transpose() * x - u).maxCoeff() <
1e-9);
                        }
                }
        }
}

TEST_CASE("random ruiz using the API") {

        for (auto const& dims : {
                                         linalg::veg::tuplify(2, 0,
2), linalg::veg::tuplify(50, 0, 0),
                                         linalg::veg::tuplify(50, 25,
0), linalg::veg::tuplify(50, 0, 25),
                                         linalg::veg::tuplify(50, 10,
25),
                         }) {
                VEG_BIND(auto const&, (n, n_eq, n_in), dims);

                double p = 1.0;

                auto H = proxqp::utils::rand::sparse_positive_definite_rand(n,
T(10.0), p); auto g = proxqp::utils::rand::vector_rand<T>(n); auto A =
proxqp::utils::rand::sparse_matrix_rand<T>(n_eq,n, p); auto b =
proxqp::utils::rand::vector_rand<T>(n_eq); auto C =
proxqp::utils::rand::sparse_matrix_rand<T>(n_in,n, p); auto l =
proxqp::utils::rand::vector_rand<T>(n_in); auto u = (l.array() +
1).matrix().eval();

                {

                        proxqp::sparse::QP<T,I> qp(n, n_eq, n_in);
                        qp.settings.eps_abs = 1.E-9;
                        qp.setup_sparse_matrices(H,g,A,b,C,u,l);
                        qp.solve();
                        CHECK(
                                        proxqp::dense::infty_norm(
                                                        H.selfadjointView<Eigen::Upper>()
* qp.results.x + g + A.transpose() * qp.results.y + C.transpose() *
qp.results.z) <= 1e-9); CHECK(proxqp::dense::infty_norm(A * qp.results.x - b) <=
1e-9); if (n_in > 0) { CHECK((C * qp.results.x - l).minCoeff() > -1e-9);
                                CHECK((C * qp.results.x - u).maxCoeff() < 1e-9);
                        }
                }
        }
}

TEST_CASE("random id") {

        for (auto const& dims : {
                                         linalg::veg::tuplify(50, 0,
0), linalg::veg::tuplify(50, 25, 0),
                                         linalg::veg::tuplify(10, 0,
10), linalg::veg::tuplify(50, 0, 25),
                                         linalg::veg::tuplify(50, 10,
25),
                         }) {
                VEG_BIND(auto const&, (n, n_eq, n_in), dims);

                double p = 1.0;

                auto H = proxqp::utils::rand::sparse_positive_definite_rand(n,
T(10.0), p); auto g = proxqp::utils::rand::vector_rand<T>(n); auto AT =
proxqp::utils::rand::sparse_matrix_rand<T>(n, n_eq, p); auto b =
proxqp::utils::rand::vector_rand<T>(n_eq); auto CT =
proxqp::utils::rand::sparse_matrix_rand<T>(n, n_in, p); auto l =
proxqp::utils::rand::vector_rand<T>(n_in); auto u = (l.array() +
1).matrix().eval();

                {
                        sparse::qpView<T, I> qp = {
                                        {linalg::sparse::from_eigen,
H}, {linalg::sparse::from_eigen, g},
                                        {linalg::sparse::from_eigen,
AT}, {linalg::sparse::from_eigen, b},
                                        {linalg::sparse::from_eigen,
CT}, {linalg::sparse::from_eigen, l},
                                        {linalg::sparse::from_eigen,
u},
                        };

                        proxqp::sparse::preconditioner::Identity<T, I> id;

                        sparse::Workspace<T, I> work;
                        Settings<T> settings;
                        Results<T> results;
                        sparse::Data<T, I> data;
                        sparse::qp_setup(qp, results, data, work, id);

                        auto& x = results.x;
                        auto& y = results.y;
                        auto& z = results.z;
                        x.setZero();
                        y.setZero();
                        z.setZero();

                        sparse::qp_solve(results, data, settings, work, id);

                        CHECK(
                                        proxqp::dense::infty_norm(
                                                        H.selfadjointView<Eigen::Upper>()
* x + g + AT * y + CT * z) <= 1e-9);
                        CHECK(proxqp::dense::infty_norm(AT.transpose() * x - b)
<= 1e-9); if (n_in > 0) { CHECK((CT.transpose() * x - l).minCoeff() > -1e-9);
                                CHECK((CT.transpose() * x - u).maxCoeff() <
1e-9);
                        }
                }
        }
}
*/
TEST_CASE("random id using the API")
{

  for (auto const& dims : {
         linalg::veg::tuplify(50, 0, 0),
         linalg::veg::tuplify(50, 25, 0),
         linalg::veg::tuplify(10, 0, 10),
         linalg::veg::tuplify(50, 0, 25),
         linalg::veg::tuplify(50, 10, 25),
       }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double p = 1.0;

    auto H = proxqp::utils::rand::sparse_positive_definite_rand(n, T(10.0), p);
    auto g = proxqp::utils::rand::vector_rand<T>(n);
    auto A = proxqp::utils::rand::sparse_matrix_rand<T>(n_eq, n, p);
    auto b = proxqp::utils::rand::vector_rand<T>(n_eq);
    auto C = proxqp::utils::rand::sparse_matrix_rand<T>(n_in, n, p);
    auto l = proxqp::utils::rand::vector_rand<T>(n_in);
    auto u = (l.array() + 1).matrix().eval();

    {
      proxqp::sparse::QP<T, I> qp(n, n_eq, n_in);
      qp.settings.eps_abs = 1.E-9;
      qp.settings.verbose = true;
      qp.init(H, g, A, b, C, l, u);
      qp.solve();

      CHECK(proxqp::dense::infty_norm(
              H.selfadjointView<Eigen::Upper>() * qp.results.x + g +
              A.transpose() * qp.results.y + C.transpose() * qp.results.z) <=
            1e-9);
      CHECK(proxqp::dense::infty_norm(A * qp.results.x - b) <= 1e-9);
      if (n_in > 0) {
        CHECK((C * qp.results.x - l).minCoeff() > -1e-9);
        CHECK((C * qp.results.x - u).maxCoeff() < 1e-9);
      }
    }
  }
}
