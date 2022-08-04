//
// Copyright (c) 2022 INRIA
//
#include <proxsuite/proxqp/sparse/sparse.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <doctest.hpp>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::utils;
using T = double;
using I = c_int;
using namespace proxsuite::linalg::sparse::tags;

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update rho")
{
  std::cout << "------------------------sparse random strongly convex qp with "
               "equality and inequality constraints: test update rho"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.verbose = false;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.e-7), std::nullopt, std::nullopt);
    std::cout << "after upating" << std::endl;
    std::cout << "rho :  " << Qp.results.info.rho << std::endl;
    Qp.solve();
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update mus")
{

  std::cout << "------------------------sparse random strongly convex qp with "
               "equality and inequality constraints: test update mus"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);
    
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, std::nullopt, T(1.E-2), T(1.E-3));
    std::cout << "after upating" << std::endl;
    std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;
    std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}
TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test with no equilibration at initialization")
{

  std::cout << "------------------------sparse random strongly convex qp with "
               "equality and inequality constraints: test with no "
               "equilibration at initialization"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, false);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test with equilibration at initialization")
{

  std::cout
    << "------------------------sparse random strongly convex qp with equality "
       "and inequality constraints: test with equilibration at initialization"
    << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}
TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test with no initial guess")
{

  std::cout << "------------------------sparse random strongly convex qp with "
               "equality and inequality constraints: test with no initial guess"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update g for unconstrained problem")
{

  std::cout << "------------------------sparse random strongly convex qp with "
               "equality and inequality constraints: test with no initial guess"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    auto g = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    std::cout << "H before update " << qp.H << std::endl;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    std::cout << "H generated " << H_new << std::endl;
    Qp.update(H_new, g, qp.A, qp.b, qp.C, qp.u, qp.l, false);
    proxsuite::linalg::sparse::MatMut<T, I> kkt_unscaled =
      Qp.model.kkt_mut_unscaled();
    auto kkt_top_n_rows =
      proxsuite::proxqp::sparse::detail::top_rows_mut_unchecked(
        proxsuite::linalg::veg::unsafe, kkt_unscaled, n);

    proxsuite::linalg::sparse::MatMut<T, I> H_unscaled =
      proxsuite::proxqp::sparse::detail::middle_cols_mut(
        kkt_top_n_rows, 0, n, Qp.model.H_nnz);
    std::cout << " H_unscaled " << H_unscaled.to_eigen() << std::endl;
    Qp.solve();

    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test warm starting")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    auto x_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    auto y_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n_eq);
    auto z_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n_in);
    std::cout << "proposed warm start" << std::endl;
    std::cout << "x_wm :  " << x_wm << std::endl;
    std::cout << "y_wm :  " << y_wm << std::endl;
    std::cout << "z_wm :  " << z_wm << std::endl;
    Qp.solve(x_wm, y_wm, z_wm);

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test with warm start with previous result")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints: test with warm start with previous result---"
    << std::endl;

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);
    T eps_abs = 1.E-9;

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in); // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in); // creating QP object
    Qp2.settings.eps_abs = 1.E-9;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);

    auto x = Qp.results.x;
    auto y = Qp.results.y;
    auto z = Qp.results.z;
    Qp2.solve(x, y, z);
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    Qp.solve();
    pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
               qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
    std::cout << "------using API solving qp with dim with Qp after warm start "
                 "with previous result: "
              << n << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
    pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
               qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp2: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with cold start option")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test with cold start option---"
            << std::endl;

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    T eps_abs = 1.E-9;
    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in); // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                         (sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                          sparse::detail::negative_part(qp.C * Qp.results.x - qp.l))
                           .lpNorm<Eigen::Infinity>());
    T dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
                 qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in); // creating QP object
    Qp2.settings.eps_abs = 1.E-9;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);

    auto x = Qp.results.x;
    auto y = Qp.results.y;
    auto z = Qp.results.z;
    // std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x
    // << std::endl;
    Qp2.ruiz.scale_primal_in_place({ proxsuite::proxqp::from_eigen, x });
    Qp2.ruiz.scale_dual_in_place_eq({ proxsuite::proxqp::from_eigen, y });
    Qp2.ruiz.scale_dual_in_place_in({ proxsuite::proxqp::from_eigen, z });
    // std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x
    // << std::endl;
    Qp2.solve(x, y, z);

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    Qp.solve();
    pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
               qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
    std::cout << "------using API solving qp with dim with Qp after warm start "
                 "with cold start option: "
              << n << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
    pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
               qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with cold start option: "
              << n << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test equilibration option")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test equilibration option---"
            << std::endl;

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    T eps_abs = 1.E-9;
    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in); // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l,true);
    Qp.solve();

    T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                         (sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                          sparse::detail::negative_part(qp.C * Qp.results.x - qp.l))
                          .lpNorm<Eigen::Infinity>());
    T dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
                 qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in); // creating QP object
    Qp2.settings.eps_abs = 1.E-9;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, false);
    Qp2.solve();
    pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
               qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp2: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test equilibration option at update")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test equilibration option at update---"
            << std::endl;

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    T eps_abs = 1.E-9;
    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in); // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);
    Qp.solve();
    T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                         (sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                          sparse::detail::negative_part(qp.C * Qp.results.x - qp.l))
                           .lpNorm<Eigen::Infinity>());
    T dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
                 qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
    Qp.solve();

    pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
               qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in); // creating QP object
    Qp2.settings.eps_abs = 1.E-9;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, false);

    Qp2.solve();
    pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
               qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp2: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;

    Qp2.update(std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               false);
    Qp2.solve();
    pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                        sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
               qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    std::cout << "------using API solving qp with dim with Qp2: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test new init")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(n, n_eq, n_in);
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    auto x_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    auto y_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n_eq);
    auto z_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n_in);
    std::cout << "proposed warm start" << std::endl;
    std::cout << "x_wm :  " << x_wm << std::endl;
    std::cout << "y_wm :  " << y_wm << std::endl;
    std::cout << "z_wm :  " << z_wm << std::endl;
    Qp.solve(x_wm, y_wm, z_wm);

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test new init")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in);
    Qp2.settings.eps_abs = 1.E-9;

    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp2.solve();

    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;
  }
}

/////  TESTS ALL INITIAL GUESS OPTIONS FOR MULTIPLE SOLVES AT ONCE
TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with no initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with equality "
          "constrained initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test with equality constrained initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with equality "
          "constrained initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test with warm start with previous result and first solve "
                 "with equality constrained initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with warm start with previous result and first solve "
                 "with no initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with cold start "
          "initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test with cold start with previous result and first solve "
                 "with equality constrained initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with warm start")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with warm start and first solve with no initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: warm start test from init")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with warm start and first solve with no initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(
      qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp2.settings.eps_abs = 1.E-9;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    std::cout << "dirty workspace for Qp2 : " << Qp2.work.internal.dirty
              << std::endl;
    Qp2.solve(Qp.results.x, Qp.results.y, Qp.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve with new QP object" << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;
  }
}

/// TESTS WITH UPDATE + INITIAL GUESS OPTIONS

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update and multiple solve at once with "
          "no initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with no initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    bool update_preconditioner = true;
    Qp.update(H_new, g_new, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
    std::cout << "dirty workspace after update : " << Qp.work.internal.dirty
              << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "equality constrained initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    

    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test with equality constrained initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    bool update_preconditioner = true;
    Qp.update(H_new, g_new, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
    std::cout << "dirty workspace after update : " << Qp.work.internal.dirty
              << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test update + multiple solve at once with equality "
  "constrained initial guess and then warm start with previous results")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test with warm start with previous result and first solve "
                 "with equality constrained initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    bool update_preconditioner = true;
    Qp.update(H_new, g_new, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
    std::cout << "dirty workspace after update : " << Qp.work.internal.dirty
              << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with warm start with previous result and first solve "
                 "with no initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    bool update_preconditioner = true;
    Qp.update(H_new, g_new, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "cold start initial guess and then cold start option")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test with cold start with previous result and first solve "
                 "with equality constrained initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    bool update_preconditioner = true;
    Qp.update(H_new,
              g_new,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              update_preconditioner);
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "warm start")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = 1.E-9;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test with warm start and first solve with no initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    auto x_wm = Qp.results.x; // keep previous result
    auto y_wm = Qp.results.y;
    auto z_wm = Qp.results.z;
    // try with a false update, the warm start should give the exact solution
    bool update_preconditioner = true;
    Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
    std::cout << "dirty workspace after update: " << Qp.work.internal.dirty
              << std::endl;
    Qp.solve(x_wm, y_wm, z_wm);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Second solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    x_wm = Qp.results.x; // keep previous result
    y_wm = Qp.results.y;
    z_wm = Qp.results.z;
    auto H_new = 2. * qp.H; // keep same sparsity structure
    auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    update_preconditioner = true;
    Qp.update(H_new, g_new, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
    std::cout << "dirty workspace after update: " << Qp.work.internal.dirty
              << std::endl;
    Qp.solve(x_wm, y_wm, z_wm);
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Third solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fourth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    std::cout << "dirty workspace : " << Qp.work.internal.dirty << std::endl;
    Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
    dua_res = proxqp::dense::infty_norm(
      H_new.selfadjointView<Eigen::Upper>() * Qp.results.x + g_new +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= 1e-9);
    CHECK(pri_res <= 1E-9);
    std::cout << "Fifth solve " << std::endl;
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
  }
}

TEST_CASE("Test initializaton with rho for different initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test initializaton with rho for different initial guess"
              << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
    Qp.solve();
    CHECK(Qp.results.info.rho == T(1.E-7));
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in);
    Qp2.settings.eps_abs = eps_abs;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
    Qp2.solve();
    CHECK(Qp2.results.info.rho == T(1.E-7));
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp3(n, n_eq, n_in);
    Qp3.settings.eps_abs = eps_abs;
    Qp3.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
    Qp3.solve();
    CHECK(Qp3.results.info.rho == T(1.E-7));
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + qp.g +
      qp.A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp3.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp3.results.info.setup_time
              << " solve time " << Qp3.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp4(n, n_eq, n_in);
    Qp4.settings.eps_abs = eps_abs;
    Qp4.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
    Qp4.solve();
    CHECK(Qp4.results.info.rho == T(1.E-7));
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + qp.g +
      qp.A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp4.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp4.results.info.setup_time
              << " solve time " << Qp4.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp5(n, n_eq, n_in);
    Qp5.settings.eps_abs = eps_abs;
    Qp5.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
    Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
    CHECK(Qp5.results.info.rho == T(1.E-7));
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + qp.g +
      qp.A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp5.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp5.results.info.setup_time
              << " solve time " << Qp5.results.info.solve_time << std::endl;
  }
}

TEST_CASE("Test g update for different initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 2, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

    std::cout << "Test g update for different initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    auto g = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    Qp.update(std::nullopt,
              g,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt);
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK((Qp.model.g - g).lpNorm<Eigen::Infinity>() <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in);
    Qp2.settings.eps_abs = eps_abs;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp2.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp2.update(std::nullopt,
               g,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp2.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK((Qp2.model.g - g).lpNorm<Eigen::Infinity>() <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp3(n, n_eq, n_in);
    Qp3.settings.eps_abs = eps_abs;
    Qp3.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp3.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + qp.g +
      qp.A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp3.update(std::nullopt,
               g,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp3.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + g +
      qp.A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    CHECK((Qp3.model.g - g).lpNorm<Eigen::Infinity>() <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp3.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp3.results.info.setup_time
              << " solve time " << Qp3.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp4(n, n_eq, n_in);
    Qp4.settings.eps_abs = eps_abs;
    Qp4.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp4.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + qp.g +
      qp.A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp4.update(std::nullopt,
               g,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp4.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + g +
      qp.A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    CHECK((Qp4.model.g - g).lpNorm<Eigen::Infinity>() <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp4.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp4.results.info.setup_time
              << " solve time " << Qp4.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp5(n, n_eq, n_in);
    Qp5.settings.eps_abs = eps_abs;
    Qp5.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + qp.g +
      qp.A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp5.update(std::nullopt,
               g,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp5.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + g +
      qp.A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    CHECK((Qp5.model.g - g).lpNorm<Eigen::Infinity>() <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp5.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp5.results.info.setup_time
              << " solve time " << Qp5.results.info.solve_time << std::endl;
  }
}

TEST_CASE("Test A update for different initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(
      qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    // proxqp::sparse::QP<T,I> Qp(n,n_eq,n_in);
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test A update for different initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    SparseMat<T> A = 2 * qp.A; // keep same sparsity structure
    Qp.update(std::nullopt,
              std::nullopt,
              A,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt);
    Qp.settings.verbose = false;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(A * Qp.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
    // get stored A from KKT matrix
    auto kkt_unscaled = Qp.model.kkt_mut_unscaled();
    auto kkt_top_n_rows =
      proxsuite::proxqp::sparse::detail::top_rows_mut_unchecked(
        proxsuite::linalg::veg::unsafe, kkt_unscaled, n);
    SparseMat<T> A_unscaled =
      proxsuite::proxqp::sparse::detail::middle_cols_mut(
        kkt_top_n_rows, n, n_eq, Qp.model.A_nnz)
        .to_eigen()
        .transpose();
    SparseMat<T> diff_mat = A_unscaled - A;
    T diff = std::max(std::abs(diff_mat.coeffs().maxCoeff()),
                      std::abs(diff_mat.coeffs().minCoeff()));
    CHECK(diff <= eps_abs);

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in);
    Qp2.settings.eps_abs = eps_abs;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp2.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp2.update(std::nullopt,
               std::nullopt,
               A,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp2.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    // get stored A from KKT matrix
    kkt_unscaled = Qp2.model.kkt_mut_unscaled();
    kkt_top_n_rows = proxsuite::proxqp::sparse::detail::top_rows_mut_unchecked(
      proxsuite::linalg::veg::unsafe, kkt_unscaled, n);
    A_unscaled = proxsuite::proxqp::sparse::detail::middle_cols_mut(
                   kkt_top_n_rows, n, n_eq, Qp2.model.A_nnz)
                   .to_eigen()
                   .transpose();
    diff_mat = A_unscaled - A;
    diff = std::max(std::abs(diff_mat.coeffs().maxCoeff()),
                    std::abs(diff_mat.coeffs().minCoeff()));
    CHECK(diff <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp3(n, n_eq, n_in);
    Qp3.settings.eps_abs = eps_abs;
    Qp3.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp3.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + qp.g +
      qp.A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp3.update(std::nullopt,
               std::nullopt,
               A,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp3.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + qp.g +
      A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    // get stored A from KKT matrix
    kkt_unscaled = Qp3.model.kkt_mut_unscaled();
    kkt_top_n_rows = proxsuite::proxqp::sparse::detail::top_rows_mut_unchecked(
      proxsuite::linalg::veg::unsafe, kkt_unscaled, n);
    A_unscaled = proxsuite::proxqp::sparse::detail::middle_cols_mut(
                   kkt_top_n_rows, n, n_eq, Qp3.model.A_nnz)
                   .to_eigen()
                   .transpose();
    diff_mat = A_unscaled - A;
    diff = std::max(std::abs(diff_mat.coeffs().maxCoeff()),
                    std::abs(diff_mat.coeffs().minCoeff()));
    CHECK(diff <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp3.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp3.results.info.setup_time
              << " solve time " << Qp3.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp4(n, n_eq, n_in);
    Qp4.settings.eps_abs = eps_abs;
    Qp4.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp4.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + qp.g +
      qp.A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp4.update(std::nullopt,
               std::nullopt,
               A,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp4.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + qp.g +
      A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    // get stored A from KKT matrix
    kkt_unscaled = Qp4.model.kkt_mut_unscaled();
    kkt_top_n_rows = proxsuite::proxqp::sparse::detail::top_rows_mut_unchecked(
      proxsuite::linalg::veg::unsafe, kkt_unscaled, n);
    A_unscaled = proxsuite::proxqp::sparse::detail::middle_cols_mut(
                   kkt_top_n_rows, n, n_eq, Qp4.model.A_nnz)
                   .to_eigen()
                   .transpose();
    diff_mat = A_unscaled - A;
    diff = std::max(std::abs(diff_mat.coeffs().maxCoeff()),
                    std::abs(diff_mat.coeffs().minCoeff()));
    CHECK(diff <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp4.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp4.results.info.setup_time
              << " solve time " << Qp4.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp5(n, n_eq, n_in);
    Qp5.settings.eps_abs = eps_abs;
    Qp5.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + qp.g +
      qp.A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp5.update(std::nullopt,
               std::nullopt,
               A,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt);
    Qp5.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + qp.g +
      A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    // get stored A from KKT matrix
    kkt_unscaled = Qp5.model.kkt_mut_unscaled();
    kkt_top_n_rows = proxsuite::proxqp::sparse::detail::top_rows_mut_unchecked(
      proxsuite::linalg::veg::unsafe, kkt_unscaled, n);
    A_unscaled = proxsuite::proxqp::sparse::detail::middle_cols_mut(
                   kkt_top_n_rows, n, n_eq, Qp5.model.A_nnz)
                   .to_eigen()
                   .transpose();
    diff_mat = A_unscaled - A;
    diff = std::max(std::abs(diff_mat.coeffs().maxCoeff()),
                    std::abs(diff_mat.coeffs().minCoeff()));
    CHECK(diff <= eps_abs);
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp5.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp5.results.info.setup_time
              << " solve time " << Qp5.results.info.solve_time << std::endl;
  }
}

TEST_CASE("Test rho update for different initial guess")
{

  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::QP<T, I> Qp(qp.H.cast<bool>(), qp.A.cast<bool>(), qp.C.cast<bool>());
    // proxqp::sparse::QP<T,I> Qp(n,n_eq,n_in);
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

    std::cout << "Test rho update for different initial guess" << std::endl;
    std::cout << "dirty workspace before any solving: "
              << Qp.work.internal.dirty << std::endl;

    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    T pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp.update(std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              true,
              T(1.E-7));
    Qp.settings.verbose = false;
    Qp.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp.results.x + qp.g +
      qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z);
    pri_res =
      std::max(proxqp::dense::infty_norm(qp.A * Qp.results.x - qp.b),
               proxqp::dense::infty_norm(
                 sparse::detail::positive_part(qp.C * Qp.results.x - qp.u) +
                 sparse::detail::negative_part(qp.C * Qp.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
              << Qp.results.info.solve_time << std::endl;
    CHECK(Qp.results.info.rho == T(1.E-7));

    proxqp::sparse::QP<T, I> Qp2(n, n_eq, n_in);
    Qp2.settings.eps_abs = eps_abs;
    Qp2.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp2.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp2.update(std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               true,
               T(1.E-7));
    Qp2.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp2.results.x + qp.g +
      qp.A.transpose() * Qp2.results.y + qp.C.transpose() * Qp2.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp2.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp2.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp2.results.x - qp.l)));
    CHECK(Qp2.results.info.rho == T(1.E-7));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp2.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp2.results.info.setup_time
              << " solve time " << Qp2.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp3(n, n_eq, n_in);
    Qp3.settings.eps_abs = eps_abs;
    Qp3.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp3.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + qp.g +
      qp.A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp3.update(std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               true,
               T(1.E-7));
    Qp3.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp3.results.x + qp.g +
      qp.A.transpose() * Qp3.results.y + qp.C.transpose() * Qp3.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp3.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp3.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp3.results.x - qp.l)));
    CHECK(Qp3.results.info.rho == T(1.E-7));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp3.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp3.results.info.setup_time
              << " solve time " << Qp3.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp4(n, n_eq, n_in);
    Qp4.settings.eps_abs = eps_abs;
    Qp4.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
    Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp4.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + qp.g +
      qp.A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp4.update(std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               true,
               T(1.E-7));
    Qp4.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp4.results.x + qp.g +
      qp.A.transpose() * Qp4.results.y + qp.C.transpose() * Qp4.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp4.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp4.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp4.results.x - qp.l)));
    CHECK(Qp4.results.info.rho == T(1.E-7));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp4.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp4.results.info.setup_time
              << " solve time " << Qp4.results.info.solve_time << std::endl;

    proxqp::sparse::QP<T, I> Qp5(n, n_eq, n_in);
    Qp5.settings.eps_abs = eps_abs;
    Qp5.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START;
    Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + qp.g +
      qp.A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    Qp5.update(std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               true,
               T(1.E-7));
    Qp5.solve();
    dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * Qp5.results.x + qp.g +
      qp.A.transpose() * Qp5.results.y + qp.C.transpose() * Qp5.results.z);
    pri_res = std::max(proxqp::dense::infty_norm(qp.A * Qp5.results.x - qp.b),
                       proxqp::dense::infty_norm(
                         sparse::detail::positive_part(qp.C * Qp5.results.x - qp.u) +
                         sparse::detail::negative_part(qp.C * Qp5.results.x - qp.l)));
    CHECK(Qp5.results.info.rho == T(1.E-7));
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in
              << std::endl;
    std::cout << "; dual residual " << dua_res << "; primal residual "
              << pri_res << std::endl;
    std::cout << "total number of iteration: " << Qp5.results.info.iter
              << std::endl;
    std::cout << "setup timing " << Qp5.results.info.setup_time
              << " solve time " << Qp5.results.info.solve_time << std::endl;
  }
}
