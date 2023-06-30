//
// Copyright (c) 2023 INRIA
//
#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using T = double;
using I = long long;

using namespace proxsuite;
using namespace proxsuite::proxqp;

int
main(int /*argc*/, const char** /*argv*/)
{
  Timer<T> timer;
  int smooth = 100;

  T sparsity_factor = 0.75;
  T eps_abs = T(1e-9);
  T elapsed_time = 0.0;
  proxqp::utils::rand::set_seed(1);
  std::cout << "Dense QP" << std::endl;
  for (proxqp::isize dim = 100; dim <= 1000; dim = dim + 100) {

    proxqp::isize n_eq(dim * 2);
    proxqp::isize n_in(dim * 2);
    std::cout << "dim: " << dim << " n_eq: " << n_eq << " n_in: " << n_in
              << " box: " << dim << std::endl;
    T strong_convexity_factor(1.e-2);

    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    Eigen::Matrix<T, Eigen::Dynamic, 1> x_sol =
      utils::rand::vector_rand<T>(dim);
    Eigen::Matrix<T, Eigen::Dynamic, 1> delta(n_in);
    for (proxqp::isize i = 0; i < n_in; ++i) {
      delta(i) = utils::rand::uniform_rand();
    }
    qp_random.u = qp_random.C * x_sol + delta;
    qp_random.b = qp_random.A * x_sol;
    Eigen::Matrix<T, Eigen::Dynamic, 1> u_box(dim);
    u_box.setZero();
    Eigen::Matrix<T, Eigen::Dynamic, 1> l_box(dim);
    l_box.setZero();
    for (proxqp::isize i = 0; i < dim; ++i) {
      T shift = utils::rand::uniform_rand();
      u_box(i) = x_sol(i) + shift;
      l_box(i) = x_sol(i) - shift;
    }
    // using Mat =
    //   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    // Mat C_enlarged(dim + n_in, dim);
    // C_enlarged.setZero();
    // C_enlarged.topLeftCorner(n_in, dim) = qp_random.C;
    // C_enlarged.bottomLeftCorner(dim, dim).diagonal().array() += 1.;
    // Eigen::Matrix<T, Eigen::Dynamic, 1> u_enlarged(n_in + dim);
    // Eigen::Matrix<T, Eigen::Dynamic, 1> l_enlarged(n_in + dim);
    // u_enlarged.head(n_in) = qp_random.u;
    // u_enlarged.tail(dim) = u_box;
    // l_enlarged.head(n_in) = qp_random.l;
    // l_enlarged.tail(dim) = l_box;

    elapsed_time = 0.0;
    timer.stop();
    proxqp::dense::QP<T> qp{ dim, n_eq, n_in, true, DenseBackend::PrimalLDLT };
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    // qp.settings.verbose = true;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    for (int j = 0; j < smooth; j++) {
      timer.start();
      qp.init(qp_random.H,
              qp_random.g,
              qp_random.A,
              qp_random.b,
              qp_random.C,
              qp_random.l,
              qp_random.u,
              l_box,
              u_box);
      qp.solve();
      timer.stop();
      elapsed_time += timer.elapsed().user;
      if (qp.results.info.pri_res > eps_abs ||
          qp.results.info.dua_res > eps_abs) {
        std::cout << "dual residual " << qp.results.info.dua_res
                  << "; primal residual " << qp.results.info.pri_res
                  << std::endl;
        std::cout << "total number of iteration: " << qp.results.info.iter
                  << std::endl;
      }
    }
    std::cout << "timings QP PrimalLDLT backend : \t"
              << elapsed_time * 1e-3 / smooth << "ms" << std::endl;

    elapsed_time = 0.0;
    proxqp::dense::QP<T> qp_compare{
      dim, n_eq, n_in, true, DenseBackend::PrimalDualLDLT
    };
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    // qp_compare.settings.verbose = true;
    for (int j = 0; j < smooth; j++) {
      timer.start();
      qp_compare.init(qp_random.H,
                      qp_random.g,
                      qp_random.A,
                      qp_random.b,
                      qp_random.C,
                      qp_random.l,
                      qp_random.u,
                      l_box,
                      u_box);
      qp_compare.solve();
      timer.stop();
      elapsed_time += timer.elapsed().user;

      if (qp_compare.results.info.pri_res > eps_abs ||
          qp_compare.results.info.dua_res > eps_abs) {
        std::cout << "dual residual " << qp_compare.results.info.dua_res
                  << "; primal residual " << qp_compare.results.info.pri_res
                  << std::endl;
        std::cout << "total number of iteration: "
                  << qp_compare.results.info.iter << std::endl;
      }
    }
    std::cout << "timings QP PrimalDualLDLT : \t"
              << elapsed_time * 1e-3 / smooth << "ms" << std::endl;
    elapsed_time = 0.0;
    proxqp::dense::QP<T> qp_compare_bis{
      dim, n_eq, n_in, true, DenseBackend::Automatic
    };
    qp_compare_bis.settings.eps_abs = eps_abs;
    qp_compare_bis.settings.eps_rel = 0;
    qp_compare_bis.settings.initial_guess =
      InitialGuessStatus::NO_INITIAL_GUESS;
    for (int j = 0; j < smooth; j++) {
      timer.start();
      qp_compare_bis.init(qp_random.H,
                          qp_random.g,
                          qp_random.A,
                          qp_random.b,
                          qp_random.C,
                          qp_random.l,
                          qp_random.u,
                          l_box,
                          u_box);
      qp_compare_bis.solve();
      timer.stop();
      elapsed_time += timer.elapsed().user;

      if (qp_compare_bis.results.info.pri_res > eps_abs ||
          qp_compare_bis.results.info.dua_res > eps_abs) {
        std::cout << "dual residual " << qp_compare_bis.results.info.dua_res
                  << "; primal residual " << qp_compare_bis.results.info.pri_res
                  << std::endl;
        std::cout << "total number of iteration: "
                  << qp_compare_bis.results.info.iter << std::endl;
      }
    }
    std::cout << "timings QP Automatic : \t" << elapsed_time * 1e-3 / smooth
              << "ms" << std::endl;
  }
}
