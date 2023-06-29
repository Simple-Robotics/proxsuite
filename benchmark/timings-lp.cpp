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
  int smooth = 0;

  T sparsity_factor = 0.75;
  T eps_abs = T(1e-9);
  T elapsed_time = 0.0;
  proxqp::utils::rand::set_seed(1);
  std::cout << "Dense QP" << std::endl;
  for (proxqp::isize dim = 10; dim <= 1000;
       dim = (dim == 10) ? 100 : dim + 100) {

    if (dim == 10 || dim == 100) {
      smooth = 1000;
    } else {
      smooth = 100;
    }

    proxqp::isize n_eq(dim / 2);
    proxqp::isize n_in(dim / 2);
    T strong_convexity_factor(1.e-2);
    std::cout << "dim: " << dim << " n_eq: " << n_eq << " n_in: " << n_in
              << std::endl;

    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp_random.H.setZero();
    auto y_sol = proxqp::utils::rand::vector_rand<T>(n_eq);
    qp_random.g = -qp_random.A.transpose() * y_sol;

    elapsed_time = 0.0;
    timer.stop();
    proxqp::dense::QP<T> qp{
      dim, n_eq, n_in, false, proxqp::HessianType::Zero
    };
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    for (int j = 0; j < smooth; j++) {
      timer.start();
      qp.init(qp_random.H,
              qp_random.g,
              qp_random.A,
              qp_random.b,
              qp_random.C,
              qp_random.l,
              qp_random.u);
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
    std::cout << "timings LP with primal dual backend: \t"
              << elapsed_time * 1e-3 / smooth << "ms" << std::endl;

    elapsed_time = 0.0;
    proxqp::dense::QP<T> qp_compare{
      dim, n_eq, n_in, false, proxqp::HessianType::Dense
    };
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    for (int j = 0; j < smooth; j++) {
      timer.start();
      qp_compare.init(qp_random.H,
                      qp_random.g,
                      qp_random.A,
                      qp_random.b,
                      qp_random.C,
                      qp_random.l,
                      qp_random.u);
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
    std::cout << "timings QP with primal dual backend: \t"
              << elapsed_time * 1e-3 / smooth << "ms" << std::endl;
  }
}
