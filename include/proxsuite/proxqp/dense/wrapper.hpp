//
// Copyright (c) 2022 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_QP_DENSE_WRAPPER_HPP
#define PROXSUITE_QP_DENSE_WRAPPER_HPP
#include <proxsuite/proxqp/dense/solver.hpp>
#include <proxsuite/proxqp/dense/helpers.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>
#include <chrono>

namespace proxsuite {
namespace proxqp {
namespace dense {
///
/// @brief This class defines the API of PROXQP solver with dense backend.
///
/*!
 * Wrapper class for using proxsuite API with dense backend
 * for solving linearly constrained convex QP problem using ProxQp algorithm.
 *
 * Example usage:
 * ```cpp
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <util.hpp>

using T = double;
auto main() -> int {

        // Generate a random QP problem with primal variable dimension of size
dim; n_eq equality constraints and n_in inequality constraints
        ::proxsuite::proxqp::test::rand::set_seed(1);
        proxqp::isize dim = 10;
        proxqp::isize n_eq(dim / 4);
        proxqp::isize n_in(dim / 4);
        T strong_convexity_factor(1.e-2);
        T sparsity_factor = 0.15; // controls the sparsity of each matrix of the
problem generated T eps_abs = T(1e-9); Qp<T> qp{
                        random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor};

        // Solve the problem
        proxqp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
        Qp.settings.eps_abs = eps_abs; // choose accuracy needed
        Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // setup the QP
object Qp.solve(); // solve the problem

        // Verify solution accuracy
        T pri_res = std::max(
                        (qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                        (proxqp::dense::positive_part(qp.C * Qp.results.x -
qp.u) + proxqp::dense::negative_part(qp.C * Qp.results.x - qp.l))
                                        .lpNorm<Eigen::Infinity>());
        T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() *
Qp.results.y + qp.C.transpose() * Qp.results.z) .lpNorm<Eigen::Infinity>();
        VEG_ASSERT(pri_res <= eps_abs);
        VEG_ASSERT(dua_res <= eps_abs);

        // Some solver statistics
        std::cout << "------solving qp with dim: " << dim
                                                << " neq: " << n_eq << " nin: "
<< n_in << std::endl; std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << Qp.results.info.iter
                                                << std::endl;
}
 * ```
 */
///// QP object
template<typename T>
struct QP
{
  Results<T> results;
  Settings<T> settings;
  Model<T> model;
  Workspace<T> work;
  preconditioner::RuizEquilibration<T> ruiz;
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in)
    : results(_dim, _n_eq, _n_in)
    , settings()
    , model(_dim, _n_eq, _n_in)
    , work(_dim, _n_eq, _n_in)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim, _n_eq + _n_in })
  {
    work.timer.stop();
  }
  /*!
   * Setups the QP model (with dense matrix format) and equilibrates it if
   * specified by the user.
   * @param H quadratic cost input defining the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param compute_preconditioner boolean parameter for executing or not the
   * preconditioner.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   */
  void init(std::optional<MatRef<T>> H,
            std::optional<VecRef<T>> g,
            std::optional<MatRef<T>> A,
            std::optional<VecRef<T>> b,
            std::optional<MatRef<T>> C,
            std::optional<VecRef<T>> l,
            std::optional<VecRef<T>> u,
            bool compute_preconditioner = true,
            std::optional<T> rho = std::nullopt,
            std::optional<T> mu_eq = std::nullopt,
            std::optional<T> mu_in = std::nullopt)
  {
    // dense case
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    // check the model is valid
    if (g != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        g.value().rows(),
        model.dim,
        "the dimension wrt the primal variable x variable for initializing g "
        "is not valid.");
    }
    if (b != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        b.value().rows(),
        model.n_eq,
        "the dimension wrt equality constrained variables for initializing b "
        "is not valid.");
    }
    if (u != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        u.value().rows(),
        model.n_in,
        "the dimension wrt inequality constrained variables for initializing u "
        "is not valid.");
    }
    if (l != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        l.value().rows(),
        model.n_in,
        "the dimension wrt inequality constrained variables for initializing l "
        "is not valid.");
    }
    if (H != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.value().rows(),
        model.dim,
        "the row dimension for initializing H is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.value().cols(),
        model.dim,
        "the column dimension for initializing H is not valid.");
    }
    if (A != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.value().rows(),
        model.n_eq,
        "the row dimension for initializing A is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.value().cols(),
        model.dim,
        "the column dimension for initializing A is not valid.");
    }
    if (C != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.value().rows(),
        model.n_in,
        "the row dimension for initializing C is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.value().cols(),
        model.dim,
        "the column dimension for initializing C is not valid.");
    }

    if (settings.initial_guess ==
        InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT) {
      work.refactorize =
        true; // necessary for the first solve (then refactorize only if there
              // is an update of the matrices)
    } else {
      work.refactorize = false;
    }
    work.proximal_parameter_update = false;
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (compute_preconditioner) {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::IDENTITY;
    }
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::setup(H,
                                    g,
                                    A,
                                    b,
                                    C,
                                    u,
                                    l,
                                    settings,
                                    model,
                                    work,
                                    results,
                                    ruiz,
                                    preconditioner_status);
    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };
  /*!
   * Updates the QP model (with dense matrix format) and re-equilibrates it if
   * specified by the user.
   * @param H quadratic cost input defining the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param update_preconditioner bool parameter for updating or not the
   * preconditioner and the associated scaled model.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   */
  void update(const std::optional<MatRef<T>> H,
              std::optional<Vec<T>> g,
              const std::optional<MatRef<T>> A,
              std::optional<Vec<T>> b,
              const std::optional<MatRef<T>> C,
              std::optional<Vec<T>> l,
              std::optional<Vec<T>> u,
              bool update_preconditioner = true,
              std::optional<T> rho = std::nullopt,
              std::optional<T> mu_eq = std::nullopt,
              std::optional<T> mu_in = std::nullopt)
  {
    // dense case
    work.refactorize = false;
    work.proximal_parameter_update = false;
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (update_preconditioner) {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::KEEP;
    }
    bool real_update =
      !(H == std::nullopt && g == std::nullopt && A == std::nullopt &&
        b == std::nullopt && C == std::nullopt && u == std::nullopt &&
        l == std::nullopt);
    if (real_update) {
      proxsuite::proxqp::dense::update(H, g, A, b, C, u, l, model, work);
    }
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::setup(std::optional(model.H),
                                    std::optional(dense::VecRef<T>(model.g)),
                                    std::optional(model.A),
                                    std::optional(dense::VecRef<T>(model.b)),
                                    std::optional(model.C),
                                    std::optional(dense::VecRef<T>(model.u)),
                                    std::optional(dense::VecRef<T>(model.l)),
                                    settings,
                                    model,
                                    work,
                                    results,
                                    ruiz,
                                    preconditioner_status);
    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };
  /*!
   * Updates the QP model vectors only (to avoid ambiguity through overloading)
   * and equilibrates it if specified by the user.
   * @param H quadratic cost input defingit reset --soft HEAD~ing the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param update_preconditioner bool parameter for executing or not the
   * preconditioner.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   */
  void update([[maybe_unused]] const std::nullopt_t H,
              std::optional<Vec<T>> g,
              [[maybe_unused]] const std::nullopt_t A,
              std::optional<Vec<T>> b,
              [[maybe_unused]] const std::nullopt_t C,
              std::optional<Vec<T>> l,
              std::optional<Vec<T>> u,
              bool update_preconditioner = true,
              std::optional<T> rho = std::nullopt,
              std::optional<T> mu_eq = std::nullopt,
              std::optional<T> mu_in = std::nullopt)
  {
    work.refactorize = false;
    work.proximal_parameter_update = false;
    // treat the case when H, A and C are nullopt, in order to avoid ambiguity
    // through overloading
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (update_preconditioner) {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::KEEP;
    }
    bool real_update = !(g == std::nullopt && b == std::nullopt &&
                         u == std::nullopt && l == std::nullopt);
    if (real_update) {
      // check the model is valid
      if (g != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(g.value().rows(),
                                      model.dim,
                                      "the dimension wrt primal variable x "
                                      "variable for updating g is not valid.");
      }
      if (b != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(b.value().rows(),
                                      model.n_eq,
                                      "the dimension wrt equality constrained "
                                      "variables for updating b is not valid.");
      }
      if (u != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(
          u.value().rows(),
          model.n_in,
          "the dimension wrt inequality constrained variables for updating u "
          "is not valid.");
      }
      if (l != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(
          l.value().rows(),
          model.n_in,
          "the dimension wrt inequality constrained variables for updating l "
          "is not valid.");
      }
      // update the model
      if (g != std::nullopt) {
        model.g = g.value().eval();
      }
      if (b != std::nullopt) {
        model.b = b.value().eval();
      }
      if (u != std::nullopt) {
        model.u = u.value().eval();
      }
      if (l != std::nullopt) {
        model.l = l.value().eval();
      }
    }
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::setup(std::optional(model.H),
                                    std::optional(dense::VecRef<T>(model.g)),
                                    std::optional(model.A),
                                    std::optional(dense::VecRef<T>(model.b)),
                                    std::optional(model.C),
                                    std::optional(dense::VecRef<T>(model.u)),
                                    std::optional(dense::VecRef<T>(model.l)),
                                    settings,
                                    model,
                                    work,
                                    results,
                                    ruiz,
                                    preconditioner_status);
    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };
  /*!
   * Solves the QP problem using PRXOQP algorithm.
   */
  void solve()
  {
    qp_solve( //
      settings,
      model,
      results,
      work,
      ruiz);
  };
  /*!
   * Solves the QP problem using PROXQP algorithm using a warm start.
   * @param x primal warm start.
   * @param y dual equality warm start.
   * @param z dual inequality warm start.
   */
  void solve(std::optional<VecRef<T>> x,
             std::optional<VecRef<T>> y,
             std::optional<VecRef<T>> z)
  {
    proxsuite::proxqp::dense::warm_start(x, y, z, results, settings, model);
    qp_solve( //
      settings,
      model,
      results,
      work,
      ruiz);
  };
  /*!
   * Clean-ups solver's results and workspace.
   */
  void cleanup()
  {
    results.cleanup(settings);
    work.cleanup();
  }
};
/*!
 * Solves the QP problem using PROXQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to set
 * up some of the solver parameters (warm start, initial guess option, proximal
 * step sizes, absolute and relative accuracies, maximum number of iterations,
 * preconditioner execution).
 * @param H quadratic cost input defining the QP model.
 * @param g linear cost input defining the QP model.
 * @param A equality constraint matrix input defining the QP model.
 * @param b equality constraint vector input defining the QP model.
 * @param C inequality constraint matrix input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param u upper inequality constraint vector input defining the QP model.
 * @param x primal warm start.
 * @param y dual equality constraint warm start.
 * @param z dual inequality constraint warm start.
 * @param verbose if set to true, the solver prints more information about each
 * iteration.
 * @param compute_preconditioner bool parameter for executing or not the
 * preconditioner.
 * @param compute_timings boolean parameter for computing the solver timings.
 * @param rho proximal step size wrt primal variable.
 * @param mu_eq proximal step size wrt equality constrained multiplier.
 * @param mu_in proximal step size wrt inequality constrained multiplier.
 * @param eps_abs absolute accuracy threshold.
 * @param eps_rel relative accuracy threshold.
 * @param max_iter maximum number of iteration.
 * @param initial_guess initial guess option for warm starting or not the
 * initial iterate values.
 */
template<typename T>
proxqp::Results<T>
solve(
  std::optional<MatRef<T>> H,
  std::optional<VecRef<T>> g,
  std::optional<MatRef<T>> A,
  std::optional<VecRef<T>> b,
  std::optional<MatRef<T>> C,
  std::optional<VecRef<T>> l,
  std::optional<VecRef<T>> u,
  std::optional<VecRef<T>> x = std::nullopt,
  std::optional<VecRef<T>> y = std::nullopt,
  std::optional<VecRef<T>> z = std::nullopt,
  std::optional<T> eps_abs = std::nullopt,
  std::optional<T> eps_rel = std::nullopt,
  std::optional<T> rho = std::nullopt,
  std::optional<T> mu_eq = std::nullopt,
  std::optional<T> mu_in = std::nullopt,
  std::optional<bool> verbose = std::nullopt,
  bool compute_preconditioner = true,
  bool compute_timings = true,
  std::optional<isize> max_iter = std::nullopt,
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS)
{
  isize n(0);
  isize n_eq(0);
  isize n_in(0);
  if (H != std::nullopt) {
    n = H.value().rows();
  }
  if (A != std::nullopt) {
    n_eq = A.value().rows();
  }
  if (C != std::nullopt) {
    n_in = C.value().rows();
  }

  QP<T> Qp(n, n_eq, n_in);
  Qp.settings.initial_guess = initial_guess;

  if (eps_abs != std::nullopt) {
    Qp.settings.eps_abs = eps_abs.value();
  }
  if (eps_rel != std::nullopt) {
    Qp.settings.eps_rel = eps_rel.value();
  }
  if (verbose != std::nullopt) {
    Qp.settings.verbose = verbose.value();
  }
  if (max_iter != std::nullopt) {
    Qp.settings.max_iter = verbose.value();
  }
  Qp.settings.compute_timings = compute_timings;
  Qp.init(H, g, A, b, C, l, u, compute_preconditioner, rho, mu_eq, mu_in);
  Qp.solve(x, y, z);

  return Qp.results;
}
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_WRAPPER_HPP */
