//
// Copyright (c) 2025 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_WRAPPER_HPP
#define PROXSUITE_OSQP_DENSE_WRAPPER_HPP

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/osqp/dense/solver.hpp>
#include <proxsuite/solvers/common/wrapper.hpp>

namespace proxsuite {
namespace osqp {
namespace dense {

namespace pp = proxsuite::proxqp;
namespace ppd = proxsuite::proxqp::dense;
namespace pod = proxsuite::osqp::dense;

///
/// @brief This class defines the API of OSQP solver with dense backend.
///
template<typename T>
struct QP : public ppd::QP<T>
{
public:
  /*!
   * Class constructors.
   */
  using ppd::QP<T>::QP;
  /*!
   * Solves the QP problem using OSQP algorithm.
   */
  void solve()
  {
    pod::qp_solve( //
      this->settings,
      this->model,
      this->results,
      this->work,
      this->get_box_constraints(),
      this->get_dense_backend(),
      this->get_hessian_type(),
      this->ruiz);
  };
  /*!
   * Solves the QP problem using OSQP algorithm using a warm start.
   * @param x primal warm start.
   * @param y dual equality warm start.
   * @param z dual inequality warm start.
   */
  void solve(optional<VecRef<T>> x,
             optional<VecRef<T>> y,
             optional<VecRef<T>> z)
  {
    ppd::warm_start(x, y, z, this->results, this->settings, this->model);
    pod::qp_solve( //
      this->settings,
      this->model,
      this->results,
      this->work,
      this->get_box_constraints(),
      this->get_dense_backend(),
      this->get_hessian_type(),
      this->ruiz);
  };
};
/*!
 * Solves the QP problem using OSQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to set
 * up some of the solver parameters (warm start, initial guess option, proximal
 * step sizes, absolute and relative accuracies, maximum number of iterations,
 * preconditioner execution). There are no box constraints in the model.
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
 * @param check_duality_gap If set to true, include the duality gap in absolute
 * and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 */
template<typename T>
proxqp::Results<T>
solve(optional<MatRef<T>> H,
      optional<VecRef<T>> g,
      optional<MatRef<T>> A,
      optional<VecRef<T>> b,
      optional<MatRef<T>> C,
      optional<VecRef<T>> l,
      optional<VecRef<T>> u,
      optional<VecRef<T>> x = nullopt,
      optional<VecRef<T>> y = nullopt,
      optional<VecRef<T>> z = nullopt,
      optional<T> eps_abs = nullopt,
      optional<T> eps_rel = nullopt,
      optional<T> rho = nullopt,
      optional<T> mu_eq = nullopt,
      optional<T> mu_in = nullopt,
      optional<bool> verbose = nullopt,
      bool compute_preconditioner = true,
      bool compute_timings = false,
      optional<isize> max_iter = nullopt,
      pp::InitialGuessStatus initial_guess =
        pp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      bool check_duality_gap = false,
      optional<T> eps_duality_gap_abs = nullopt,
      optional<T> eps_duality_gap_rel = nullopt,
      bool primal_infeasibility_solving = false,
      optional<T> manual_minimal_H_eigenvalue = nullopt)
{
  isize n(0);
  isize n_eq(0);
  isize n_in(0);
  if (H != nullopt) {
    n = H.value().rows();
  }
  if (A != nullopt) {
    n_eq = A.value().rows();
  }
  if (C != nullopt) {
    n_in = C.value().rows();
  }

  QP<T> Qp(n, n_eq, n_in, false, DenseBackend::PrimalDualLDLT);

  return proxsuite::common::solve_without_api(Qp,
                                              H,
                                              g,
                                              A,
                                              b,
                                              C,
                                              l,
                                              u,
                                              x,
                                              y,
                                              z,
                                              eps_abs,
                                              eps_rel,
                                              rho,
                                              mu_eq,
                                              mu_in,
                                              verbose,
                                              compute_preconditioner,
                                              compute_timings,
                                              max_iter,
                                              initial_guess,
                                              check_duality_gap,
                                              eps_duality_gap_abs,
                                              eps_duality_gap_rel,
                                              primal_infeasibility_solving,
                                              manual_minimal_H_eigenvalue);
}
/*!
 * Solves the QP problem using OSQP algorithm without the need to define a QP
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
 * @param l_box lower box inequality constraint vector input defining the QP
 * model.
 * @param u_box upper box inequality constraint vector input defining the QP
 * model.
 * @param x primal warm start.
 * @param y dual equality constraint warm start.
 * @param z dual inequality constraint warm start. The upper part must contain a
 * warm start for inequality constraints wrt C matrix, whereas the latter wrt
 * the box inequalities.
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
 * @param check_duality_gap If set to true, include the duality gap in absolute
 * and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 */
template<typename T>
proxqp::Results<T>
solve(optional<MatRef<T>> H,
      optional<VecRef<T>> g,
      optional<MatRef<T>> A,
      optional<VecRef<T>> b,
      optional<MatRef<T>> C,
      optional<VecRef<T>> l,
      optional<VecRef<T>> u,
      optional<VecRef<T>> l_box,
      optional<VecRef<T>> u_box,
      optional<VecRef<T>> x = nullopt,
      optional<VecRef<T>> y = nullopt,
      optional<VecRef<T>> z = nullopt,
      optional<T> eps_abs = nullopt,
      optional<T> eps_rel = nullopt,
      optional<T> rho = nullopt,
      optional<T> mu_eq = nullopt,
      optional<T> mu_in = nullopt,
      optional<bool> verbose = nullopt,
      bool compute_preconditioner = true,
      bool compute_timings = false,
      optional<isize> max_iter = nullopt,
      pp::InitialGuessStatus initial_guess =
        pp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      bool check_duality_gap = false,
      optional<T> eps_duality_gap_abs = nullopt,
      optional<T> eps_duality_gap_rel = nullopt,
      bool primal_infeasibility_solving = false,
      optional<T> manual_minimal_H_eigenvalue = nullopt)
{
  isize n(0);
  isize n_eq(0);
  isize n_in(0);
  if (H != nullopt) {
    n = H.value().rows();
  }
  if (A != nullopt) {
    n_eq = A.value().rows();
  }
  if (C != nullopt) {
    n_in = C.value().rows();
  }

  QP<T> Qp(n, n_eq, n_in, true, DenseBackend::PrimalDualLDLT);

  return proxsuite::common::solve_without_api(Qp,
                                              H,
                                              g,
                                              A,
                                              b,
                                              C,
                                              l,
                                              u,
                                              l_box,
                                              u_box,
                                              x,
                                              y,
                                              z,
                                              eps_abs,
                                              eps_rel,
                                              rho,
                                              mu_eq,
                                              mu_in,
                                              verbose,
                                              compute_preconditioner,
                                              compute_timings,
                                              max_iter,
                                              initial_guess,
                                              check_duality_gap,
                                              eps_duality_gap_abs,
                                              eps_duality_gap_rel,
                                              primal_infeasibility_solving,
                                              manual_minimal_H_eigenvalue);
}

template<typename T>
bool
operator==(const QP<T>& qp1, const QP<T>& qp2)
{
  return proxsuite::common::is_equal(qp1, qp2);
}

template<typename T>
bool
operator!=(const QP<T>& qp1, const QP<T>& qp2)
{
  return !proxsuite::common::is_equal(qp1, qp2);
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_WRAPPER_HPP */