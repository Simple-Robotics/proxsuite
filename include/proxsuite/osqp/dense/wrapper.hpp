//
// Copyright (c) 2022 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_WRAPPER_HPP
#define PROXSUITE_OSQP_DENSE_WRAPPER_HPP

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/osqp/dense/solver.hpp>

namespace proxsuite {
namespace osqp {
namespace dense {
///
/// @brief This class defines the API of OSQP solver (same as PROXQP but with
/// proper solve) with dense backend.
///

namespace ppd = proxsuite::proxqp::dense;
using namespace ppd;

template<typename T>
struct QP : public ppd::QP<T>
{
public:
  using ppd::QP<T>::QP;
  /*!
   * Solves the QP problem using OSQP algorithm.
   */
  void solve()
  {
    proxsuite::osqp::dense::qp_solve(this->settings,
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
    warm_start(x, y, z, this->results, this->settings, this->model);
    proxsuite::osqp::dense::qp_solve(this->settings,
                                     this->model,
                                     this->results,
                                     this->work,
                                     this->get_box_constraints(),
                                     this->get_dense_backend(),
                                     this->get_hessian_type(),
                                     this->ruiz);
  };
};

// TODO: The init and init_settings functions are copy pasted from the ones in
// proxqp/dense/wrapper The goal is to keep them in the later only (to
// refactorize the code) When I try it, I get the following error: no viable
// conversion from 'QP<double>' to 'QP<double>' in the line: QP<T> Qp = init(H,
// .....

template<typename T>
QP<T>
init_settings(
  optional<MatRef<T>> H,
  optional<MatRef<T>> A,
  optional<MatRef<T>> C,
  optional<T> eps_abs = nullopt,
  optional<T> eps_rel = nullopt,
  optional<bool> verbose = nullopt,
  bool compute_timings = false,
  optional<isize> max_iter = nullopt,
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false)
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

  QP<T> Qp(
    n, n_eq, n_in, false, proxsuite::proxqp::DenseBackend::PrimalDualLDLT);
  Qp.settings.initial_guess = initial_guess;
  Qp.settings.check_duality_gap = check_duality_gap;

  if (eps_abs != nullopt) {
    Qp.settings.eps_abs = eps_abs.value();
  }
  if (eps_rel != nullopt) {
    Qp.settings.eps_rel = eps_rel.value();
  }
  if (verbose != nullopt) {
    Qp.settings.verbose = verbose.value();
  }
  if (max_iter != nullopt) {
    Qp.settings.max_iter = max_iter.value();
  }
  if (eps_duality_gap_abs != nullopt) {
    Qp.settings.eps_duality_gap_abs = eps_duality_gap_abs.value();
  }
  if (eps_duality_gap_rel != nullopt) {
    Qp.settings.eps_duality_gap_rel = eps_duality_gap_rel.value();
  }
  Qp.settings.compute_timings = compute_timings;
  Qp.settings.primal_infeasibility_solving = primal_infeasibility_solving;

  return Qp;
}

template<typename T>
QP<T>
init(
  optional<MatRef<T>> H,
  optional<VecRef<T>> g,
  optional<MatRef<T>> A,
  optional<VecRef<T>> b,
  optional<MatRef<T>> C,
  optional<VecRef<T>> l,
  optional<VecRef<T>> u,
  optional<T> eps_abs = nullopt,
  optional<T> eps_rel = nullopt,
  optional<T> rho = nullopt,
  optional<T> mu_eq = nullopt,
  optional<T> mu_in = nullopt,
  optional<bool> verbose = nullopt,
  bool compute_preconditioner = true,
  bool compute_timings = false,
  optional<isize> max_iter = nullopt,
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false,
  optional<T> manual_minimal_H_eigenvalue = nullopt)
{

  QP<T> Qp = init_settings(H,
                           A,
                           C,
                           eps_abs,
                           eps_rel,
                           verbose,
                           compute_timings,
                           max_iter,
                           initial_guess,
                           check_duality_gap,
                           eps_duality_gap_abs,
                           eps_duality_gap_rel,
                           primal_infeasibility_solving);

  if (manual_minimal_H_eigenvalue != nullopt) {
    Qp.init(H,
            g,
            A,
            b,
            C,
            l,
            u,
            compute_preconditioner,
            rho,
            mu_eq,
            mu_in,
            manual_minimal_H_eigenvalue.value());
  } else {
    Qp.init(
      H, g, A, b, C, l, u, compute_preconditioner, rho, mu_eq, mu_in, nullopt);
  }

  return Qp;
}

template<typename T>
QP<T>
init(
  optional<MatRef<T>> H,
  optional<VecRef<T>> g,
  optional<MatRef<T>> A,
  optional<VecRef<T>> b,
  optional<MatRef<T>> C,
  optional<VecRef<T>> l,
  optional<VecRef<T>> u,
  optional<VecRef<T>> l_box,
  optional<VecRef<T>> u_box,
  optional<T> eps_abs = nullopt,
  optional<T> eps_rel = nullopt,
  optional<T> rho = nullopt,
  optional<T> mu_eq = nullopt,
  optional<T> mu_in = nullopt,
  optional<bool> verbose = nullopt,
  bool compute_preconditioner = true,
  bool compute_timings = false,
  optional<isize> max_iter = nullopt,
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false,
  optional<T> manual_minimal_H_eigenvalue = nullopt)
{

  QP<T> Qp = init_settings(H,
                           A,
                           C,
                           eps_abs,
                           eps_rel,
                           verbose,
                           compute_timings,
                           max_iter,
                           initial_guess,
                           check_duality_gap,
                           eps_duality_gap_abs,
                           eps_duality_gap_rel,
                           primal_infeasibility_solving);

  if (manual_minimal_H_eigenvalue != nullopt) {
    Qp.init(H,
            g,
            A,
            b,
            C,
            l,
            u,
            l_box,
            u_box,
            compute_preconditioner,
            rho,
            mu_eq,
            mu_in,
            manual_minimal_H_eigenvalue.value());
  } else {
    Qp.init(H,
            g,
            A,
            b,
            C,
            l,
            u,
            l_box,
            u_box,
            compute_preconditioner,
            rho,
            mu_eq,
            mu_in,
            nullopt);
  }

  return Qp;
}

template<typename T>
proxqp::Results<T>
solve(
  optional<MatRef<T>> H,
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
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false,
  optional<T> manual_minimal_H_eigenvalue = nullopt)
{
  QP<T> Qp = init(H,
                  g,
                  A,
                  b,
                  C,
                  l,
                  u,
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

  Qp.solve(x, y, z);

  return Qp.results;
}

template<typename T>
proxqp::Results<T>
solve(
  optional<MatRef<T>> H,
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
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false,
  optional<T> manual_minimal_H_eigenvalue = nullopt)
{
  QP<T> Qp = init(H,
                  g,
                  A,
                  b,
                  C,
                  l,
                  u,
                  l_box,
                  u_box,
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

  Qp.solve(x, y, z);

  return Qp.results;
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_WRAPPER_HPP */

// TODO: See if it is needed to add / adapt the last functions of
// proxqp/dense/wrapper.hpp (after the solve without api)