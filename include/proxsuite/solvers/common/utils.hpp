//
// Copyright (c) 2025 INRIA
//
/**
 * @file utils.hpp
 */

#ifndef PROXSUITE_SOLVERS_COMMON_UTILS_HPP
#define PROXSUITE_SOLVERS_COMMON_UTILS_HPP

#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
#include "proxsuite/linalg/veg/internal/typedefs.hpp"

namespace proxsuite {
namespace common {

namespace ppd = proxsuite::proxqp::dense;
namespace plv = proxsuite::linalg::veg;

/*!
 * Generic function to solve the QP. Used in the functions solve() to solve the
 * the problem without defining the API. There are no box constraints in the
 * model.
 * @param Qp QP object on which the problem is solved.
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
template<typename T, typename QPStruct>
proxqp::Results<T>
solve_without_api(QPStruct& Qp,
                  optional<ppd::MatRef<T>> H,
                  optional<ppd::VecRef<T>> g,
                  optional<ppd::MatRef<T>> A,
                  optional<ppd::VecRef<T>> b,
                  optional<ppd::MatRef<T>> C,
                  optional<ppd::VecRef<T>> l,
                  optional<ppd::VecRef<T>> u,
                  optional<ppd::VecRef<T>> x,
                  optional<ppd::VecRef<T>> y,
                  optional<ppd::VecRef<T>> z,
                  optional<T> eps_abs,
                  optional<T> eps_rel,
                  optional<T> rho,
                  optional<T> mu_eq,
                  optional<T> mu_in,
                  optional<bool> verbose,
                  bool compute_preconditioner,
                  bool compute_timings,
                  optional<plv::isize> max_iter,
                  proxsuite::proxqp::InitialGuessStatus initial_guess,
                  bool check_duality_gap,
                  optional<T> eps_duality_gap_abs,
                  optional<T> eps_duality_gap_rel,
                  bool primal_infeasibility_solving,
                  optional<T> manual_minimal_H_eigenvalue)
{
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
  Qp.solve(x, y, z);

  return Qp.results;
}
/*!
 * Generic function to solve the QP. Used in the functions solve() to solve the
 * the problem without defining the API. There are box constraints in the model.
 * @param Qp QP object on which the problem is solved.
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
template<typename T, typename QPStruct>
proxqp::Results<T>
solve_without_api(QPStruct& Qp,
                  optional<ppd::MatRef<T>> H,
                  optional<ppd::VecRef<T>> g,
                  optional<ppd::MatRef<T>> A,
                  optional<ppd::VecRef<T>> b,
                  optional<ppd::MatRef<T>> C,
                  optional<ppd::VecRef<T>> l,
                  optional<ppd::VecRef<T>> u,
                  optional<ppd::VecRef<T>> l_box,
                  optional<ppd::VecRef<T>> u_box,
                  optional<ppd::VecRef<T>> x,
                  optional<ppd::VecRef<T>> y,
                  optional<ppd::VecRef<T>> z,
                  optional<T> eps_abs,
                  optional<T> eps_rel,
                  optional<T> rho,
                  optional<T> mu_eq,
                  optional<T> mu_in,
                  optional<bool> verbose,
                  bool compute_preconditioner,
                  bool compute_timings,
                  optional<plv::isize> max_iter,
                  proxsuite::proxqp::InitialGuessStatus initial_guess,
                  bool check_duality_gap,
                  optional<T> eps_duality_gap_abs,
                  optional<T> eps_duality_gap_rel,
                  bool primal_infeasibility_solving,
                  optional<T> manual_minimal_H_eigenvalue)
{
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
  Qp.solve(x, y, z);

  return Qp.results;
}
/*!
 * Generic function to test wether two QP objects are equal.
 * @param qp1 First QP object.
 * @param qp2 Second QP object.
 */
template<typename QPStruct>
bool
is_equal(const QPStruct& qp1, const QPStruct& qp2)
{
  bool value = qp1.model == qp2.model && qp1.settings == qp2.settings &&
               qp1.results == qp2.results &&
               qp1.is_box_constrained() == qp2.is_box_constrained();
  return value;
}

} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_SOLVERS_COMMON_UTILS_HPP */