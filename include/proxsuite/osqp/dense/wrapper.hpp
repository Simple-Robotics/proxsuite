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
/// @brief This class defines the API of OSQP solver (same as PROXQP but with proper solve) with dense backend.
///

template<typename T>
struct QP : public proxsuite::proxqp::dense::QP<T> 
{
public:
    using proxsuite::proxqp::dense::QP<T>::QP;
    /*!
    * Solves the QP problem using OSQP algorithm.
    */
    void solve()
    {
        proxsuite::osqp::dense::qp_solve( // To avoid overload
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
    void solve(optional<proxsuite::proxqp::dense::VecRef<T>> x,
                optional<proxsuite::proxqp::dense::VecRef<T>> y,
                optional<proxsuite::proxqp::dense::VecRef<T>> z)
    {
        proxsuite::proxqp::dense::warm_start(x, y, z, this->results, this->settings, this->model);
        proxsuite::osqp::dense::qp_solve( // To avoid overload
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

// Here is the definition of the functions to solve the problem without directly creating a QP object
// via the API. 
// !!! This is a (more or less) copy paste from the proxqp wrapper -> change it soon //

using namespace proxsuite::proxqp::dense;

/*!
 * Solves the QP problem using PROXQP algorithm without the need to define a QP
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

  QP<T> Qp(n, n_eq, n_in, false, proxsuite::proxqp::DenseBackend::PrimalDualLDLT);
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

  QP<T> Qp(n, n_eq, n_in, true, proxsuite::proxqp::DenseBackend::PrimalDualLDLT);
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

// TODO: See how to use the two solve() functions from proxqp/dense/wrapper.hpp that allow to solve the 
// problem without the API (in particular their last call 'Qp.solve(x, y, z);')
// => Do a function that just changes the last solve call (depending on the solver) ?
/// !!! Do not forget, because it is a copy-paste (-> code duplication) from the proxqp wrapper

// TODO: Make the notations of the namespaces lighter
// => Do a using namespace proxsuite::proxqp ? 

} // namespace dense
} // namespace osqp
} // namespace proxsuite


#endif /* end of include guard PROXSUITE_OSQP_DENSE_WRAPPER_HPP */