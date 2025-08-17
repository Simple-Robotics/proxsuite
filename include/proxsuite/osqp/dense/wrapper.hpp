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
#include "proxsuite/osqp/dense/aliases.hpp"

namespace proxsuite {
namespace osqp {
namespace dense {

///
/// @brief This class defines the API of OSQP solver with dense backend.
///
template<typename T>
struct QP : public proxsuite::proxqp::dense::QP<T>
{
public:
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   * @param _dense_backend specify which factorization is used.
   */
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     HessianType _hessian_type,
     DenseBackend _dense_backend)
    : proxqp::dense::QP<T>(
        _dim,
        _n_eq,
        _n_in,
        _box_constraints,
        _hessian_type,
        proxqp::dense::dense_backend_choice<T>(_dense_backend,
                                               _dim,
                                               _n_eq,
                                               _n_in,
                                               _box_constraints))
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   * @param _dense_backend specify which factorization is used.
   */
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     DenseBackend _dense_backend,
     HessianType _hessian_type)
    : proxqp::dense::QP<T>(
        _dim,
        _n_eq,
        _n_in,
        _box_constraints,
        proxqp::dense::dense_backend_choice<T>(_dense_backend,
                                               _dim,
                                               _n_eq,
                                               _n_in,
                                               _box_constraints),
        _hessian_type)
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     HessianType _hessian_type)
    : proxqp::dense::QP<T>(_dim,
                           _n_eq,
                           _n_in,
                           _box_constraints,
                           _hessian_type,
                           proxqp::dense::dense_backend_choice<T>(
                             DenseBackend::PrimalDualLDLT,
                             // TODO: Automatic when PrimalLDLT coded
                             _dim,
                             _n_eq,
                             _n_in,
                             _box_constraints))
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   * @param _dense_backend specify which factorization is used.
   */
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     DenseBackend _dense_backend)
    : proxqp::dense::QP<T>(
        _dim,
        _n_eq,
        _n_in,
        _box_constraints,
        proxqp::dense::dense_backend_choice<T>(_dense_backend,
                                               _dim,
                                               _n_eq,
                                               _n_in,
                                               _box_constraints),
        HessianType::Dense)
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in, bool _box_constraints)
    : proxqp::dense::QP<T>(_dim,
                           _n_eq,
                           _n_in,
                           _box_constraints,
                           proxqp::dense::dense_backend_choice<T>(
                             DenseBackend::PrimalDualLDLT,
                             // TODO: Automatic when PrimalLDLT coded
                             _dim,
                             _n_eq,
                             _n_in,
                             _box_constraints),
                           HessianType::Dense)
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type specify that there are (or not) box constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in, HessianType _hessian_type)
    : proxqp::dense::QP<T>(_dim,
                           _n_eq,
                           _n_in,
                           false,
                           _hessian_type,
                           proxqp::dense::dense_backend_choice<T>(
                             DenseBackend::PrimalDualLDLT,
                             // TODO: Automatic when PrimalLDLT coded
                             _dim,
                             _n_eq,
                             _n_in,
                             false))
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in)
    : proxqp::dense::QP<T>(_dim,
                           _n_eq,
                           _n_in,
                           false,
                           HessianType::Dense,
                           proxqp::dense::dense_backend_choice<T>(
                             DenseBackend::PrimalDualLDLT,
                             // TODO: Automatic when PrimalLDLT coded
                             _dim,
                             _n_eq,
                             _n_in,
                             false))
  {
    this->work.timer.stop();
    init_osqp_settings_and_results();
  }
  /*!
   * Solves the QP problem using OSQP algorithm.
   */
  void solve()
  {
    proxsuite::osqp::dense::qp_solve( //
      this->settings,
      this->model,
      this->results,
      this->work,
      this->is_box_constrained(),
      this->which_dense_backend(),
      this->which_hessian_type(),
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
    proxsuite::osqp::dense::qp_solve( //
      this->settings,
      this->model,
      this->results,
      this->work,
      this->is_box_constrained(),
      this->which_dense_backend(),
      this->which_hessian_type(),
      this->ruiz);
  };
  /*!
   * Initializes the settings as in the source code of OSQP.
   * code: https://github.com/osqp/osqp-python
   * Commented names of settings are related to ProxQP only.
   * Mention TODO for potential improvement or future implementations.
   */
  void init_osqp_settings_and_results()
  {
    T default_mu_eq_osqp = 1e-2;
    T default_mu_in_osqp = 1e1;

    // From proxsuite/common/settings.hpp (proxsuite)
    this->settings.verbose = false;

    this->settings.default_rho = 1e-6;
    this->settings.default_mu_eq = default_mu_eq_osqp;
    this->settings.default_mu_in = default_mu_in_osqp;

    this->settings.mu_max_in_inv = 1e6;
    // TODO: this->settings.mu_min_in = 1e-6;
    // TODO: this->settings.mu_min_eq = ;
    // TODO: this->settings.mu_max_eq_inv = ;

    // TODO: this->settings.cold_reset_mu_eq = ;
    // TODO: this->settings.cold_reset_mu_in = ;
    // TODO: this->settings.cold_reset_mu_eq_inv = ;
    // TODO: this->settings.cold_reset_mu_in_inv = ;

    this->settings.eps_abs = 1e-3;
    this->settings.eps_rel = 1e-3;
    this->settings.check_duality_gap = false;
    this->settings.eps_duality_gap_abs = 1e-3;
    this->settings.eps_duality_gap_abs = 1e-3;

    this->settings.eps_primal_inf = 1e-4;
    this->settings.eps_dual_inf = 1e-4;
    this->settings.primal_infeasibility_solving = false;
    this->settings.frequence_infeasibility_check =
      1; // TODO: 25 + adaptation to source later

    this->settings.update_preconditioner = false; // TODO: Check
    this->settings.compute_preconditioner =
      true; // TODO: Check if same computation
    this->settings.preconditioner_max_iter = 10;
    this->settings.preconditioner_accuracy = 1e-3;

    this->settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    this->settings.max_iter = 4000;

    this->settings.compute_timings = true;

    this->settings.default_H_eigenvalue_estimate = 0.;

    // TODO: this->settings.sparse_backend = ;

    // max_iter_in
    // nb_iterative_refinement
    // eps_refact
    // safe_guard

    // alpha_bcl
    // beta_bcl
    // bcl_update

    // mu_update_factor
    // mu_update_inv_factor

    // refactor_dual_feasibility_threshold
    // refactor_rho_threshold

    // From osqp_api_constants.h (OSQP)
    this->settings.alpha_osqp = 1.6;

    this->settings.mu_min_in_inv = 1e-6;
    // TODO: this->settings.mu_max_in = ;
    // TODO: this->settings.mu_max_eq = ;
    // TODO: this->settings.mu_min_eq_inv = 1e-3;
    // TODO: this->settings.mu_tol = 1e-4;

    // TODO: this->settings.cg_max_iter = 20;
    // TODO: this->settings.cg_tol_reduction = 10;
    // TODO: this->settings.cg_tol_fraction = 0.15;

    this->settings.adaptive_mu = true;
    // TODO: this->settings.adaptive_mu_update_disable = false;
    // TODO: this->settings.adaptive_mu_update_kkt_error = false;
    // TODO: this->settings.adaptive_mu_update_time = false;
    // TODO: this->settings.adaptive_mu_fraction = 0.4;
    // TODO: this->settings.adaptive_mu_update_iterations = true;
    this->settings.adaptive_mu_interval = 50;
    this->settings.adaptive_mu_tolerance = 5.;
    // TODO: this->settings.adaptive_mu_multiple_termination = 4;
    // TODO: this->settings.adaptive_mu_fixed = 100;

    this->settings.polishing = false;
    this->settings.delta_osqp = 1e-6;
    this->settings.polish_refine_iter = 3;

    // TODO: this->settings.check_termination = 1; // TODO: 25 + adaptation to
    // source later

    // TODO numerics:
    // this->settings.infty
    // this->settings.division_tol

    // this->settings.min_scaling
    // this->settings.max_scaling

    // this->settings.cg_tol_min
    // this->settings.cg_polish_tol

    // this->settings.zero_deadzone

    // Results
    this->results.info.mu_eq = default_mu_eq_osqp;
    this->results.info.mu_in = default_mu_in_osqp;
    this->results.info.mu_eq_inv = T(1) / default_mu_eq_osqp;
    this->results.info.mu_in_inv = T(1) / default_mu_in_osqp;
  };
};

/*!
 * Solves the QP problem using OSQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to
 * set up some of the solver parameters (warm start, initial guess option,
 * proximal step sizes, absolute and relative accuracies, maximum number of
 * iterations, preconditioner execution). There are no box constraints in the
 * model.
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
 * @param verbose if set to true, the solver prints more information about
 * each iteration.
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
 * @param check_duality_gap If set to true, include the duality gap in
 * absolute and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 * @param adaptive_mu if set to true, perform updates of mu.
 * @param adaptive_mu_interval minimum interval between to mu update iterations.
 * @param adaptive_mu_tolerance tolerance on the ratio of residuals in mu
 * update.
 */
template<typename T>
Results<T>
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
      InitialGuessStatus initial_guess = InitialGuessStatus::NO_INITIAL_GUESS,
      bool check_duality_gap = false,
      optional<T> eps_duality_gap_abs = nullopt,
      optional<T> eps_duality_gap_rel = nullopt,
      bool primal_infeasibility_solving = false,
      optional<T> manual_minimal_H_eigenvalue = nullopt,
      optional<bool> adaptive_mu = nullopt,
      optional<isize> adaptive_mu_interval = nullopt,
      optional<T> adaptive_mu_tolerance = nullopt)
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
  if (adaptive_mu != nullopt) {
    Qp.settings.adaptive_mu = adaptive_mu.value();
  }
  if (adaptive_mu_interval != nullopt) {
    Qp.settings.adaptive_mu_interval = adaptive_mu_interval.value();
  }
  if (adaptive_mu_tolerance != nullopt) {
    Qp.settings.adaptive_mu_tolerance = adaptive_mu_tolerance.value();
  }
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
 * Solves the QP problem using OSQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to
 * set up some of the solver parameters (warm start, initial guess option,
 * proximal step sizes, absolute and relative accuracies, maximum number of
 * iterations, preconditioner execution).
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
 * @param z dual inequality constraint warm start. The upper part must contain
 * a warm start for inequality constraints wrt C matrix, whereas the latter
 * wrt the box inequalities.
 * @param verbose if set to true, the solver prints more information about
 * each iteration.
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
 * @param check_duality_gap If set to true, include the duality gap in
 * absolute and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 * @param adaptive_mu if set to true, perform updates of mu.
 * @param adaptive_mu_interval minimum interval between to mu update iterations.
 * @param adaptive_mu_tolerance tolerance on the ratio of residuals in mu
 * update.
 */
template<typename T>
Results<T>
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
      InitialGuessStatus initial_guess = InitialGuessStatus::NO_INITIAL_GUESS,
      bool check_duality_gap = false,
      optional<T> eps_duality_gap_abs = nullopt,
      optional<T> eps_duality_gap_rel = nullopt,
      bool primal_infeasibility_solving = false,
      optional<T> manual_minimal_H_eigenvalue = nullopt,
      optional<bool> adaptive_mu = nullopt,
      optional<isize> adaptive_mu_interval = nullopt,
      optional<T> adaptive_mu_tolerance = nullopt)
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
  if (adaptive_mu != nullopt) {
    Qp.settings.adaptive_mu = adaptive_mu.value();
  }
  if (adaptive_mu_interval != nullopt) {
    Qp.settings.adaptive_mu_interval = adaptive_mu_interval.value();
  }
  if (adaptive_mu_tolerance != nullopt) {
    Qp.settings.adaptive_mu_tolerance = adaptive_mu_tolerance.value();
  }
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

template<typename T>
bool
operator==(const QP<T>& qp1, const QP<T>& qp2)
{
  bool value = qp1.model == qp2.model && qp1.settings == qp2.settings &&
               qp1.results == qp2.results &&
               qp1.is_box_constrained() == qp2.is_box_constrained();
  return value;
}

template<typename T>
bool
operator!=(const QP<T>& qp1, const QP<T>& qp2)
{
  return !(qp1 == qp2);
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_WRAPPER_HPP */