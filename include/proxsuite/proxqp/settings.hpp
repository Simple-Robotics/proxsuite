//
// Copyright (c) 2022 INRIA
//
/**
 * @file settings.hpp
 */
#ifndef PROXSUITE_PROXQP_SETTINGS_HPP
#define PROXSUITE_PROXQP_SETTINGS_HPP

#include <Eigen/Core>
#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/proxqp/dense/views.hpp>
#include <proxsuite/proxqp/sparse/fwd.hpp>

namespace proxsuite {
namespace proxqp {
///
/// @brief This class defines the settings of PROXQP solvers with sparse and
/// dense backends.
///
/*!
 * Settings class, which defines the parameters used by the dense and sparse
 * solver (and its preconditioner).
 */
template<typename T>
struct Settings
{

  T default_rho;
  T default_mu_eq;
  T default_mu_in;

  T alpha_bcl;
  T beta_bcl;

  T refactor_dual_feasibility_threshold;
  T refactor_rho_threshold;

  T mu_min_eq;
  T mu_min_in;
  T mu_max_eq_inv;
  T mu_max_in_inv;

  T mu_update_factor;
  T mu_update_inv_factor;

  T cold_reset_mu_eq;
  T cold_reset_mu_in;
  T cold_reset_mu_eq_inv;
  T cold_reset_mu_in_inv;
  T eps_abs;
  T eps_rel;

  isize max_iter;
  isize max_iter_in;
  isize safe_guard;
  isize nb_iterative_refinement;
  T eps_refact;

  bool verbose;
  InitialGuessStatus initial_guess;
  bool update_preconditioner;
  bool compute_preconditioner;
  bool compute_timings;

  isize preconditioner_max_iter;
  T preconditioner_accuracy;
  T eps_primal_inf;
  T eps_dual_inf;
  bool bcl_update;
  /*!
   * Default constructor.
   * @param default_rho default rho parameter of result class
   * @param default_mu_eq default mu_eq parameter of result class
   * @param default_mu_in default mu_in parameter of result class
   * @param alpha_bcl alpha parameter of the BCL algorithm.
   * @param beta_bcl beta parameter of the BCL algorithm.
   * @param refactor_dual_feasibility_threshold threshold above which
   * refactorization is performed to change rho parameter.
   * @param refactor_rho_threshold new rho parameter used if the
   * refactor_dual_feasibility_threshold_ condition has been satisfied.
   * @param mu_min_eq minimal authorized value for mu_eq.
   * @param mu_min_in minimal authorized value for mu_in.
   * @param mu_max_eq_inv maximal authorized value for the inverse of
   * mu_eq_inv.
   * @param mu_max_in_inv maximal authorized value for the inverse of
   * mu_in_inv.
   * @param mu_update_factor update factor used for updating mu_eq and mu_in.
   * @param mu_update_inv_factor update factor used for updating mu_eq_inv and
   * mu_in_inv.
   * @param cold_reset_mu_eq value used for cold restarting mu_eq.
   * @param cold_reset_mu_in value used for cold restarting mu_in.
   * @param cold_reset_mu_eq_inv value used for cold restarting mu_eq_inv.
   * @param cold_reset_mu_in_inv value used for cold restarting mu_in_inv.
   * @param eps_abs asbolute stopping criterion of the solver.
   * @param eps_rel relative stopping criterion of the solver.
   * @param max_iter maximal number of authorized iteration.
   * @param max_iter_in maximal number of authorized iterations for an inner
   * loop.
   * @param nb_iterative_refinement number of iterative refinements.
   * @param eps_refact threshold value for refactorizing the ldlt factorization
   * in the iterative refinement loop.
   * @param safe_guard safeguard parameter ensuring global convergence of ProxQP
   * scheme.
   * @param VERBOSE if set to true, the solver prints information at each loop.
   * @param initial_guess sets the initial guess option for initilizing x, y
   * and z.
   * @param update_preconditioner If set to true, the preconditioner will be
   * re-derived with the update method.
   * @param compute_preconditioner If set to true, the preconditioner will be
   * derived with the init method.
   * @param compute_timings If set to true, timings will be computed by the
   * solver (setup time, solving time, and run time = setup time + solving
   * time).
   * @param preconditioner_max_iter maximal number of authorized iterations for
   * the preconditioner.
   * @param preconditioner_accuracy accuracy level of the preconditioner.
   * @param eps_primal_inf threshold under which primal infeasibility is
   * detected.
   * @param eps_dual_inf threshold under which dual infeasibility is detected.
   * @param bcl_update if set to true, BCL strategy is used for calibrating
   * mu_eq and mu_in. If set to false, a strategy developped by Martinez & al is
   * used.
   */

  Settings(
    T default_rho = 1.E-6,
    T default_mu_eq = 1.E-3,
    T default_mu_in = 1.E-1,
    T alpha_bcl = 0.1,
    T beta_bcl = 0.9,
    T refactor_dual_feasibility_threshold = 1e-2,
    T refactor_rho_threshold = 1e-7,
    T mu_min_eq = 1e-9,
    T mu_min_in = 1e-8,
    T mu_max_eq_inv = 1e9,
    T mu_max_in_inv = 1e8,
    T mu_update_factor = 0.1,
    T mu_update_inv_factor = 10,
    T cold_reset_mu_eq = 1. / 1.1,
    T cold_reset_mu_in = 1. / 1.1,
    T cold_reset_mu_eq_inv = 1.1,
    T cold_reset_mu_in_inv = 1.1,
    T eps_abs = 1.e-5,
    T eps_rel = 0,
    isize max_iter = 10000,
    isize max_iter_in = 1500,
    isize safe_guard = 1.E4,
    isize nb_iterative_refinement = 10,
    T eps_refact = 1.e-6, // before eps_refact_=1.e-6
    bool verbose = false,
    InitialGuessStatus initial_guess = InitialGuessStatus::
      EQUALITY_CONSTRAINED_INITIAL_GUESS, // default to
                                          // EQUALITY_CONSTRAINED_INITIAL_GUESS,
                                          // as most often we run only
                                          // once a problem
    bool update_preconditioner = true,
    bool compute_preconditioner = true,
    bool compute_timings = true,
    isize preconditioner_max_iter = 10,
    T preconditioner_accuracy = 1.e-3,
    T eps_primal_inf = 1.E-4,
    T eps_dual_inf = 1.E-4,
    bool bcl_update = true)
    : default_rho(default_rho)
    , default_mu_eq(default_mu_eq)
    , default_mu_in(default_mu_in)
    , alpha_bcl(alpha_bcl)
    , beta_bcl(beta_bcl)
    , refactor_dual_feasibility_threshold(refactor_dual_feasibility_threshold)
    , refactor_rho_threshold(refactor_rho_threshold)
    , mu_min_eq(mu_min_eq)
    , mu_min_in(mu_min_in)
    , mu_max_eq_inv(mu_max_eq_inv)
    , mu_max_in_inv(mu_max_in_inv)
    , mu_update_factor(mu_update_factor)
    , mu_update_inv_factor(mu_update_inv_factor)
    , cold_reset_mu_eq(cold_reset_mu_eq)
    , cold_reset_mu_in(cold_reset_mu_in)
    , cold_reset_mu_eq_inv(cold_reset_mu_eq_inv)
    , cold_reset_mu_in_inv(cold_reset_mu_in_inv)
    , eps_abs(eps_abs)
    , eps_rel(eps_rel)
    , max_iter(max_iter)
    , max_iter_in(max_iter_in)
    , safe_guard(safe_guard)
    , nb_iterative_refinement(nb_iterative_refinement)
    , eps_refact(eps_refact)
    , verbose(verbose)
    , initial_guess(initial_guess)
    , update_preconditioner(update_preconditioner)
    , compute_preconditioner(compute_preconditioner)
    , compute_timings(compute_timings)
    , preconditioner_max_iter(preconditioner_max_iter)
    , preconditioner_accuracy(preconditioner_accuracy)
    , eps_primal_inf(eps_primal_inf)
    , eps_dual_inf(eps_dual_inf)
    , bcl_update(bcl_update)
  {
  }
  /*
 void set(
           T alpha_bcl_ = 0.1,
           T beta_bcl_ = 0.9,
           T refactor_dual_feasibility_threshold_ = 1e-2,
           T refactor_rho_threshold_ = 1e-7,
           T mu_min_eq_ = 1e-9,
           T mu_min_in_ = 1e-8,
           T mu_max_eq_inv_ = 1e9,
           T mu_max_in_inv_ = 1e8,
           T mu_update_factor_ = 0.1,
           T mu_update_inv_factor_ = 10,
           T cold_reset_mu_eq_ = 1. / 1.1,
           T cold_reset_mu_in_ = 1. / 1.1,
           T cold_reset_mu_eq_inv_ = 1.1,
           T cold_reset_mu_in_inv_ = 1.1,
           T eps_abs_ = 1.e-8,
           T eps_rel_ = 0,
           isize max_iter_ = 10000,
           isize max_iter_in_ = 1500,
           isize safe_guard_ = 1.E4,
           isize nb_iterative_refinement_ = 10,
           T eps_refact_ = 1.e-6, // before eps_refact_=1.e-6
           bool VERBOSE = false,
           InitialGuessStatus initial_guess_ =
             InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT,
           bool update_preconditioner_ = true,
           bool compute_preconditioner_ = true,
           bool compute_timings_ = true,
           isize preconditioner_max_iter_ = 10,
           T preconditioner_accuracy_ = 1.e-3,
           T eps_primal_inf_ = 1.E-4,
           T eps_dual_inf_ = 1.E-4,
           bool bcl_update_ = true
 ){
    alpha_bcl = alpha_bcl_;
    beta_bcl = beta_bcl_ ;
    refactor_dual_feasibility_threshold = refactor_dual_feasibility_threshold_;
    refactor_rho_threshold = refactor_rho_threshold_;
    mu_min_eq = mu_min_eq_;
    mu_min_in = mu_min_in_;
    mu_max_eq_inv = mu_max_eq_inv_;
    mu_max_in_inv = mu_max_in_inv_;
    mu_update_factor = mu_update_factor_;
    mu_update_inv_factor = mu_update_inv_factor_;
    cold_reset_mu_eq = cold_reset_mu_eq_;
    cold_reset_mu_in = cold_reset_mu_in_;
    cold_reset_mu_eq_inv = cold_reset_mu_eq_inv_;
    cold_reset_mu_in_inv = cold_reset_mu_in_inv_;
    eps_abs = eps_abs_;
    eps_rel = eps_rel_;
    max_iter = max_iter_;
    max_iter_in = max_iter_in_;
    safe_guard = safe_guard_;
    nb_iterative_refinement = nb_iterative_refinement_;
    eps_refact = eps_refact_;
    verbose = VERBOSE;
    initial_guess = initial_guess_;
    update_preconditioner = update_preconditioner_;
    compute_preconditioner = compute_preconditioner_;
    compute_timings = compute_timings_;
    preconditioner_max_iter = preconditioner_max_iter_;
    preconditioner_accuracy = preconditioner_accuracy_;
    eps_primal_inf = eps_primal_inf_;
    eps_dual_inf = eps_dual_inf_;
    bcl_update = bcl_update_;

 }
 */
};

} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_SETTINGS_HPP */
