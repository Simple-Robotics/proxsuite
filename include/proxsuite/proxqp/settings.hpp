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

// Sparse backend specifications
enum struct SparseBackend
{
  Automatic,      // the solver will select the appropriate sparse backend.
  SparseCholesky, // sparse cholesky backend.
  MatrixFree,     // iterative matrix free sparse backend.
};
// Sparse backend specifications
enum struct DenseBackend
{
  Automatic,      // the solver will select the appropriate dense backend.
  PrimalDualLDLT, // Factorization of the full regularized KKT matrix.
  PrimalLDLT, // Factorize only the primal Hessian corresponding to H+rho I +
              // 1/mu AT*A.
};
// MERIT FUNCTION
enum struct MeritFunctionType
{
  GPDAL, // Generalized Primal Dual Augmented Lagrangian
  PDAL,  // Primal Dual Augmented Lagrangian
};
// COST FUNCTION TYPE
enum struct HessianType
{
  Zero,    // Linear Program
  Dense,   // Quadratic Program
  Diagonal // Quadratic Program with diagonal Hessian
};
// Augment the minimal eigen value of H with some regularizer
enum struct EigenValueEstimateMethodOption
{
  PowerIteration, // Process a power iteration algorithm
  ExactMethod     // Use Eigen's method to estimate the minimal eigenvalue
                  // watch out, the last option is only available for dense
                  // matrices!
};
// Choice of approach for the iteraion condition to update mu in OSQP.
enum struct UpdateMuIterationCriteria
{
  FactorizationTime,    // Time spent since last update wrt factorization time.
  FixedNumberIterations // Fixed number of iteration to outpass.
};
inline std::ostream&
operator<<(std::ostream& os, const SparseBackend& sparse_backend)
{
  if (sparse_backend == SparseBackend::Automatic)
    os << "Automatic";
  else if (sparse_backend == SparseBackend::SparseCholesky) {
    os << "SparseCholesky";
  } else {
    os << "MatrixFree";
  }
  return os;
}
inline std::ostream&
operator<<(std::ostream& os, const DenseBackend& dense_backend)
{
  if (dense_backend == DenseBackend::PrimalDualLDLT) {
    os << "PrimalDualLDLT";
  } else if (dense_backend == DenseBackend::PrimalLDLT) {
    os << "PrimalLDLT";
  } else {
    os << "Automatic";
  }
  return os;
}

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

  T alpha_osqp;

  T refactor_dual_feasibility_threshold;
  T refactor_rho_threshold;

  T mu_min_eq;
  T mu_max_eq; // for osqp
  T mu_min_in; // osqp 1e-6
  T mu_max_in; // for osqp

  T mu_max_eq_inv;
  T mu_min_eq_inv; // for osqp
  T mu_max_in_inv; // osqp 1e6
  T mu_min_in_inv; // for osqp

  T mu_min_in_osqp;
  T mu_max_in_inv_osqp;

  T mu_update_factor;
  T mu_update_inv_factor;

  T cold_reset_mu_eq;
  T cold_reset_mu_in;
  T cold_reset_mu_eq_inv;
  T cold_reset_mu_in_inv;

  T cold_reset_mu_eq_osqp;
  T cold_reset_mu_in_osqp;
  T cold_reset_mu_eq_inv_osqp;
  T cold_reset_mu_in_inv_osqp;

  T eps_abs;
  T eps_rel;

  bool high_accuracy;

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

  bool check_duality_gap;
  T eps_duality_gap_abs;
  T eps_duality_gap_rel;

  bool update_mu;
  T threshold_ratio_update_mu;
  T threshold_ratio_update_mu_inv;
  T percentage_factorization_time_update_mu;
  UpdateMuIterationCriteria update_mu_iteration_criteria;
  isize interval_update_mu;

  bool polish;
  isize polish_refine_iter;
  T delta;
  bool resume_admm;

  isize preconditioner_max_iter;
  T preconditioner_accuracy;
  T eps_primal_inf;
  T eps_dual_inf;
  bool bcl_update;
  MeritFunctionType merit_function_type;
  T alpha_gpdal;

  SparseBackend sparse_backend;
  bool primal_infeasibility_solving;
  isize frequence_infeasibility_check;
  T default_H_eigenvalue_estimate;
  /*!
   * Default constructor.
   * @param default_rho default rho parameter of result class
   * @param default_mu_eq default mu_eq parameter of result class
   * @param default_mu_in default mu_in parameter of result class
   * @param alpha_bcl alpha parameter of the BCL algorithm.
   * @param beta_bcl beta parameter of the BCL algorithm.
   * @param alpha_osqp alpha parameter in the ADMM step in OSQP.
   * @param refactor_dual_feasibility_threshold threshold above which
   * refactorization is performed to change rho parameter.
   * @param refactor_rho_threshold new rho parameter used if the
   * refactor_dual_feasibility_threshold_ condition has been satisfied.
   * @param mu_min_eq minimal authorized value for mu_eq.
   * @param mu_max_eq maximal authorized value for mu_eq.
   * @param mu_min_in minimal authorized value for mu_in.
   * @param mu_max_in maximal authorized value for mu_in.
   * @param mu_max_eq_inv maximal authorized value for the inverse of
   * mu_eq_inv.
   * @param mu_min_eq_inv minimal authorized value for the inverse of
   * mu_eq_inv.
   * @param mu_max_in_inv maximal authorized value for the inverse of
   * mu_in_inv.
   * @param mu_min_in_inv minimal authorized value for the inverse of
   * mu_in_inv.
   * @param mu_min_in_osqp minimal authorized value for mu_in in osqp.
   * @param mu_max_in_inv_osqp maximal authorized value for the inverse of mu_in
   * in osqp.
   * @param mu_update_factor update factor used for updating mu_eq and mu_in.
   * @param mu_update_inv_factor update factor used for updating mu_eq_inv and
   * mu_in_inv.
   * @param cold_reset_mu_eq value used for cold restarting mu_eq.
   * @param cold_reset_mu_in value used for cold restarting mu_in.
   * @param cold_reset_mu_eq_inv value used for cold restarting mu_eq_inv.
   * @param cold_reset_mu_in_inv value used for cold restarting mu_in_inv.
   * @param cold_reset_mu_eq_osqp value used for cold restarting mu_eq in osqp.
   * @param cold_reset_mu_in_osqp value used for cold restarting mu_in in osqp.
   * @param cold_reset_mu_eq_inv_osqp value used for cold restarting mu_eq_inv
   * in osqp.
   * @param cold_reset_mu_in_inv_osqp value used for cold restarting mu_in_inv
   * in osqp.
   * @param eps_abs asbolute stopping criterion of the solver.
   * @param eps_rel relative stopping criterion of the solver.
   * @param high_accuracy if set to true, epsilon for ADMM set to 1e-5 instead
   * of 1e-3 in osqp.
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
   * re-computed when calling the update method.
   * @param compute_preconditioner If set to true, the preconditioner will be
   * computed with the init method.
   * @param compute_timings If set to true, timings in microseconds will be
   * computed by the solver (setup time, solving time, and run time = setup time
   * + solving time).
   * @param check_duality_gap If set to true, duality gap will be calculated and
   * included in the stopping criterion.
   * @param eps_duality_gap_abs absolute duality-gap stopping criterion.
   * @param eps_duality_gap_rel relative duality-gap stopping criterion.
   * @param update_mu If set to true, the proximal parameters in OSQP are
   * updated during the solve.
   * @param threshold_ratio_update_mu update mu if below the scaled ratio
   * between primal and dual residuals.
   * @param threshold_ratio_update_mu_inv update mu if above the scaled ratio
   * between primal and dual residuals.
   * @param percentage_factorization_time_update_mu percentage of the
   * factorization time of the complete KKT matrix in OSQP involved in the
   * update of mu.
   * @param update_mu_iteration_criteria choose wether we potnetially update mu
   * after a fixed amount of iterations or some percentage of factorization
   * time.
   * @param interval_update_mu minimum number of ADMM iterations between two mu
   * updates in OSQP if iteration based criteria
   * @param polish if set to true, performs solution polishing in OSQP
   * @param polish_refine_iter number of iterative refinements in polishing in
   * OSQP
   * @param delta regularization parameter in solution polishing in OSQP
   * @param resume_admm if set to true, resumes to ADMM iterations after polish
   * failed to found active sets or provide a better solution.
   * @param preconditioner_max_iter maximal number of authorized iterations for
   * the preconditioner.
   * @param preconditioner_accuracy accuracy level of the preconditioner.
   * @param eps_primal_inf threshold under which primal infeasibility is
   * detected.
   * @param eps_dual_inf threshold under which dual infeasibility is detected.
   * @param bcl_update if set to true, BCL strategy is used for calibrating
   * mu_eq and mu_in. If set to false, a strategy developped by Martinez & al is
   * used.
   * @param sparse_backend Default automatic. User can choose between sparse
   * cholesky or iterative matrix free sparse backend.
   * @param primal_infeasibility_solving solves the closest primal feasible
   * problem if activated
   * @param frequence_infeasibility_check frequence at which infeasibility is
   * checked
   * @param find_H_minimal_eigenvalue track the minimal eigen value of the
   * quadratic cost H
   * @param default_H_eigenvalue_estimate default H eigenvalue estimate (i.e.,
   * if we make a model update and H does not change this one is used)
   */

  Settings(
    DenseBackend dense_backend = DenseBackend::PrimalDualLDLT,
    T default_mu_eq = 1.E-3,
    T default_mu_in = 1.E-1,
    T alpha_bcl = 0.1,
    T beta_bcl = 0.9,
    T alpha_osqp = 1.6,
    T refactor_dual_feasibility_threshold = 1e-2,
    T refactor_rho_threshold = 1e-7,
    T mu_min_eq = 1e-9,
    T mu_max_eq = 1e3, // for osqp
    T mu_min_in = 1e-8,
    T mu_max_in = 1e6, // for osqp
    T mu_max_eq_inv = 1e9,
    T mu_min_eq_inv = 1e-3, // for osqp
    T mu_max_in_inv = 1e8,
    T mu_min_in_inv = 1e-6, // for osqp
    T mu_min_in_osqp = 1e-6,
    T mu_max_in_inv_osqp = 1e6,
    T mu_update_factor = 0.1,
    T mu_update_inv_factor = 10,
    T cold_reset_mu_eq = 1. / 1.1,
    T cold_reset_mu_in = 1. / 1.1,
    T cold_reset_mu_eq_inv = 1.1,
    T cold_reset_mu_in_inv = 1.1,
    T cold_reset_mu_eq_osqp =
      1.E-2, // TODO: tune (given scenari, algo, experiments)
    T cold_reset_mu_in_osqp = 1.E1,      // idem
    T cold_reset_mu_eq_inv_osqp = 1.E2,  // idem
    T cold_reset_mu_in_inv_osqp = 1.E-1, // idem
    T eps_abs = 1.e-5,
    T eps_rel = 0,
    bool high_accuracy = false,
    isize max_iter = 10000,
    isize max_iter_in = 1500,
    isize safe_guard = 1.E4,
    isize nb_iterative_refinement = 10,
    T eps_refact = 1.e-6, // before eps_refact_=1.e-6
    bool verbose = true,
    InitialGuessStatus initial_guess = InitialGuessStatus::
      EQUALITY_CONSTRAINED_INITIAL_GUESS, // default to
                                          // EQUALITY_CONSTRAINED_INITIAL_GUESS,
                                          // as most often we run only
                                          // once a problem
    bool update_preconditioner = false,
    bool compute_preconditioner = true,
    bool compute_timings = false,
    bool check_duality_gap = false,
    T eps_duality_gap_abs = 1.e-4,
    T eps_duality_gap_rel = 0,
    bool update_mu = true,
    T threshold_ratio_update_mu = 5.0,
    T threshold_ratio_update_mu_inv = 0.2,
    T percentage_factorization_time_update_mu = 0.4,
    UpdateMuIterationCriteria update_mu_iteration_criteria =
      UpdateMuIterationCriteria::FixedNumberIterations,
    isize interval_update_mu = 50,
    bool polish = true,
    isize polish_refine_iter = 3,
    T delta = 1.e-6,
    bool resume_admm = true,
    isize preconditioner_max_iter = 10,
    T preconditioner_accuracy = 1.e-3,
    T eps_primal_inf = 1.E-4,
    T eps_dual_inf = 1.E-4,
    bool bcl_update = true,
    MeritFunctionType merit_function_type = MeritFunctionType::GPDAL,
    T alpha_gpdal = 0.95,
    SparseBackend sparse_backend = SparseBackend::Automatic,
    bool primal_infeasibility_solving = false,
    isize frequence_infeasibility_check = 1,
    T default_H_eigenvalue_estimate = 0.)
    : default_mu_eq(default_mu_eq)
    , default_mu_in(default_mu_in)
    , alpha_bcl(alpha_bcl)
    , beta_bcl(beta_bcl)
    , alpha_osqp(alpha_osqp)
    , refactor_dual_feasibility_threshold(refactor_dual_feasibility_threshold)
    , refactor_rho_threshold(refactor_rho_threshold)
    , mu_min_eq(mu_min_eq)
    , mu_max_eq(mu_max_eq)
    , mu_min_in(mu_min_in)
    , mu_max_in(mu_max_in)
    , mu_max_eq_inv(mu_max_eq_inv)
    , mu_min_eq_inv(mu_min_eq_inv)
    , mu_max_in_inv(mu_max_in_inv)
    , mu_min_in_inv(mu_min_in_inv)
    , mu_min_in_osqp(mu_min_in_osqp)
    , mu_max_in_inv_osqp(mu_max_in_inv_osqp)
    , mu_update_factor(mu_update_factor)
    , mu_update_inv_factor(mu_update_inv_factor)
    , cold_reset_mu_eq(cold_reset_mu_eq)
    , cold_reset_mu_in(cold_reset_mu_in)
    , cold_reset_mu_eq_inv(cold_reset_mu_eq_inv)
    , cold_reset_mu_in_inv(cold_reset_mu_in_inv)
    , cold_reset_mu_eq_osqp(cold_reset_mu_eq_osqp)
    , cold_reset_mu_in_osqp(cold_reset_mu_in_osqp)
    , cold_reset_mu_eq_inv_osqp(cold_reset_mu_eq_inv_osqp)
    , cold_reset_mu_in_inv_osqp(cold_reset_mu_in_inv_osqp)
    , eps_abs(eps_abs)
    , eps_rel(eps_rel)
    , high_accuracy(high_accuracy)
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
    , check_duality_gap(check_duality_gap)
    , eps_duality_gap_abs(eps_duality_gap_abs)
    , eps_duality_gap_rel(eps_duality_gap_rel)
    , update_mu(update_mu)
    , threshold_ratio_update_mu(threshold_ratio_update_mu)
    , threshold_ratio_update_mu_inv(threshold_ratio_update_mu_inv)
    , percentage_factorization_time_update_mu(
        percentage_factorization_time_update_mu)
    , update_mu_iteration_criteria(update_mu_iteration_criteria)
    , interval_update_mu(interval_update_mu)
    , polish(polish)
    , polish_refine_iter(polish_refine_iter)
    , delta(delta)
    , resume_admm(resume_admm)
    , preconditioner_max_iter(preconditioner_max_iter)
    , preconditioner_accuracy(preconditioner_accuracy)
    , eps_primal_inf(eps_primal_inf)
    , eps_dual_inf(eps_dual_inf)
    , bcl_update(bcl_update)
    , merit_function_type(merit_function_type)
    , alpha_gpdal(alpha_gpdal)
    , sparse_backend(sparse_backend)
    , primal_infeasibility_solving(primal_infeasibility_solving)
    , frequence_infeasibility_check(frequence_infeasibility_check)
    , default_H_eigenvalue_estimate(default_H_eigenvalue_estimate)
  {
    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT:
        default_rho = 1.E-6;
        break;
      case DenseBackend::PrimalLDLT:
        default_rho = 1.E-5;
        break;
      case DenseBackend::Automatic:
        default_rho = 1.E-6;
        break;
    }
  }
};

template<typename T>
bool
operator==(const Settings<T>& settings1, const Settings<T>& settings2)
{
  bool value =
    settings1.default_rho == settings2.default_rho &&
    settings1.default_mu_eq == settings2.default_mu_eq &&
    settings1.default_mu_in == settings2.default_mu_in &&
    settings1.alpha_bcl == settings2.alpha_bcl &&
    settings1.alpha_bcl == settings2.alpha_bcl &&
    settings1.beta_bcl == settings2.beta_bcl &&
    settings1.beta_bcl == settings2.beta_bcl &&
    settings1.alpha_osqp == settings2.alpha_osqp &&
    settings1.alpha_osqp == settings2.alpha_osqp &&
    settings1.refactor_dual_feasibility_threshold ==
      settings2.refactor_dual_feasibility_threshold &&
    settings1.refactor_rho_threshold == settings2.refactor_rho_threshold &&
    settings1.mu_min_eq == settings2.mu_min_eq &&
    settings1.mu_max_eq == settings2.mu_max_eq &&
    settings1.mu_min_in == settings2.mu_min_in &&
    settings1.mu_max_in == settings2.mu_max_in &&
    settings1.mu_max_eq_inv == settings2.mu_max_eq_inv &&
    settings1.mu_min_eq_inv == settings2.mu_min_eq_inv &&
    settings1.mu_max_in_inv == settings2.mu_max_in_inv &&
    settings1.mu_min_in_inv == settings2.mu_min_in_inv &&
    settings1.mu_min_in_osqp == settings2.mu_min_in_osqp &&
    settings1.mu_max_in_inv_osqp == settings2.mu_max_in_inv_osqp &&
    settings1.mu_update_factor == settings2.mu_update_factor &&
    settings1.mu_update_factor == settings2.mu_update_factor &&
    settings1.cold_reset_mu_eq == settings2.cold_reset_mu_eq &&
    settings1.cold_reset_mu_in == settings2.cold_reset_mu_in &&
    settings1.cold_reset_mu_eq_inv == settings2.cold_reset_mu_eq_inv &&
    settings1.cold_reset_mu_in_inv == settings2.cold_reset_mu_in_inv &&
    settings1.cold_reset_mu_eq_osqp == settings2.cold_reset_mu_eq_osqp &&
    settings1.cold_reset_mu_in_osqp == settings2.cold_reset_mu_in_osqp &&
    settings1.cold_reset_mu_eq_inv_osqp ==
      settings2.cold_reset_mu_eq_inv_osqp &&
    settings1.cold_reset_mu_in_inv_osqp ==
      settings2.cold_reset_mu_in_inv_osqp &&
    settings1.eps_abs == settings2.eps_abs &&
    settings1.eps_rel == settings2.eps_rel &&
    settings1.high_accuracy == settings2.high_accuracy &&
    settings1.max_iter == settings2.max_iter &&
    settings1.max_iter_in == settings2.max_iter_in &&
    settings1.safe_guard == settings2.safe_guard &&
    settings1.nb_iterative_refinement == settings2.nb_iterative_refinement &&
    settings1.eps_refact == settings2.eps_refact &&
    settings1.verbose == settings2.verbose &&
    settings1.initial_guess == settings2.initial_guess &&
    settings1.update_preconditioner == settings2.update_preconditioner &&
    settings1.compute_preconditioner == settings2.compute_preconditioner &&
    settings1.compute_timings == settings2.compute_timings &&
    settings1.check_duality_gap == settings2.check_duality_gap &&
    settings1.eps_duality_gap_abs == settings2.eps_duality_gap_abs &&
    settings1.eps_duality_gap_rel == settings2.eps_duality_gap_rel &&
    settings1.update_mu == settings2.update_mu &&
    settings1.threshold_ratio_update_mu ==
      settings2.threshold_ratio_update_mu &&
    settings1.threshold_ratio_update_mu_inv ==
      settings2.threshold_ratio_update_mu_inv &&
    settings1.percentage_factorization_time_update_mu ==
      settings2.percentage_factorization_time_update_mu &&
    settings1.update_mu_iteration_criteria ==
      settings2.update_mu_iteration_criteria &&
    settings1.interval_update_mu == settings2.interval_update_mu &&
    settings1.polish == settings2.polish &&
    settings1.polish_refine_iter == settings2.polish_refine_iter &&
    settings1.delta == settings2.delta &&
    settings1.resume_admm == settings2.resume_admm &&
    settings1.preconditioner_max_iter == settings2.preconditioner_max_iter &&
    settings1.preconditioner_accuracy == settings2.preconditioner_accuracy &&
    settings1.eps_primal_inf == settings2.eps_primal_inf &&
    settings1.eps_dual_inf == settings2.eps_dual_inf &&
    settings1.bcl_update == settings2.bcl_update &&
    settings1.merit_function_type == settings2.merit_function_type &&
    settings1.alpha_gpdal == settings2.alpha_gpdal &&
    settings1.sparse_backend == settings2.sparse_backend &&
    settings1.primal_infeasibility_solving ==
      settings2.primal_infeasibility_solving &&
    settings1.frequence_infeasibility_check ==
      settings2.frequence_infeasibility_check &&
    settings1.default_H_eigenvalue_estimate ==
      settings2.default_H_eigenvalue_estimate;
  return value;
}

template<typename T>
bool
operator!=(const Settings<T>& settings1, const Settings<T>& settings2)
{
  return !(settings1 == settings2);
}

} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_SETTINGS_HPP */
