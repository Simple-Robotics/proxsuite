/**
 * @file settings.hpp 
*/
#ifndef PROXSUITE_QP_SETTINGS_HPP
#define PROXSUITE_QP_SETTINGS_HPP

#include <Eigen/Core>
#include <qp/dense/views.hpp>
#include "qp/sparse/fwd.hpp"

namespace proxsuite {
namespace qp {

template <typename T>
struct Settings {
	T alpha_bcl;
	T beta_bcl;

	T refactor_dual_feasibility_threshold;
	T refactor_rho_threshold;

	T mu_max_eq;
	T mu_max_in;
	T mu_max_eq_inv;
	T mu_max_in_inv;

	T mu_update_factor;
	T mu_update_inv_factor;

	T cold_reset_mu_eq;
	T cold_reset_mu_in;
	T cold_reset_mu_eq_inv;
	T cold_reset_mu_in_inv;

	isize max_iter;
	isize max_iter_in;
	T eps_abs;
	T eps_rel;
	T eps_refact;
	isize nb_iterative_refinement;

	bool verbose;
	bool warm_start;

	T eps_primal_inf;
	T eps_dual_inf;

	Settings(
			T alpha_bcl_ = 0.1,
			T beta_bcl_ = 0.9,
			T refactor_dual_feasibility_threshold_ = 1e-2,
			T refactor_rho_threshold_ = 1e-7,
			T mu_max_eq_ = 1e-9,
			T mu_max_in_ = 1e-8,
			T mu_max_eq_inv_ = 1e9,
			T mu_max_in_inv_ = 1e8,
			T mu_update_factor_ = 0.1,
			T mu_update_inv_factor_ = 10,
			T cold_reset_mu_eq_ = 1. / 1.1,
			T cold_reset_mu_in_ = 1. / 1.1,
			T cold_reset_mu_eq_inv_ = 1.1,
			T cold_reset_mu_in_inv_ = 1.1,
			T eps_abs_ = 1.e-9,
			T eps_rel_ = 0.,
			isize max_iter_ = 10000,
			isize max_iter_in_ = 1500,
			isize nb_iterative_refinement_ = 10,
			T eps_refact_ = 1.e-6, // before eps_refact_=1.e-6
			bool VERBOSE = false,
			bool warm_start_ = false,
			T eps_primal_inf_ = 1.E-14,
			T eps_dual_inf_ = 1.E-14)
			: alpha_bcl(alpha_bcl_),
				beta_bcl(beta_bcl_),
				refactor_dual_feasibility_threshold(
						refactor_dual_feasibility_threshold_),
				refactor_rho_threshold(refactor_rho_threshold_),
				mu_max_eq(mu_max_eq_),
				mu_max_in(mu_max_in_),
				mu_max_eq_inv(mu_max_eq_inv_),
				mu_max_in_inv(mu_max_in_inv_),
				mu_update_factor(mu_update_factor_),
				mu_update_inv_factor(mu_update_inv_factor_),
				cold_reset_mu_eq(cold_reset_mu_eq_),
				cold_reset_mu_in(cold_reset_mu_in_),
				cold_reset_mu_eq_inv(cold_reset_mu_eq_inv_),
				cold_reset_mu_in_inv(cold_reset_mu_in_inv_),
				max_iter(max_iter_),
				max_iter_in(max_iter_in_),
				eps_abs(eps_abs_),
				eps_rel(eps_rel_),
				eps_refact(eps_refact_),
				nb_iterative_refinement(nb_iterative_refinement_),
				verbose(VERBOSE),
				warm_start(warm_start_),
				eps_primal_inf(eps_primal_inf_),
				eps_dual_inf(eps_dual_inf_) {}
};

} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SETTINGS_HPP */
