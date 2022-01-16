#ifndef INRIA_LDLT_Qpsettings_HPP_VCVSK3EOS
#define INRIA_LDLT_Qpsettings_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"

namespace qp {

template <typename T>
struct Qpsettings {
public:

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
	T eps_IG;
	T R;
    T eps_refact;
    isize nb_iterative_refinement;

    bool verbose;

	Qpsettings(T alpha_bcl=0.1,T beta_bcl=0.9,T refactor_dual_feasibility_threshold=1e-2,
               T refactor_rho_threshold=1e-7, T refactor_rho_update_factor=0.1,
               T mu_max_eq=1e9, T mu_max_in=1e8, T mu_max_eq_inv=1e-9,T mu_max_in_inv=1e-8,
               T mu_update_factor=10, T mu_update_inv_factor=0.1,
               T cold_reset_mu_eq=1.1, T cold_reset_mu_in=1.1,
               T cold_reset_mu_eq_inv=1./1.1, T cold_reset_mu_in_inv=1./1.1,
               T eps_abs=1.e-9,T eps_rel=0.,T err_IG=1.e-2, T r=5.,
               isize max_iter=10000,isize max_iter_in=1500,
               isize nb_iterative_refinement=10,T eps_refact=1.e-6,
               bool VERBOSE = false)
                {
        
                    alpha_bcl=alpha_bcl;
                    beta_bcl=beta_bcl;

                    refactor_dual_feasibility_threshold=refactor_dual_feasibility_threshold;
                    refactor_rho_threshold=refactor_rho_threshold;
                    refactor_rho_update_factor=refactor_rho_update_factor;

                    mu_max_eq=mu_max_eq;
                    mu_max_in=mu_max_in;
                    mu_max_eq_inv=mu_max_eq_inv;
                    mu_max_in_inv=mu_max_in_inv;
                    mu_update_factor=mu_update_factor;
                    mu_update_inv_factor=mu_update_inv_factor;

                    cold_reset_mu_eq=cold_reset_mu_eq;
                    cold_reset_mu_in=cold_reset_mu_in;
                    cold_reset_mu_eq_inv=cold_reset_mu_eq_inv;
                    cold_reset_mu_in_inv=cold_reset_mu_in_inv;

                    eps_abs=eps_abs;
                    eps_rel=eps_rel;
                    eps_IG=err_IG;
                    R=r;
                    eps_refact = eps_refact;

                    max_iter=max_iter;
                    max_iter_in=max_iter_in;
                    nb_iterative_refinement=nb_iterative_refinement;
                    verbose=VERBOSE;
                }
    
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_Qpsettings_HPP_VCVSK3EOS */
