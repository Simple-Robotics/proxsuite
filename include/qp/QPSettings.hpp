#ifndef INRIA_LDLT_Qpsettings_HPP_VCVSK3EOS
#define INRIA_LDLT_Qpsettings_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"

namespace qp {

template <typename T>
struct Qpsettings {
public:

    T _alpha_bcl;
    T _beta_bcl;

	T _refactor_dual_feasibility_threshold;
	T _refactor_rho_threshold;
	T _refactor_rho_update_factor;

	T _mu_max_eq;
    T _mu_max_in;
	T _mu_max_eq_inv;
    T _mu_max_in_inv;

    T _mu_update_factor;
    T _mu_update_inv_factor;

	T _cold_reset_mu_eq;
	T _cold_reset_mu_in;
	T _cold_reset_mu_eq_inv;
	T _cold_reset_mu_in_inv;

	isize _max_iter;
	isize _max_iter_in;
	T _eps_abs;
	T _eps_rel;
	T _eps_IG;
	T _R;
    T _eps_refact;
    isize _nb_iterative_refinement;

    bool _VERBOSE;

	Qpsettings(T alpha_bcl=0.1,T beta_bcl=0.9,T refactor_dual_feasibility_threshold=1e-2,
               T refactor_rho_threshold=1e-7, T refactor_rho_update_factor=0.1,
               T mu_max_eq=1e9, T mu_max_in=1e8, T mu_max_eq_inv=1e-9,T mu_max_in_inv=1e-8,
               T mu_update_factor=10, T mu_update_inv_factor=0.1,
               T cold_reset_mu_eq=1.1, T cold_reset_mu_in=1.1,
               T cold_reset_mu_eq_inv=1./1.1, T cold_reset_mu_in_inv=1./1.1,
               T eps_abs=1.e-9,T eps_rel=0.,T err_IG=1.e-2, T R=5.,
               isize max_iter=10000,isize max_iter_in=1500,
               isize nb_iterative_refinement=10,T eps_refact=1.e-6,
               bool VERBOSE = false)
                {
        
                    _alpha_bcl=alpha_bcl;
                    _beta_bcl=beta_bcl;

                    _refactor_dual_feasibility_threshold=refactor_dual_feasibility_threshold;
                    _refactor_rho_threshold=refactor_rho_threshold;
                    _refactor_rho_update_factor=refactor_rho_update_factor;

                    _mu_max_eq=mu_max_eq;
                    _mu_max_in=mu_max_in;
                    _mu_max_eq_inv=mu_max_eq_inv;
                    _mu_max_in_inv=mu_max_in_inv;
                    _mu_update_factor=mu_update_factor;
                    _mu_update_inv_factor=mu_update_inv_factor;

                    _cold_reset_mu_eq=cold_reset_mu_eq;
                    _cold_reset_mu_in=cold_reset_mu_in;
                    _cold_reset_mu_eq_inv=cold_reset_mu_eq_inv;
                    _cold_reset_mu_in_inv=cold_reset_mu_in_inv;

                    _eps_abs=eps_abs;
                    _eps_rel=eps_rel;
                    _eps_IG=err_IG;
                    _R=R;
                    _eps_refact = eps_refact;

                    _max_iter=max_iter;
                    _max_iter_in=max_iter_in;
                    _nb_iterative_refinement=nb_iterative_refinement;
                    _VERBOSE=VERBOSE;
                }
    
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_Qpsettings_HPP_VCVSK3EOS */
