#ifndef INRIA_LDLT_OLD_NEW_SOLVER_EQ_HPP_HDWGZKCLS
#define INRIA_LDLT_OLD_NEW_SOLVER_EQ_HPP_HDWGZKCLS

//#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "qp/proxqp/solver.hpp"
//#include "ldlt/factorize.hpp"
//#include "ldlt/detail/meta.hpp"
//#include "ldlt/solve.hpp"
//#include "ldlt/update.hpp"
#include <cmath>

#include <iostream>
#include<fstream>

namespace qp {
inline namespace tags {
//using namespace ldlt::tags;
}


namespace detail {

template<typename T>
T newton_eq(
		const qp::QPSettings<T>& QPSettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps_int
		){

			isize numactive_inequalities = qpwork.active_inequalities.count();
			isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq ;
			qpwork.rhs.setZero();
			qpwork.rhs.head(qpmodel.dim) = -qpwork.dual_residual_scaled ;
			qpwork.rhs.segment(qpmodel.dim,qpmodel.n_eq) = -qpwork.primal_residual_eq_scaled ;	
            qp::detail::iterative_solve_with_permut_fact( //
					QPSettings,
					qpmodel,
					qpresults,
					qpwork,
                    eps_int,
                    inner_pb_dim
					);
			// use active_part_z as a temporary variable to permut back dw_aug newton step

			qpwork.Adx.noalias() = (qpwork.A_scaled*qpwork.dw_aug.head(qpmodel.dim)- qpwork.dw_aug.segment(qpmodel.dim,qpmodel.n_eq) * qpresults.mu_eq_inv).eval() ; 
			qpwork.Hdx.noalias() = qpwork.H_scaled*qpwork.dw_aug.head(qpmodel.dim)+qpwork.A_scaled.transpose()*qpwork.dw_aug.segment(qpmodel.dim,qpmodel.n_eq);
			qpwork.Hdx.noalias() += qpresults.rho*qpwork.dw_aug.head(qpmodel.dim) ; 

			if (QPSettings.verbose){
				std::cout << "alpha from initial guess " << qpwork.alpha << std::endl;
			}
			qpresults.x.noalias() +=  qpwork.dw_aug.head(qpmodel.dim) ; 
			qpresults.y.noalias() +=  qpwork.dw_aug.segment(qpmodel.dim,qpmodel.n_eq) ; 
			qpwork.primal_residual_eq_scaled.noalias() += qpwork.Adx ;
			qpwork.dual_residual_scaled.noalias() += qpwork.Hdx ;
			qpwork.dw_aug.setZero();

			T err_saddle_point = qp::detail::compute_primal_dual_residual( 
					qpmodel,
					qpresults,
					qpwork
				);

			return err_saddle_point;
}

template <typename T>
void qp_solve_eq( //
		const qp::QPSettings<T>& QPSettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	//using namespace ldlt::tags;

	/*** TEST WITH MATRIX FULL OF NAN FOR DEBUG
    static constexpr Layout layout = rowmajor;
    static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
	RowMat test(2,2); // test it is full of nan for debug
	std::cout << "test " << test << std::endl;
	*/

	//::Eigen::internal::set_is_malloc_allowed(false);
	
	const T machine_eps = std::numeric_limits<T>::epsilon();

	T bcl_eta_ext_init = pow(T(0.1),QPSettings.alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(QPSettings.eps_abs,T(1.E-9));

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);
	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);
	isize saturation(0);
	
	for (i64 iter = 0; iter <= QPSettings.max_iter; ++iter) {

		qpresults.n_ext +=1;
		if (iter == QPSettings.max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs
		);

		qp::detail::global_dual_residual(
			qpmodel,
			qpresults,
			qpwork,
			dual_feasibility_lhs,
			dual_feasibility_rhs_0,
			dual_feasibility_rhs_1,
        	dual_feasibility_rhs_3
		);
		
		
		T new_bcl_mu_in(qpresults.mu_in);
		T new_bcl_mu_eq(qpresults.mu_eq);
		T new_bcl_mu_in_inv(qpresults.mu_in_inv);
		T new_bcl_mu_eq_inv(qpresults.mu_eq_inv);

		T rhs_pri(QPSettings.eps_abs);
		if (QPSettings.eps_rel !=0){
			rhs_pri+= QPSettings.eps_rel * max2(  max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),  max2(max2( qpwork.primal_feasibility_rhs_1_eq, qpwork.primal_feasibility_rhs_1_in_u ),qpwork.primal_feasibility_rhs_1_in_l ) );
		}
		bool is_primal_feasible = primal_feasibility_lhs <= rhs_pri;

		T rhs_dua(QPSettings.eps_abs );
		if (QPSettings.eps_rel !=0){
			rhs_dua+=QPSettings.eps_rel * max2( max2(   dual_feasibility_rhs_3, dual_feasibility_rhs_0),
													max2( dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2)) ;
		}

		bool is_dual_feasible = dual_feasibility_lhs <= rhs_dua;
		
		if (QPSettings.verbose){
			std::cout << "---------------it : " << iter << " primal residual : " << primal_feasibility_lhs << " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "bcl_eta_ext : " << bcl_eta_ext << " bcl_eta_in : " << bcl_eta_in <<  " rho : " << qpresults.rho << " bcl_mu_eq : " << qpresults.mu_eq << " bcl_mu_in : " << qpresults.mu_in <<std::endl;
			std::cout << "QPSettings.eps_abs " << QPSettings.eps_abs << "  QPSettings.eps_rel *rhs " <<  QPSettings.eps_rel * max2(  max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),  max2(max2( qpwork.primal_feasibility_rhs_1_eq, qpwork.primal_feasibility_rhs_1_in_u ),qpwork.primal_feasibility_rhs_1_in_l ) ) << std::endl;
			std::cout << "is_primal_feasible " << is_primal_feasible << " is_dual_feasible " << is_dual_feasible << std::endl;
		}
		if (is_primal_feasible){
			
			if (dual_feasibility_lhs >= QPSettings.refactor_dual_feasibility_threshold && qpresults.rho != QPSettings.refactor_rho_threshold){

				T rho_new(QPSettings.refactor_rho_threshold);

				
				qp::detail::refactorize(
						qpmodel,
						qpresults,
						qpwork,
						rho_new
				);

				qpresults.rho = rho_new;
			}
			if (is_dual_feasible){

				qpwork.ruiz.unscale_primal_in_place(VectorViewMut<T>{from_eigen,qpresults.x}); 
				qpwork.ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{from_eigen,qpresults.y});
				qpwork.ruiz.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen,qpresults.z});

				qpresults.objValue = (0.5 * qpmodel.H * qpresults.x + qpmodel.g).dot(qpresults.x) ;
				break;

			}
		}
		
		qpwork.x_prev = qpresults.x; 
		qpwork.y_prev = qpresults.y; 
		qpwork.z_prev = qpresults.z; 

        T err_in = qp::detail::newton_eq<T>(
                        QPSettings,
                        qpmodel,
                        qpresults,
                        qpwork,
                        bcl_eta_in

        );
        qpresults.n_tot +=1;

		if (QPSettings.verbose){
			std::cout << " error from initial guess : " << err_in << " bcl_eta_in " << bcl_eta_in << std::endl;
		}

		T primal_feasibility_lhs_new(primal_feasibility_lhs) ; 

		qp::detail::global_primal_residual(
						qpmodel,
						qpresults,
						qpwork,
						primal_feasibility_lhs_new,
						primal_feasibility_eq_rhs_0,
						primal_feasibility_in_rhs_0,
						primal_feasibility_eq_lhs,
						primal_feasibility_in_lhs
		);

		is_primal_feasible = primal_feasibility_lhs_new <= (QPSettings.eps_abs + QPSettings.eps_rel * max2(  max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),  max2(max2( qpwork.primal_feasibility_rhs_1_eq, qpwork.primal_feasibility_rhs_1_in_u ),qpwork.primal_feasibility_rhs_1_in_l ) ));

		if (is_primal_feasible){
			T dual_feasibility_lhs_new(dual_feasibility_lhs) ; 
		
			qp::detail::global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3
			);

			is_dual_feasible = dual_feasibility_lhs_new <=(QPSettings.eps_abs + QPSettings.eps_rel * max2( max2(   dual_feasibility_rhs_3, dual_feasibility_rhs_0),
													max2( dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2)) );

			if (is_dual_feasible){
				
				qpwork.ruiz.unscale_primal_in_place(VectorViewMut<T>{from_eigen,qpresults.x}); 
				qpwork.ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{from_eigen,qpresults.y});
				qpwork.ruiz.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen,qpresults.z});

				qpresults.objValue = (0.5 * qpmodel.H * qpresults.x + qpmodel.g).dot(qpresults.x) ;
				break;
			}
		}
		

		qp::detail::bcl_update(
					QPSettings,
					qpmodel,
					qpresults,
					qpwork,
					primal_feasibility_lhs_new,
					bcl_eta_ext,
					bcl_eta_in,
					bcl_eta_ext_init,
					eps_in_min,

					new_bcl_mu_in,
					new_bcl_mu_eq,
					new_bcl_mu_in_inv,
					new_bcl_mu_eq_inv

		);
		
		/// effective mu upddate

		if (qpresults.mu_in != new_bcl_mu_in || qpresults.mu_eq != new_bcl_mu_eq) {
					{
					++qpresults.n_mu_change;
					}
			}	
			qp::detail::mu_update(
				qpmodel,
				qpresults,
				qpwork,
				new_bcl_mu_eq_inv,
				new_bcl_mu_in_inv);
			qpresults.mu_eq = new_bcl_mu_eq;
			qpresults.mu_in = new_bcl_mu_in;
			qpresults.mu_eq_inv = new_bcl_mu_eq_inv;
			qpresults.mu_in_inv = new_bcl_mu_in_inv;
	}
	
	
	//::Eigen::internal::set_is_malloc_allowed(true);
	qpresults.objValue = (0.5 * qpmodel.H * qpresults.x + qpmodel.g).dot(qpresults.x) ;

}


} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_SOLVER_EQ_HPP_HDWGZKCLS */
