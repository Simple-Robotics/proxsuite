#ifndef INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS

#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/proxqp/old_new_line_search.hpp"
#include <cmath>

#include <iostream>
#include<fstream>

template <typename T>
void saveData(std::string fileName, Eigen::Matrix<T, Eigen::Dynamic, 1>  vector)
{
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << vector.format(CSVFormat);
        file.close();
    }
}

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}


namespace detail {


template <typename T>
void oldNew_refactorize(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,

		T rho_new,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<T> dw_aug_
		) {
		
	auto dw_aug = dw_aug_.to_eigen();
	dw_aug.setZero();
	qpwork._kkt.diagonal().head(qpmodel._dim).array() += rho_new - qpresults._rho; 
	qpwork._kkt.diagonal().segment(qpmodel._dim,qpmodel._n_eq).array() = -qpresults._mu_eq_inv; 
	ldl.factorize(qpwork._kkt);

	for (isize j = 0; j < qpresults._n_c; ++j) {
		for (isize i = 0; i < qpmodel._n_in; ++i) {
			if (j == qpwork._current_bijection_map(i)) {
					dw_aug.head(qpmodel._dim) = qpwork._c_scaled.row(i);
					dw_aug(qpmodel._dim + qpmodel._n_eq + j) = - qpresults._mu_in_inv; // mu_in stores the inverse of mu_in
					ldl.insert_at(qpmodel._n_eq + qpmodel._dim + j, dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c));
					dw_aug(qpmodel._dim + qpmodel._n_eq + j) = T(0);
			}
		}
	}
	dw_aug.setZero();
}

template <typename T>
void oldNew_mu_update(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<T>  dw_aug_,
		T mu_eq_new_inv,
		T mu_in_new_inv) {
	T diff = 0;
	auto dw_aug = dw_aug_.to_eigen();

	dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c).setZero();
	if (qpmodel._n_eq > 0) {
		diff = qpresults._mu_eq_inv -  mu_eq_new_inv; // mu stores the inverse of mu

		for (isize i = 0; i < qpmodel._n_eq; i++) {
			dw_aug(qpmodel._dim + i) = T(1);
			ldl.rank_one_update(dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c), diff);
			dw_aug(qpmodel._dim + i) = T(0);
		}
	}
	if (qpresults._n_c > 0) {
		diff = qpresults._mu_in_inv - mu_in_new_inv; // mu stores the inverse of mu
		for (isize i = 0; i < qpresults._n_c; i++) {
			dw_aug(qpmodel._dim + qpmodel._n_eq + i) = T(1);
			ldl.rank_one_update(dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c), diff);
			dw_aug(qpmodel._dim + qpmodel._n_eq + i) = T(0);
		}
	}
}

template <typename T>
void oldNew_iterative_residual(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,

		VectorViewMut<T> rhs_,
        VectorViewMut<T>  dw_aug_,
        VectorViewMut<T>  err_,
		isize inner_pb_dim,
		isize it_,
		std::string str,
		const bool chekNoAlias) {
 
    auto rhs = rhs_.to_eigen();
    auto dw_aug = dw_aug_.to_eigen();
    auto err = err_.to_eigen();

	err.head(inner_pb_dim).noalias()  = rhs.head(inner_pb_dim);
	if (chekNoAlias){
		saveData("_err_after_copy"+std::to_string(it_)+str,err.eval());
	}

	err.head(qpmodel._dim).noalias()  -= qpwork._h_scaled * dw_aug.head(qpmodel._dim);
	if (chekNoAlias){
		saveData("_err_after_H"+std::to_string(it_)+str,err.eval());
	}
	
    err.head(qpmodel._dim).noalias()  -= qpresults._rho * dw_aug.head(qpmodel._dim);
	if (chekNoAlias){
		saveData("_err_after_rho"+std::to_string(it_)+str,err.eval()); 
	}

    err.head(qpmodel._dim).noalias()  -= qpwork._a_scaled.transpose() * dw_aug.segment(qpmodel._dim, qpmodel._n_eq);
	if (chekNoAlias){
		saveData("_err_after_Ahead"+std::to_string(it_)+str,err.eval());
	}

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			err.head(qpmodel._dim).noalias()  -= dw_aug(qpmodel._dim + qpmodel._n_eq + j) * qpwork._c_scaled.row(i);
			err(qpmodel._dim + qpmodel._n_eq + j) -=
					(qpwork._c_scaled.row(i).dot(dw_aug.head(qpmodel._dim)) -
					 dw_aug(qpmodel._dim + qpmodel._n_eq + j)  * qpresults._mu_in_inv); // mu stores the inverse of mu
		}
	}
	if (chekNoAlias){
		saveData("_err_after_act"+std::to_string(it_)+str,err.eval());
	}

	err.segment(qpmodel._dim, qpmodel._n_eq).noalias()  -= (qpwork._a_scaled * dw_aug.head(qpmodel._dim) - dw_aug.segment(qpmodel._dim, qpmodel._n_eq) * qpresults._mu_eq_inv); // mu stores the inverse of mu

}

template <typename T>
void oldNew_iterative_solve_with_permut_fact_new( //
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,
		VectorViewMut<T> rhs_,
        VectorViewMut<T>  dw_aug_,
		
		T eps,
		isize inner_pb_dim,
        ldlt::Ldlt<T>& ldl,
        const bool VERBOSE,
		VectorViewMut<T> _err,
		isize it_,
		std::string str,
		const bool chekNoAlias
        ){

    auto rhs = rhs_.to_eigen();
    auto dw_aug = dw_aug_.to_eigen();
    auto err = _err.to_eigen();
	err.setZero();
	i32 it = 0;

	dw_aug.head(inner_pb_dim) = rhs.head(inner_pb_dim);
	
	ldl.solve_in_place(dw_aug.head(inner_pb_dim));

	if (chekNoAlias){
		std::cout << "-----------before computing residual" << std::endl;
		saveData("rhs_before"+std::to_string(it_)+str,rhs_.to_eigen().eval());
		saveData("dw_aug_before"+std::to_string(it_)+str,rhs_.to_eigen().eval());
		saveData("_err_before"+std::to_string(it_)+str,_err.to_eigen().eval());
		std::cout << "rhs_" << rhs_.to_eigen() << std::endl;
		std::cout << "dw_aug_" << dw_aug_.to_eigen() << std::endl;
		std::cout << "_err" << _err.to_eigen() << std::endl;
	}


	qp::detail::oldNew_iterative_residual<T>( 
					qpmodel,
					qpresults,
					qpwork,

                    rhs_,
                    dw_aug_,
                    _err,
                    inner_pb_dim,
					it_,
					str,
					chekNoAlias);
	if (chekNoAlias){
		std::cout << "-----------after computing residual" << std::endl;
		std::cout << "rhs_" << rhs_.to_eigen() << std::endl;
		std::cout << "dw_aug_" << dw_aug_.to_eigen() << std::endl;
		std::cout << "_err" << _err.to_eigen() << std::endl;
		saveData("rhs_after"+std::to_string(it_)+str,rhs_.to_eigen().eval());
		saveData("dw_aug_after"+std::to_string(it_)+str,rhs_.to_eigen().eval());
		saveData("_err_after"+std::to_string(it_)+str,_err.to_eigen().eval());
	}

	++it;
	if (VERBOSE){
		std::cout << "infty_norm(res) " << qp::infty_norm( err.head(inner_pb_dim)) << std::endl;
	}
	while (infty_norm( err.head(inner_pb_dim)) >= eps) {
		if (it >= qpsettings._nb_iterative_refinement) {
			break;
		} 
		++it;
		ldl.solve_in_place( err.head(inner_pb_dim));
		dw_aug.head(inner_pb_dim).noalias() +=  err.head(inner_pb_dim);

		err.head(inner_pb_dim).setZero();
		qp::detail::oldNew_iterative_residual<T>(
					qpmodel,
					qpresults,
					qpwork,

                    rhs_,
                    dw_aug_,
                    _err,
                    inner_pb_dim,
					it_,
					str,
					chekNoAlias);

		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(err.head(inner_pb_dim)) << std::endl;
		}
	}
	if (qp::infty_norm(err.head(inner_pb_dim))>= std::max(eps,qpsettings._eps_refact)){
		{
			
			LDLT_MULTI_WORKSPACE_MEMORY(
				(_htot,Uninit, Mat(qpmodel._dim+qpmodel._n_eq+qpresults._n_c, qpmodel._dim+qpmodel._n_eq+qpresults._n_c),LDLT_CACHELINE_BYTES, T)
				);
			auto Htot = _htot.to_eigen().eval();

			Htot.setZero();
			
			qpwork._kkt.diagonal().segment(qpmodel._dim,qpmodel._n_eq).array() = -qpresults._mu_eq_inv; 
			Htot.topLeftCorner(qpmodel._dim+qpmodel._n_eq, qpmodel._dim+qpmodel._n_eq) = qpwork._kkt;

			Htot.diagonal().segment(qpmodel._dim+qpmodel._n_eq,qpresults._n_c).array() = -qpresults._mu_in_inv; 

			for (isize i = 0; i< qpmodel._n_in ; ++i){
					
					isize j = qpwork._current_bijection_map(i);
					if (j<qpresults._n_c){
						Htot.block(j+qpmodel._dim+qpmodel._n_eq,0,1,qpmodel._dim) = qpwork._c_scaled.row(i) ; 
						Htot.block(0,j+qpmodel._dim+qpmodel._n_eq,qpmodel._dim,1) = qpwork._c_scaled.row(i) ; 
					}
			}
			
			oldNew_refactorize(
						qpmodel,
						qpresults,
						qpwork,

						qpresults._rho,
						ldl,
						VectorViewMut<T>{from_eigen,dw_aug}
						);
			
			std::cout << " ldl.reconstructed_matrix() - Htot " << infty_norm(ldl.reconstructed_matrix() - Htot)<< std::endl;
		}
		it = 0;
		dw_aug.head(inner_pb_dim) = rhs.head(inner_pb_dim);

		ldl.solve_in_place(dw_aug.head(inner_pb_dim));

		qp::detail::oldNew_iterative_residual<T>(
					qpmodel,
					qpresults,
					qpwork,

                    rhs_,
                    dw_aug_,
                    _err,
                    inner_pb_dim,
					it_,
					str,
					chekNoAlias);
		++it;
		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm( err.head(inner_pb_dim)) << std::endl;
		}
		while (infty_norm( err.head(inner_pb_dim)) >= eps) {
			if (it >= qpsettings._nb_iterative_refinement) {
				break;
			}
			++it;
			ldl.solve_in_place( err.head(inner_pb_dim) );
			dw_aug.head(inner_pb_dim).noalias()  +=  err.head(inner_pb_dim);
  
			err.head(inner_pb_dim).setZero();
			qp::detail::oldNew_iterative_residual<T>(
					qpmodel,
					qpresults,
					qpwork,

                    rhs_,
                    dw_aug_,
                    _err,
                    inner_pb_dim,
					it_,
					str,
					chekNoAlias);

			if (VERBOSE){
				std::cout << "infty_norm(res) " << qp::infty_norm(err.head(inner_pb_dim)) << std::endl;
			}
		}
	}
	rhs.head(inner_pb_dim).setZero();
}


template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
void oldNew_BCL_update(
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,
		T& primal_feasibility_lhs,
		VectorViewMut<T> primal_residual_in_scaled_u,
		VectorViewMut<T> primal_residual_in_scaled_l,
		VectorViewMut<T> primal_residual_eq_scaled,
		Preconditioner& precond,
		T& bcl_eta_ext,
		T& bcl_eta_in,

		VectorViewMut<T> dw_aug_,
		ldlt::Ldlt<T>& ldl,
		T bcl_eta_ext_init,
		T eps_in_min
		){
		precond.scale_primal_residual_in_place_eq(primal_residual_eq_scaled);
		precond.scale_primal_residual_in_place_in(primal_residual_in_scaled_l);
		precond.scale_primal_residual_in_place_in(primal_residual_in_scaled_u);
		T primal_feasibility_eq_lhs = infty_norm(primal_residual_eq_scaled.to_eigen());
		T primal_feasibility_in_lhs = max2(infty_norm(primal_residual_in_scaled_l.to_eigen()),infty_norm(primal_residual_in_scaled_u.to_eigen()));
		T tmp = max2(primal_feasibility_eq_lhs,primal_feasibility_in_lhs);
		if (tmp <= bcl_eta_ext) {
			std::cout << "good step"<< std::endl;
			bcl_eta_ext = bcl_eta_ext * pow(qpresults._mu_in_inv, qpsettings._beta_bcl);
			bcl_eta_in = max2(bcl_eta_in * qpresults._mu_in_inv,eps_in_min);
		} else {
			std::cout << "bad step"<< std::endl; 
			qpresults._y = qpwork._ye;
			qpresults._z = qpwork._ze;
			T new_bcl_mu_in(std::min(qpresults._mu_in * qpsettings._mu_update_factor, qpsettings._mu_max_in));
			T new_bcl_mu_eq(std::min(qpresults._mu_eq * qpsettings._mu_update_factor, qpsettings._mu_max_eq));
			T new_bcl_mu_in_inv(max2(qpresults._mu_in_inv * qpsettings._mu_update_inv_factor, qpsettings._mu_max_in_inv)); // mu stores the inverse of mu
			T new_bcl_mu_eq_inv(max2(qpresults._mu_eq_inv * qpsettings._mu_update_inv_factor, qpsettings._mu_max_eq_inv)); // mu stores the inverse of mu


			if (qpresults._mu_in != new_bcl_mu_in || qpresults._mu_eq != new_bcl_mu_eq) {
					{
					++qpresults._n_mu_change;
					}
			}	
			qp::detail::oldNew_mu_update(
				qpmodel,
				qpresults,
				ldl,
				dw_aug_,
				new_bcl_mu_eq_inv,
				new_bcl_mu_in_inv);
			qpresults._mu_eq = new_bcl_mu_eq;
			qpresults._mu_in = new_bcl_mu_in;
			qpresults._mu_eq_inv = new_bcl_mu_eq_inv;
			qpresults._mu_in_inv = new_bcl_mu_in_inv;
			bcl_eta_ext = bcl_eta_ext_init * pow(qpresults._mu_in_inv, qpsettings._alpha_bcl);
			bcl_eta_in = max2(  qpresults._mu_in_inv  ,eps_in_min);
	}
}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
void oldNew_global_primal_residual(
			qp::Qpdata<T>& qpmodel,
			qp::Qpresults<T>& qpresults,
			qp::OldNew_Qpworkspace<T>& qpwork,
			T& primal_feasibility_lhs,
			T& primal_feasibility_eq_rhs_0,
        	T& primal_feasibility_in_rhs_0,
			T& primal_feasibility_eq_lhs,
			T& primal_feasibility_in_lhs,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  primal_residual_eq_scaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  primal_residual_in_scaled_u,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  primal_residual_in_scaled_l,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  primal_residual_eq_unscaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  primal_residual_in_l_unscaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  primal_residual_in_u_unscaled,
			Preconditioner& precond
		){		

				qpwork._residual_scaled_tmp.setZero();
				// A×x - b and Cx - u and Cx - l  /!\ optimization surely possible
				primal_residual_eq_scaled.setZero();
				primal_residual_eq_scaled.noalias() += qpwork._a_scaled * qpresults._x;
    
				primal_residual_in_scaled_u.setZero();
				primal_residual_in_scaled_u.noalias() += qpwork._c_scaled * qpresults._x;
				primal_residual_in_scaled_l.setZero();
				primal_residual_in_scaled_l = primal_residual_in_scaled_u;

				{
					qpwork._residual_scaled_tmp.middleRows(qpmodel._dim,qpmodel._n_eq) = primal_residual_eq_scaled;
                    qpwork._residual_scaled_tmp.bottomRows(qpmodel._n_in) = primal_residual_in_scaled_u;
					precond.unscale_primal_residual_in_place_eq(
							VectorViewMut<T>{from_eigen, qpwork._residual_scaled_tmp.middleRows(qpmodel._dim,qpmodel._n_eq)});
                    precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, qpwork._residual_scaled_tmp.bottomRows(qpmodel._n_in)});
					primal_feasibility_eq_rhs_0 = infty_norm(qpwork._residual_scaled_tmp.middleRows(qpmodel._dim,qpmodel._n_eq));
                    primal_feasibility_in_rhs_0 = infty_norm(qpwork._residual_scaled_tmp.bottomRows(qpmodel._n_in));
				}
				primal_residual_eq_scaled -= qpwork._b_scaled;
                primal_residual_in_scaled_u -= qpwork._u_scaled ;
                primal_residual_in_scaled_l -= qpwork._l_scaled ;

                //primal_residual_in_scaled_u = ( primal_residual_in_scaled_u.array() >0).select(primal_residual_in_scaled_u, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)); 
				primal_residual_in_scaled_u = detail::positive_part(primal_residual_in_scaled_u);
				//primal_residual_in_scaled_l = ( primal_residual_in_scaled_l.array() < 0).select(primal_residual_in_scaled_l, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)); 
                primal_residual_in_scaled_l = detail::negative_part(primal_residual_in_scaled_l);
				primal_residual_eq_unscaled = primal_residual_eq_scaled;
				precond.unscale_primal_residual_in_place_eq(
						VectorViewMut<T>{from_eigen, primal_residual_eq_unscaled});
                precond.unscale_primal_residual_in_place_in(
						VectorViewMut<T>{from_eigen, primal_residual_in_scaled_u});
                precond.unscale_primal_residual_in_place_in(
						VectorViewMut<T>{from_eigen, primal_residual_in_scaled_l});

				primal_feasibility_eq_lhs = infty_norm(primal_residual_eq_unscaled);
                primal_feasibility_in_lhs = max2(infty_norm(primal_residual_in_scaled_l),infty_norm(primal_residual_in_scaled_u));
                primal_feasibility_lhs = max2(primal_feasibility_eq_lhs,primal_feasibility_in_lhs);
}


template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
void oldNew_global_dual_residual(
			qp::Qpdata<T>& qpmodel,
			qp::Qpresults<T>& qpresults,
			qp::OldNew_Qpworkspace<T>& qpwork,
			T& dual_feasibility_lhs,
			T& dual_feasibility_rhs_0,
			T& dual_feasibility_rhs_1,
        	T& dual_feasibility_rhs_3,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  dual_residual_scaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  dual_residual_unscaled,
			Preconditioner& precond
		){

			qpwork._residual_scaled_tmp.setZero();

			dual_residual_scaled = qpwork._g_scaled;
			{
				qpwork._residual_scaled_tmp.topRows(qpmodel._dim).noalias() = (qpwork._h_scaled * qpresults._x);
				{ dual_residual_scaled += qpwork._residual_scaled_tmp.topRows(qpmodel._dim); }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, qpwork._residual_scaled_tmp.topRows(qpmodel._dim)});
				dual_feasibility_rhs_0 = infty_norm(qpwork._residual_scaled_tmp.topRows(qpmodel._dim));

				qpwork._residual_scaled_tmp.topRows(qpmodel._dim).noalias() = qpwork._a_scaled.transpose() * qpresults._y;
				{ dual_residual_scaled += qpwork._residual_scaled_tmp.topRows(qpmodel._dim); }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, qpwork._residual_scaled_tmp.topRows(qpmodel._dim)});
				dual_feasibility_rhs_1 = infty_norm(qpwork._residual_scaled_tmp.topRows(qpmodel._dim));

				qpwork._residual_scaled_tmp.topRows(qpmodel._dim).noalias() = qpwork._c_scaled.transpose() * qpresults._z;
				{ dual_residual_scaled += qpwork._residual_scaled_tmp.topRows(qpmodel._dim); }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, qpwork._residual_scaled_tmp.topRows(qpmodel._dim)});
				dual_feasibility_rhs_3 = infty_norm(qpwork._residual_scaled_tmp.topRows(qpmodel._dim));
			}

			dual_residual_unscaled = dual_residual_scaled;
			precond.unscale_dual_residual_in_place(
					VectorViewMut<T>{from_eigen, dual_residual_unscaled});

			dual_feasibility_lhs = infty_norm(dual_residual_unscaled);
        };


template<typename T> 
T oldNew_SaddlePoint(
			qp::Qpdata<T>& qpmodel,
			qp::Qpresults<T>& qpresults,
			qp::OldNew_Qpworkspace<T>& qpwork,
			VectorViewMut<T> prim_in_u,
			VectorViewMut<T> prim_in_l,
			VectorViewMut<T> prim_eq,
			VectorViewMut<T> dual_eq
			){
			
			auto z_ = qpresults._z;

			prim_in_u.to_eigen().noalias()-=  (z_*qpresults._mu_in_inv); 
			prim_in_l.to_eigen().noalias() -= (z_*qpresults._mu_in_inv) ; 
			T prim_eq_e = infty_norm(prim_eq.to_eigen()) ; 
			dual_eq.to_eigen().noalias() += (qpwork._c_scaled.transpose()*z_);
			T dual_e = infty_norm(dual_eq.to_eigen());
			T err = max2(prim_eq_e,dual_e);

			T prim_in_e(0);

			for (isize i = 0 ; i< qpmodel._n_in ; i=i+1){
				if (z_(i) >0){
					prim_in_e = max2(prim_in_e,std::abs(prim_in_u(i)));
				}else if (z_(i) < 0){
					prim_in_e = max2(prim_in_e,std::abs(prim_in_l(i)));
				}else{
					prim_in_e = max2(prim_in_e,max2(prim_in_u(i),T(0.))) ;
					prim_in_e = max2(prim_in_e, std::abs(std::min(prim_in_l(i),T(0.))));
				}
			}
			err = max2(err,prim_in_e);
			return err;
}


template<typename T>
void oldNew_newton_step_new(
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,

		T eps,
		Eigen::Matrix<T,Eigen::Dynamic,1>& z_pos,
		Eigen::Matrix<T,Eigen::Dynamic,1>& z_neg,
		Eigen::Matrix<T,Eigen::Dynamic,1>& res_y, 
		Eigen::Matrix<T,Eigen::Dynamic,1>& dual_for_eq,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& l_active_set_n_u,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& l_active_set_n_l,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& active_inequalities,

        Eigen::Matrix<T,Eigen::Dynamic,1>& dw_aug,
        ldlt::Ldlt<T>& ldl,
        const bool VERBOSE,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
		VectorViewMut<T> err_,
		std::string str,
		const bool chekNoAlias
	){

		l_active_set_n_u.noalias() = (z_pos.array() > 0).matrix();
		l_active_set_n_l.noalias() = (z_neg.array() < 0).matrix();

		active_inequalities.noalias() = l_active_set_n_u || l_active_set_n_l ; 

		isize num_active_inequalities = active_inequalities.count();
		isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + num_active_inequalities;

		rhs.setZero();
		dw_aug.setZero();
		
		{
				rhs.topRows(qpmodel._dim).noalias() -=  dual_for_eq ;
				for (isize j = 0 ; j < qpmodel._n_in; ++j){
					rhs.topRows(qpmodel._dim).noalias() -= qpresults._mu_in*(max2(z_pos(j),T(0.)) + std::min(z_neg(j),T(0.))) * qpwork._c_scaled.row(j) ; 
				}

		}
        qp::line_search::oldNew_active_set_change(
					qpmodel,
					qpresults,
					qpwork,
                    VectorView<bool>{from_eigen,active_inequalities},
                    ldl,
                    dw_aug);

		auto err = err_.to_eigen();
        oldNew_iterative_solve_with_permut_fact_new( //
					qpsettings,
					qpmodel,
					qpresults,
					qpwork,

                    VectorViewMut<T>{from_eigen,rhs.head(inner_pb_dim)},
                    VectorViewMut<T>{from_eigen,dw_aug},
                    eps,
                    inner_pb_dim,
                    ldl,
                    VERBOSE,
					VectorViewMut<T>{from_eigen,err.head(inner_pb_dim)},
					isize(1),
					str,
					chekNoAlias
					);

}

template<typename T,typename Preconditioner>
T oldNew_initial_guess(
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,
	
        VectorViewMut<T> ze,
		
		T eps_int,
		Preconditioner precond,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_eq,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& prim_in_u,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& prim_in_l,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& dual_for_eq,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& d_dual_for_eq,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& Cdx_,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& d_primal_residual_eq,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& l_active_set_n_u,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& l_active_set_n_l,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& active_inequalities,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& dw_aug,

        ldlt::Ldlt<T>& ldl,
        const bool VERBOSE,

		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& active_part_z,

		Eigen::Matrix<T, Eigen::Dynamic, 1>& tmp_u,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& tmp_l,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& aux_u,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& aux_l,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& dz_p,

		VectorViewMut<isize> _active_set_l,
		VectorViewMut<isize> _active_set_u,
		VectorViewMut<isize> _inactive_set,

		VectorViewMut<T> _tmp_d2_u,
		VectorViewMut<T> _tmp_d2_l,
		VectorViewMut<T> _tmp_d3,
		VectorViewMut<T> _tmp2_u,
		VectorViewMut<T> _tmp2_l,
		VectorViewMut<T> _tmp3_local_saddle_point,

		VectorViewMut<T> _err,

		std::vector<T>& alphas,

		std::string str,

		const bool chekNoAlias

		){

			auto z_e = ze.to_eigen().eval();
			prim_in_u.noalias() =  (qpwork._c_scaled*qpresults._x-qpwork._u_scaled) ; 
			prim_in_l.noalias() = (qpwork._c_scaled*qpresults._x-qpwork._l_scaled) ; 

			precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_u});
			precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_l});
			
			precond.unscale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, z_e}); 
			
			prim_in_u.noalias() += (z_e*qpresults._mu_in_inv) ; 

			prim_in_l.noalias() += (z_e*qpresults._mu_in_inv) ; 
			/*
			precond.unscale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, qpwork._ze}); 
			
			prim_in_u.noalias() += ( qpwork._ze*qpresults._mu_in_inv) ; 

			prim_in_l.noalias() += ( qpwork._ze*qpresults._mu_in_inv) ; 
			*/
			l_active_set_n_u = (prim_in_u.array() >= 0.).matrix();
			l_active_set_n_l = (prim_in_l.array() <= 0.).matrix();

			active_inequalities = l_active_set_n_u || l_active_set_n_l ;   

			
			prim_in_u.noalias() -= (z_e*qpresults._mu_in_inv) ; 
			prim_in_l.noalias() -= (z_e*qpresults._mu_in_inv) ; 
			
			/*
			prim_in_u.noalias() -= (qpwork._ze*qpresults._mu_in_inv) ; 
			prim_in_l.noalias() -= (qpwork._ze*qpresults._mu_in_inv) ; 
			*/
			precond.scale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_u});
			precond.scale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_l});
			
			precond.scale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, z_e});
			/*
			precond.scale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, qpwork._ze});
			*/
			isize num_active_inequalities = active_inequalities.count();
			isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + num_active_inequalities;

			rhs.setZero();
			active_part_z.setZero();
            qp::line_search::oldNew_active_set_change(
								qpmodel,
								qpresults,
								qpwork,
                                VectorView<bool>{from_eigen,active_inequalities},
                                ldl,
                                dw_aug);
			
			rhs.head(qpmodel._dim).noalias() = -dual_for_eq ;
			rhs.segment(qpmodel._dim,qpmodel._n_eq).noalias() = -primal_residual_eq ;
			for (isize i = 0; i < qpmodel._n_in; i++) {
				isize j = qpwork._current_bijection_map(i);
				if (j < qpresults._n_c) {
					if (l_active_set_n_u(i)) {
						rhs(j + qpmodel._dim + qpmodel._n_eq) = -prim_in_u(i);
					} else if (l_active_set_n_l(i)) {
						rhs(j + qpmodel._dim + qpmodel._n_eq) = -prim_in_l(i);
					}
				} else {
					rhs.head(qpmodel._dim).noalias() += qpresults._z(i) * qpwork._c_scaled.row(i); // unactive unrelevant columns
				}
			}	

			auto err_iter = _err.to_eigen();
            oldNew_iterative_solve_with_permut_fact_new( //
					qpsettings,
					qpmodel,
					qpresults,
					qpwork,
					
                    VectorViewMut<T>{from_eigen,rhs.head(inner_pb_dim)},
                    VectorViewMut<T>{from_eigen,dw_aug},
                    eps_int,
                    inner_pb_dim,
                    ldl,
                    VERBOSE,
					VectorViewMut<T>{from_eigen,err_iter.head(inner_pb_dim)},
					isize(2),
					str,
					chekNoAlias
					);
			
			// use active_part_z as a temporary variable to permut back dw_aug newton step
			for (isize j = 0; j < qpmodel._n_in; ++j) {
				isize i = qpwork._current_bijection_map(j);
				if (i < qpresults._n_c) {
					//dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
					
					active_part_z(j) = dw_aug(qpmodel._dim + qpmodel._n_eq + i);
					
				} else {
					//dw_aug_(j + dim + n_eq) = -z_(j);
					active_part_z(j) = -qpresults._z(j);
				}
			}
			dw_aug.tail(qpmodel._n_in) = active_part_z ;

			
			prim_in_u.noalias() += (z_e*qpresults._mu_in_inv) ; 
			prim_in_l.noalias() += (z_e*qpresults._mu_in_inv) ; 
			/*
			prim_in_u.noalias() += (qpwork._ze*qpresults._mu_in_inv) ; 
			prim_in_l.noalias() += (qpwork._ze*qpresults._mu_in_inv) ; 
			*/

			d_primal_residual_eq.noalias() = (qpwork._a_scaled*dw_aug.topRows(qpmodel._dim)- dw_aug.middleRows(qpmodel._dim,qpmodel._n_eq) * qpresults._mu_eq_inv).eval() ;
			d_dual_for_eq.noalias() = (qpwork._h_scaled*dw_aug.topRows(qpmodel._dim)+qpwork._a_scaled.transpose()*dw_aug.middleRows(qpmodel._dim,qpmodel._n_eq)+qpresults._rho*dw_aug.topRows(qpmodel._dim)).eval() ;
			Cdx_.noalias() = qpwork._c_scaled*dw_aug.topRows(qpmodel._dim) ;
			dual_for_eq.noalias() -= qpwork._c_scaled.transpose()*z_e ; 
			//dual_for_eq.noalias() -= C_.transpose()*qpwork._ze ; 

			T alpha_step = qp::line_search::oldNew_initial_guess_LS(
						qpsettings,
						qpmodel,
						qpresults,
						qpwork,
						ze.as_const(),
						VectorView<T>{from_eigen,dw_aug.tail(qpmodel._n_in)},
						VectorView<T>{from_eigen,prim_in_l},
						VectorView<T>{from_eigen,prim_in_u},
						VectorView<T>{from_eigen,Cdx_},
						VectorView<T>{from_eigen,d_dual_for_eq},
						VectorView<T>{from_eigen,dual_for_eq},
						VectorView<T>{from_eigen,d_primal_residual_eq},
						VectorView<T>{from_eigen,primal_residual_eq},

						active_part_z,
						tmp_u,
						tmp_l,
						aux_u,
						aux_l,
						rhs,
						dz_p,

						_active_set_l,
						_active_set_u,
						_inactive_set,

						_tmp_d2_u,
						_tmp_d2_l,
						_tmp_d3,
						_tmp2_u,
						_tmp2_l,
						_tmp3_local_saddle_point,
						
						alphas

			);
			
			std::cout << "alpha from initial guess " << alpha_step << std::endl;

			prim_in_u.noalias() += (alpha_step*Cdx_);
			prim_in_l.noalias() += (alpha_step*Cdx_);
			l_active_set_n_u.noalias() = (prim_in_u.array() >= 0.).matrix();
			l_active_set_n_l.noalias() = (prim_in_l.array() <= 0.).matrix();
			active_inequalities.noalias() = l_active_set_n_u || l_active_set_n_l ; 

			qpresults._x.noalias() += (alpha_step * dw_aug.topRows(qpmodel._dim)) ; 
			qpresults._y.noalias() += (alpha_step * dw_aug.middleRows(qpmodel._dim,qpmodel._n_eq)) ; 

			for (isize i = 0; i< qpmodel._n_in ; ++i){
				if (l_active_set_n_u(i)){
					qpresults._z(i) = std::max(qpresults._z(i)+alpha_step*dw_aug(qpmodel._dim+qpmodel._n_eq+i),T(0.)) ; 
				}else if (l_active_set_n_l(i)){
					qpresults._z(i) = std::min(qpresults._z(i)+alpha_step*dw_aug(qpmodel._dim+qpmodel._n_eq+i),T(0.)) ; 
				} else{
					qpresults._z(i) += alpha_step*dw_aug(qpmodel._dim+qpmodel._n_eq+i) ; 
				}
			}
			primal_residual_eq.noalias() += (alpha_step*d_primal_residual_eq);
			dual_for_eq.noalias() += alpha_step* (d_dual_for_eq) ;
			dw_aug.setZero();
			// TODO try for acceleration with a rhs relative error inside
			T err_saddle_point = oldNew_SaddlePoint(
				qpmodel,
				qpresults,
				qpwork,
				VectorViewMut<T>{from_eigen,prim_in_u},
				VectorViewMut<T>{from_eigen,prim_in_l},
				VectorViewMut<T>{from_eigen,primal_residual_eq},
				VectorViewMut<T>{from_eigen,dual_for_eq}
			);
			
			return err_saddle_point;
}


template<typename T>
T oldNew_correction_guess(
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,
		
		T eps_int,
		Eigen::Matrix<T,Eigen::Dynamic,1>& residual_in_y,
		Eigen::Matrix<T,Eigen::Dynamic,1>& z_pos,
		Eigen::Matrix<T,Eigen::Dynamic,1>& z_neg,
		Eigen::Matrix<T,Eigen::Dynamic,1>& dual_for_eq,
		Eigen::Matrix<T,Eigen::Dynamic,1>& Hdx,
		Eigen::Matrix<T,Eigen::Dynamic,1>& Adx,
		Eigen::Matrix<T,Eigen::Dynamic,1>& Cdx,
		Eigen::Matrix<bool,Eigen::Dynamic,1>& l_active_set_n_u,
		Eigen::Matrix<bool,Eigen::Dynamic,1>& l_active_set_n_l,
		Eigen::Matrix<bool,Eigen::Dynamic,1>& active_inequalities,

        Eigen::Matrix<T, Eigen::Dynamic, 1>& dw_aug,
        ldlt::Ldlt<T>& ldl,
        const bool VERBOSE,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,

		VectorViewMut<T> _tmp1,
		VectorViewMut<T> _tmp2,
		VectorViewMut<T> _tmp3,
		VectorViewMut<T> _tmp4,

		VectorViewMut<T> _tmp_u,
		VectorViewMut<T> _tmp_l,
		VectorViewMut<isize> _active_set_u,
		VectorViewMut<isize> _active_set_l,

		VectorViewMut<T> _tmp_a0_u,
		VectorViewMut<T> _tmp_b0_u,
		VectorViewMut<T> _tmp_a0_l,
		VectorViewMut<T> _tmp_b0_l,

		VectorViewMut<T> err_,

		std::vector<T>& alphas,

		Eigen::Matrix<T, Eigen::Dynamic, 1>& aux_u,

		std::string str,

		const bool checkNoAlias

		){

		T err_in = 1.e6;
		auto tmp1 = _tmp1.to_eigen();
		auto tmp2 = _tmp2.to_eigen();
		auto tmp3 = _tmp3.to_eigen();
		auto grad_n = _tmp4.to_eigen();

		for (i64 iter = 0; iter <= qpsettings._max_iter_in; ++iter) {

			if (iter == qpsettings._max_iter_in) {
				qpresults._n_tot += qpsettings._max_iter_in;
				break;
			}
			
			qp::detail::oldNew_newton_step_new<T>(
											qpsettings,
											qpmodel,
											qpresults,
											qpwork,

											eps_int,
											z_pos,
											z_neg,
											residual_in_y,
											dual_for_eq,
											l_active_set_n_u,
											l_active_set_n_l,
											active_inequalities,
                                            dw_aug,
                                            ldl,
                                            VERBOSE,
											rhs,
											err_,
											str,
											checkNoAlias
			);
			T alpha_step = 1;
			Hdx.noalias() = qpwork._h_scaled * dw_aug.head(qpmodel._dim) ; 
			Adx.noalias() = qpwork._a_scaled * dw_aug.head(qpmodel._dim) ; 
			Cdx.noalias() = qpwork._c_scaled * dw_aug.head(qpmodel._dim) ; 
			if (qpmodel._n_in > 0){
				alpha_step = qp::line_search::oldNew_correction_guess_LS(
										qpmodel,
										qpresults,
										qpwork,
										Hdx,
									 	VectorView<T>{from_eigen,dw_aug.head(qpmodel._dim)},
										Adx,  
										Cdx,
										residual_in_y,
										z_pos,
										z_neg,
										
										_tmp_u,
										_tmp_l,
										_active_set_u,
										_active_set_l,

										_tmp_a0_u,
										_tmp_b0_u,
										_tmp_a0_l,
										_tmp_b0_l,
										l_active_set_n_u,
										l_active_set_n_l,

										alphas,

										aux_u
						
				) ;
			}
			if (infty_norm(alpha_step * dw_aug.head(qpmodel._dim))< 1.E-11){
				qpresults._n_tot += iter+1;
				std::cout << "infty_norm(alpha_step * dx) " << infty_norm(alpha_step * dw_aug.head(qpmodel._dim)) << std::endl;
				break;
			}
			
			qpresults._x.noalias()+= (alpha_step *dw_aug.head(qpmodel._dim)) ; 
			z_pos.noalias() += (alpha_step *Cdx) ;
			z_neg.noalias() += (alpha_step *Cdx); 
			residual_in_y.noalias() += (alpha_step * Adx);
 			qpresults._y.noalias() = qpresults._mu_eq *  residual_in_y  ;
			dual_for_eq.noalias() += (alpha_step * ( qpresults._mu_eq * qpwork._a_scaled.transpose() * Adx + qpresults._rho * dw_aug.head(qpmodel._dim) + Hdx  )) ;
			for (isize j = 0 ; j < qpmodel._n_in; ++j){
				qpresults._z(j) = qpresults._mu_in*(max2(z_pos(j),T(0)) + std::min(z_neg(j),T(0))); 
			}

			tmp1.noalias() = qpwork._h_scaled *qpresults._x ;
			tmp2.noalias() = qpwork._a_scaled.transpose() * ( qpresults._y );
			tmp3.noalias() = qpwork._c_scaled.transpose() * ( qpresults._z )   ; 
			grad_n.noalias() = tmp1 + tmp2 + tmp3  + qpwork._g_scaled + qpresults._rho* (qpresults._x-qpwork._xe) ;

			err_in = infty_norm(  grad_n  );
			std::cout << "---it in " << iter << " projection norm " << err_in << " alpha " << alpha_step << " rhs " << eps_int * (1 + max2(max2(max2(infty_norm(tmp1), infty_norm(tmp2)), infty_norm(tmp3)), infty_norm(qpwork._g_scaled) ))   <<  std::endl;

			if (err_in<= eps_int * (1. + max2(max2(max2(infty_norm(tmp1), infty_norm(tmp2)), infty_norm(tmp3)), infty_norm(qpwork._g_scaled)) )  ){
				qpresults._n_tot +=iter+1;
				break;
			}
		}
	
		return err_in;

}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpSolveStats oldNew_qpSolve( //
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::OldNew_Qpworkspace<T>& qpwork,
		std::string str,
		Preconditioner precond = Preconditioner{},
		const bool chekNoAlias = false) {

	using namespace ldlt::tags;
    static constexpr Layout layout = rowmajor;
    static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
    const bool VERBOSE = true;


	T machine_eps = std::numeric_limits<T>::epsilon();

	T bcl_eta_ext_init = pow(T(0.1),qpsettings._alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(qpsettings._eps_abs,T(1.E-9));
	
	//// 4/ malloc for no allocation
	/// 3/ structure préallouée QPData, QPSettings, QPResults
	/// 5/ load maros problems from c++ parser

	Eigen::Matrix<T, Eigen::Dynamic, 1> dw_aug(qpmodel._n_total);
	Eigen::Matrix<T, Eigen::Dynamic, 1> rhs(qpmodel._n_total);
	Eigen::Matrix<T, Eigen::Dynamic, 1> active_part_z(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> d_dual_for_eq(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> Cdx_(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> d_primal_residual_eq(qpmodel._dim);
	Eigen::Matrix<bool, Eigen::Dynamic, 1> l_active_set_n_u(qpmodel._n_in);
	Eigen::Matrix<bool, Eigen::Dynamic, 1> l_active_set_n_l(qpmodel._n_in);
	Eigen::Matrix<bool, Eigen::Dynamic, 1> active_inequalities(qpmodel._n_in);

	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_u(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_l(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> aux_u(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> aux_l(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> dz_p(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp1(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp2(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp3(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp4(qpmodel._dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_d2_u(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_d2_l(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_d3(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp2_u(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp2_l(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp3_local_saddle_point(qpmodel._n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> err(qpmodel._n_total);
	Eigen::Matrix<isize, Eigen::Dynamic, 1> active_set_l(qpmodel._n_in);
	Eigen::Matrix<isize, Eigen::Dynamic, 1> active_set_u(qpmodel._n_in);
	Eigen::Matrix<isize, Eigen::Dynamic, 1> inactive_set(qpmodel._n_in);

	RowMat test(2,2); // test it is full of nan for debug
	std::cout << "test " << test << std::endl;

	std::vector<T> alphas;
	alphas.reserve( 3*qpmodel._n_in );

	T primal_feasibility_rhs_1_eq = infty_norm(qpmodel._b);
    T primal_feasibility_rhs_1_in_u = infty_norm(qpmodel._u);
    T primal_feasibility_rhs_1_in_l = infty_norm(qpmodel._l);
	T dual_feasibility_rhs_2 = infty_norm(qpmodel._g);

	qpwork._h_scaled = qpmodel._H;
	qpwork._g_scaled = qpmodel._g;
	qpwork._a_scaled = qpmodel._A;
	qpwork._b_scaled = qpmodel._b;
    qpwork._c_scaled = qpmodel._C;
    qpwork._u_scaled = qpmodel._u;
    qpwork._l_scaled = qpmodel._l;

    qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen,qpwork._h_scaled},
			{from_eigen,qpwork._g_scaled},
			{from_eigen,qpwork._a_scaled},
			{from_eigen,qpwork._b_scaled},
			{from_eigen,qpwork._c_scaled},
			{from_eigen,qpwork._u_scaled},
			{from_eigen,qpwork._l_scaled}
	};

    
	precond.scale_qp_in_place(qp_scaled,VectorViewMut<T>{from_eigen,dw_aug});
    dw_aug.setZero();
	
	////

	qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim) = qpwork._h_scaled ;
	qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim).diagonal().array() += qpresults._rho;	
	qpwork._kkt.block(0, qpmodel._dim, qpmodel._dim, qpmodel._n_eq) = qpwork._a_scaled.transpose();
	qpwork._kkt.block(qpmodel._dim, 0, qpmodel._n_eq, qpmodel._dim) = qpwork._a_scaled;
	qpwork._kkt.bottomRightCorner(qpmodel._n_eq, qpmodel._n_eq).setZero();
	qpwork._kkt.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults._mu_eq_inv); // mu stores the inverse of mu

	ldlt::Ldlt<T> ldl{decompose, qpwork._kkt};
	rhs.head(qpmodel._dim) = -qpwork._g_scaled;
	rhs.segment(qpmodel._dim,qpmodel._n_eq) = qpwork._b_scaled;
	
    oldNew_iterative_solve_with_permut_fact_new( //
		qpsettings,
		qpmodel,
		qpresults,
		qpwork,

		VectorViewMut<T>{from_eigen,rhs.head(qpmodel._dim+qpmodel._n_eq)},
        VectorViewMut<T>{from_eigen,dw_aug},
		bcl_eta_in,
		qpmodel._dim+qpmodel._n_eq,
        ldl,
        VERBOSE,
		VectorViewMut<T>{from_eigen,err.head(qpmodel._dim+qpmodel._n_eq)},
		isize(0),
		str,
		chekNoAlias
        );
	
	qpresults._x = dw_aug.head(qpmodel._dim);
	qpresults._y = dw_aug.segment(qpmodel._dim,qpmodel._n_eq);
	
	dw_aug.setZero();

	auto dual_residual_scaled = qpwork._residual_scaled.topRows(qpmodel._dim).eval();
	auto primal_residual_eq_scaled = qpwork._residual_scaled.middleRows(qpmodel._dim,qpmodel._n_eq).eval();
	auto primal_residual_in_scaled_u = qpwork._residual_scaled.bottomRows(qpmodel._n_in).eval();
	auto primal_residual_in_scaled_l = qpwork._residual_scaled.bottomRows(qpmodel._n_in).eval();
	auto dual_residual_unscaled = qpwork._residual_unscaled.topRows(qpmodel._dim).eval();
	auto primal_residual_eq_unscaled = qpwork._residual_unscaled.middleRows(qpmodel._dim,qpmodel._n_eq).eval();
	auto primal_residual_in_u_unscaled = qpwork._residual_unscaled.bottomRows(qpmodel._n_in).eval();
	auto primal_residual_in_l_unscaled = qpwork._residual_unscaled.bottomRows(qpmodel._n_in).eval();
	
	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);
	T mu_fact_update(10);
	T mu_fact_update_inv(0.1);
	
	for (i64 iter = 0; iter <= qpsettings._max_iter; ++iter) {
		qpresults._n_ext +=1;
		if (iter == qpsettings._max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::oldNew_global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs,
				primal_residual_eq_scaled,
				primal_residual_in_scaled_u,
				primal_residual_in_scaled_l,
				primal_residual_eq_unscaled,
				primal_residual_in_l_unscaled,
				primal_residual_in_u_unscaled,
				precond
		);
		qp::detail::oldNew_global_dual_residual(
			qpmodel,
			qpresults,
			qpwork,
			dual_feasibility_lhs,
			dual_feasibility_rhs_0,
			dual_feasibility_rhs_1,
        	dual_feasibility_rhs_3,
			dual_residual_scaled,
			dual_residual_unscaled,
			precond
		);
		
		std::cout << "---------------it : " << iter << " primal residual : " << primal_feasibility_lhs << " dual residual : " << dual_feasibility_lhs << std::endl;
		std::cout << "bcl_eta_ext : " << bcl_eta_ext << " bcl_eta_in : " << bcl_eta_in <<  " rho : " << qpresults._rho << " bcl_mu_eq : " << qpresults._mu_eq << " bcl_mu_in : " << qpresults._mu_in <<std::endl;

		bool is_primal_feasible =
				primal_feasibility_lhs <=
				(qpsettings._eps_abs + qpsettings._eps_rel * max2(
                                          max2(
																 primal_feasibility_eq_rhs_0,
                                                                 primal_feasibility_in_rhs_0),
                                          max2(
										                    max2(
                                                                 primal_feasibility_rhs_1_eq,
                                                                 primal_feasibility_rhs_1_in_u
                                                                ),
                                                            primal_feasibility_rhs_1_in_l
                                              ) 
                                         
                                        ));

		bool is_dual_feasible =
				dual_feasibility_lhs <=
				(qpsettings._eps_abs + qpsettings._eps_rel * max2(                      
                                                                max2(   dual_feasibility_rhs_3,
																        dual_feasibility_rhs_0
                                                                ),
																max2( //
																		 dual_feasibility_rhs_1,
																		 dual_feasibility_rhs_2
																	)
										  )
																		 );

		if (is_primal_feasible){
			
			if (dual_feasibility_lhs >= qpsettings._refactor_dual_feasibility_threshold && qpresults._rho != qpsettings._refactor_rho_threshold){

				T rho_new(qpsettings._refactor_rho_threshold);
				oldNew_refactorize(
						qpmodel,
						qpresults,
						qpwork,

						rho_new,
						ldl,
						VectorViewMut<T>{from_eigen,dw_aug}
						);

				qpresults._rho = rho_new;
			}
			if (is_dual_feasible){
				{
				LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
				precond.unscale_primal_in_place(VectorViewMut<T>{from_eigen,qpresults._x}); 
				precond.unscale_dual_in_place_eq(VectorViewMut<T>{from_eigen,qpresults._y});
				precond.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen,qpresults._z});
				}
				return {qpresults._n_ext, qpresults._n_mu_change,qpresults._n_tot};
			}
		}
		
		
		qpwork._xe = qpresults._x.eval(); 
		qpwork._ye = qpresults._y.eval(); 
		qpwork._ze = qpresults._z.eval(); 
		

		const bool do_initial_guess_fact = (primal_feasibility_lhs < qpsettings._eps_IG || qpmodel._n_in == 0 ) ;

		T err_in(0.);

		if (do_initial_guess_fact){

			err_in = qp::detail::oldNew_initial_guess<T,Preconditioner>(
							qpsettings,
							qpmodel,
							qpresults,
							qpwork,

							VectorViewMut<T>{from_eigen,qpwork._ze},
							
							bcl_eta_in,
							precond,
							primal_residual_eq_scaled,
							primal_residual_in_scaled_u,
							primal_residual_in_scaled_l,
							dual_residual_scaled,
							d_dual_for_eq,
							Cdx_,
							d_primal_residual_eq,
							l_active_set_n_u,
							l_active_set_n_l,
							active_inequalities,
							dw_aug,

							ldl,
							VERBOSE,
							rhs,
							active_part_z,

							tmp_u,
							tmp_l,
							aux_u,
							aux_l,
							dz_p,

							VectorViewMut<isize>{from_eigen,active_set_l},
							VectorViewMut<isize>{from_eigen,active_set_u},
							VectorViewMut<isize>{from_eigen,inactive_set},

							VectorViewMut<T>{from_eigen,tmp_d2_u},
							VectorViewMut<T>{from_eigen,tmp_d2_l},
							VectorViewMut<T>{from_eigen,tmp_d3},
							VectorViewMut<T>{from_eigen,tmp2_u},
							VectorViewMut<T>{from_eigen,tmp2_l},
							VectorViewMut<T>{from_eigen,tmp3_local_saddle_point},

							VectorViewMut<T>{from_eigen,err},
							alphas,
							
							str,
							chekNoAlias

			);
			qpresults._n_tot +=1;

		}

		bool do_correction_guess = (!do_initial_guess_fact && qpmodel._n_in != 0) ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in && qpmodel._n_in != 0) ;
		
		std::cout << " error from initial guess : " << err_in << " bcl_eta_in " << bcl_eta_in << std::endl;
		
		if ((do_initial_guess_fact && err_in >= bcl_eta_in && qpmodel._n_in != 0)){

			dual_residual_scaled.noalias() += (-qpwork._c_scaled.transpose()*qpresults._z + qpresults._mu_eq * qpwork._a_scaled.transpose()*primal_residual_eq_scaled );
			primal_residual_eq_scaled.noalias()  += (qpresults._y*qpresults._mu_eq_inv);
			primal_residual_in_scaled_u.noalias()  += (qpresults._z*qpresults._mu_in_inv);
			primal_residual_in_scaled_l.noalias() += (qpresults._z*qpresults._mu_in_inv); //

		}
		if (!do_initial_guess_fact && qpmodel._n_in != 0){ // y=ye, x=xe, 

			primal_residual_eq_scaled.noalias()  += (qpwork._ye*qpresults._mu_eq_inv); // Axe-b+ye/mu_eq
			primal_residual_eq_scaled.noalias()  -= (qpresults._y*qpresults._mu_eq_inv);

			primal_residual_in_scaled_u.noalias()  = qpwork._c_scaled * qpresults._x + qpwork._ze*qpresults._mu_in_inv;
			primal_residual_in_scaled_l.noalias()  = primal_residual_in_scaled_u ;
			primal_residual_in_scaled_u.noalias()  -= qpwork._u_scaled;
			primal_residual_in_scaled_l.noalias()  -= qpwork._l_scaled;

			dual_residual_scaled.noalias() += -qpwork._c_scaled.transpose()*qpresults._z;
			dual_residual_scaled.noalias() += qpresults._mu_eq * qpwork._a_scaled.transpose()*primal_residual_eq_scaled ; // no need of rho * (x-xe) as x=xe
			primal_residual_eq_scaled.noalias()  += (qpresults._y*qpresults._mu_eq_inv);

		}

		if (do_correction_guess){
			
			err_in = qp::detail::oldNew_correction_guess(
						qpsettings,
						qpmodel,
						qpresults,
						qpwork,
						
						bcl_eta_in,
						primal_residual_eq_scaled,
						primal_residual_in_scaled_u,
						primal_residual_in_scaled_l,

						dual_residual_scaled,
						d_dual_for_eq,
						d_primal_residual_eq,
						Cdx_,
						l_active_set_n_u,
						l_active_set_n_l,
						active_inequalities,

                        dw_aug,
                        ldl,
                        VERBOSE,
						rhs,

						VectorViewMut<T>{from_eigen,tmp1},
						VectorViewMut<T>{from_eigen,tmp2},
						VectorViewMut<T>{from_eigen,tmp3},
						VectorViewMut<T>{from_eigen,tmp4},

						VectorViewMut<T>{from_eigen,tmp_u},
						VectorViewMut<T>{from_eigen,tmp_l},
						VectorViewMut<isize>{from_eigen,active_set_u},
						VectorViewMut<isize>{from_eigen,active_set_l},

						VectorViewMut<T>{from_eigen,tmp_d2_u},
						VectorViewMut<T>{from_eigen,tmp2_u},
						VectorViewMut<T>{from_eigen,tmp_d2_l},
						VectorViewMut<T>{from_eigen,tmp2_l},

						VectorViewMut<T>{from_eigen,err},

						alphas,

						aux_u,

						str,
						chekNoAlias


			);
			std::cout << " error from correction guess : " << err_in << std::endl;
		}
		
		T primal_feasibility_lhs_new(primal_feasibility_lhs) ; 

		qp::detail::oldNew_global_primal_residual(
						qpmodel,
						qpresults,
						qpwork,
						primal_feasibility_lhs_new,
						primal_feasibility_eq_rhs_0,
						primal_feasibility_in_rhs_0,
						primal_feasibility_eq_lhs,
						primal_feasibility_in_lhs,
						primal_residual_eq_scaled,
						primal_residual_in_scaled_u,
						primal_residual_in_scaled_l,
						primal_residual_eq_unscaled,
						primal_residual_in_l_unscaled,
						primal_residual_in_u_unscaled,
						precond
		);

		qp::detail::oldNew_BCL_update(
					qpsettings,
					qpmodel,
					qpresults,
					qpwork,
					primal_feasibility_lhs_new,
					VectorViewMut<T>{from_eigen,primal_residual_in_scaled_u},
					VectorViewMut<T>{from_eigen,primal_residual_in_scaled_l},
					VectorViewMut<T>{from_eigen,primal_residual_eq_scaled},
					precond,
					bcl_eta_ext,
					bcl_eta_in,

					VectorViewMut<T>{from_eigen,dw_aug},
					ldl,
					bcl_eta_ext_init,
					eps_in_min
		);

		// COLD RESTART
		
		T dual_feasibility_lhs_new(dual_feasibility_lhs) ; 
		
		qp::detail::oldNew_global_dual_residual(
			qpmodel,
			qpresults,
			qpwork,
			dual_feasibility_lhs_new,
			dual_feasibility_rhs_0,
			dual_feasibility_rhs_1,
        	dual_feasibility_rhs_3,
			dual_residual_scaled,
			dual_residual_unscaled,
			precond
		);

		if ((primal_feasibility_lhs_new / max2(primal_feasibility_lhs,machine_eps) >= 1.) && (dual_feasibility_lhs_new / max2(primal_feasibility_lhs,machine_eps) >= 1.) && qpresults._mu_in >= 1.E5){
			std::cout << "cold restart" << std::endl;

			qp::detail::oldNew_mu_update(
				qpmodel,
				qpresults,
				ldl,
				VectorViewMut<T>{from_eigen,dw_aug},
				qpsettings._cold_reset_mu_eq_inv,
				qpsettings._cold_reset_mu_in_inv);
			
			qpresults._mu_in = qpsettings._cold_reset_mu_in;
			qpresults._mu_eq = qpsettings._cold_reset_mu_eq;
			qpresults._mu_in_inv = qpsettings._cold_reset_mu_in_inv;
			qpresults._mu_eq_inv = qpsettings._cold_reset_mu_eq_inv;
			qpresults._n_mu_change+=1;

		}
			
	}
	
	return {qpsettings._max_iter, qpresults._n_mu_change, qpresults._n_tot};
}


} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS */
