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
		qp::QpViewBox<T> qp_scaled,
		T rho_new,
		T rho_old,
		ldlt::Ldlt<T>& ldl,
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& kkt,
		T mu_eq,
		T mu_in,
		T mu_eq_inv,
		T mu_in_inv,
		isize dim,
		isize n_eq,
		isize n_in,
		isize n_c,
		VectorViewMut<isize> current_bijection_map_,
		VectorViewMut<T> dw_aug_
		) {
		
	auto dw_aug = dw_aug_.to_eigen();
	dw_aug.setZero();
	auto current_bijection_map = current_bijection_map_.to_eigen();
	kkt.diagonal().head(dim).array() += rho_new - rho_old; 
	kkt.diagonal().segment(dim,n_eq).array() = -T(1)/mu_eq; 
	ldl.factorize(kkt);

	for (isize j = 0; j < n_c; ++j) {
		for (isize i = 0; i < n_in; ++i) {
			if (j == current_bijection_map(i)) {
					dw_aug.head(dim) = qp_scaled.C.to_eigen().row(i);
					dw_aug(dim + n_eq + j) = - T(1)/mu_in; // mu_in stores the inverse of mu_in
					ldl.insert_at(n_eq + dim + j, dw_aug.head(dim+n_eq+n_c));
					dw_aug(dim + n_eq + j) = T(0);
			}
		}
	}
	dw_aug.setZero();
}

template <typename T>
void oldNew_mu_update(
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<T>  dw_aug_,
		T mu_eq_inv,
		T mu_in_inv,
		T mu_eq_new_inv,
		T mu_in_new_inv,
		isize dim,
		isize n_eq,
		isize n_in,
		isize n_c) {
	T diff = 0;
	auto dw_aug = dw_aug_.to_eigen();

	dw_aug.head(dim+n_eq+n_c).setZero();
	if (n_eq > 0) {
		diff = mu_eq_inv -  mu_eq_new_inv; // mu stores the inverse of mu

		for (isize i = 0; i < n_eq; i++) {
			dw_aug(dim + i) = T(1);
			ldl.rank_one_update(dw_aug.head(dim+n_eq+n_c), diff);
			dw_aug(dim + i) = T(0);
		}
	}
	if (n_c > 0) {
		diff = mu_in_inv - mu_in_new_inv; // mu stores the inverse of mu
		for (isize i = 0; i < n_c; i++) {
			dw_aug(dim + n_eq + i) = T(1);
			ldl.rank_one_update(dw_aug.head(dim+n_eq+n_c), diff);
			dw_aug(dim + n_eq + i) = T(0);
		}
	}
}

template <typename T>
void oldNew_iterative_residual(
		qp::QpViewBox<T> qp_scaled,
		T rho,
        T mu_eq_inv,
        T mu_in_inv,
		VectorViewMut<T> rhs_,
        VectorViewMut<T>  dw_aug_,
        VectorViewMut<T>  err_,
        VectorViewMut<isize> current_bijection_map_,
		isize inner_pb_dim,
        isize dim,
        isize n_eq,
        isize n_in,
        isize n_c,
		isize it_,
		std::string str,
		const bool chekNoAlias) {
 
    auto rhs = rhs_.to_eigen();
    auto dw_aug = dw_aug_.to_eigen();
    auto err = err_.to_eigen();
    auto current_bijection_map = current_bijection_map_.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();

	err.head(inner_pb_dim).noalias()  = rhs.head(inner_pb_dim);
	if (chekNoAlias){
		saveData("_err_after_copy"+std::to_string(it_)+str,err.eval());
	}

	err.head(dim).noalias()  -= qp_scaled.H.to_eigen() * dw_aug.head(dim);
	if (chekNoAlias){
		saveData("_err_after_H"+std::to_string(it_)+str,err.eval());
	}
	
    err.head(dim).noalias()  -= rho * dw_aug.head(dim);
	if (chekNoAlias){
		saveData("_err_after_rho"+std::to_string(it_)+str,err.eval()); 
	}

    err.head(dim).noalias()  -= A_.transpose() * dw_aug.segment(dim, n_eq);
	if (chekNoAlias){
		saveData("_err_after_Ahead"+std::to_string(it_)+str,err.eval());
	}

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			err.head(dim).noalias()  -= dw_aug(dim + n_eq + j) * C_.row(i);
			err(dim + n_eq + j) -=
					(C_.row(i).dot(dw_aug.head(dim)) -
					 dw_aug(dim + n_eq + j)  * mu_in_inv); // mu stores the inverse of mu
		}
	}
	if (chekNoAlias){
		saveData("_err_after_act"+std::to_string(it_)+str,err.eval());
	}

	err.segment(dim, n_eq).noalias()  -= (A_ * dw_aug.head(dim) - dw_aug.segment(dim, n_eq) * mu_eq_inv); // mu stores the inverse of mu

}

template <typename T>
void oldNew_iterative_solve_with_permut_fact_new( //
        qp::QpViewBox<T> qp_scaled,
		T rho,
        T mu_eq,
        T mu_in,
        T mu_eq_inv,
        T mu_in_inv,
		VectorViewMut<T> rhs_,
        VectorViewMut<T>  dw_aug_,
        VectorViewMut<isize> current_bijection_map_,
		T eps,
		isize inner_pb_dim,
        isize dim,
        isize n_eq,
        isize n_in,
        isize& n_c,
        ldlt::Ldlt<T>& ldl,
        const bool VERBOSE,
        isize nb_iterative_refinement,
		VectorViewMut<T> _err,
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& kkt,
		T eps_refact,
		isize it_,
		std::string str,
		const bool chekNoAlias
        ){

    auto rhs = rhs_.to_eigen();
    auto dw_aug = dw_aug_.to_eigen();
    auto err = _err.to_eigen();
	err.setZero();
    auto current_bijection_map = current_bijection_map_.to_eigen();
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
					qp_scaled,
                    rho,
                    mu_eq_inv,
                    mu_in_inv,
                    rhs_,
                    dw_aug_,
                    _err,
                    current_bijection_map_,
                    inner_pb_dim,
                    dim,
                    n_eq,
                    n_in,
                    n_c,
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
		if (it >= nb_iterative_refinement) {
			break;
		} 
		++it;
		ldl.solve_in_place( err.head(inner_pb_dim));
		dw_aug.head(inner_pb_dim).noalias() +=  err.head(inner_pb_dim);

		err.head(inner_pb_dim).setZero();
		qp::detail::oldNew_iterative_residual<T>(
					qp_scaled,
                    rho,
                    mu_eq_inv,
                    mu_in_inv,
                    rhs_,
                    dw_aug_,
                    _err,
                    current_bijection_map_,
                    inner_pb_dim,
                    dim,
                    n_eq,
                    n_in,
                    n_c,
					it_,
					str,
					chekNoAlias);

		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(err.head(inner_pb_dim)) << std::endl;
		}
	}
	if (qp::infty_norm(err.head(inner_pb_dim))>= std::max(eps,eps_refact)){
		{
			
			LDLT_MULTI_WORKSPACE_MEMORY(
				(_htot,Uninit, Mat(dim+n_eq+n_c, dim+n_eq+n_c),LDLT_CACHELINE_BYTES, T)
				);
			auto Htot = _htot.to_eigen().eval();

			Htot.setZero();
			
			kkt.diagonal().segment(dim,n_eq).array() = -mu_eq_inv; 
			Htot.topLeftCorner(dim+n_eq, dim+n_eq) = kkt;

			Htot.diagonal().segment(dim+n_eq,n_c).array() = -mu_in_inv; 

			for (isize i = 0; i< n_in ; ++i){
					
					isize j = current_bijection_map(i);
					if (j<n_c){
						Htot.block(j+dim+n_eq,0,1,dim) = qp_scaled.C.to_eigen().row(i) ; 
						Htot.block(0,j+dim+n_eq,dim,1) = qp_scaled.C.to_eigen().transpose().col(i) ; 
					}
			}
			
			oldNew_refactorize(
						qp_scaled,
						rho,
						rho,
						ldl,
						kkt,
						mu_eq,
						mu_in,
						mu_eq_inv,
						mu_in_inv,
						dim,
						n_eq,
						n_in,
						n_c,
						VectorViewMut<isize>{from_eigen,current_bijection_map},
						VectorViewMut<T>{from_eigen,dw_aug}
						);
			
			std::cout << " ldl.reconstructed_matrix() - Htot " << infty_norm(ldl.reconstructed_matrix() - Htot)<< std::endl;
		}
		it = 0;
		dw_aug.head(inner_pb_dim) = rhs.head(inner_pb_dim);

		ldl.solve_in_place(dw_aug.head(inner_pb_dim));

		qp::detail::oldNew_iterative_residual<T>(
					qp_scaled,
                    rho,
                    mu_eq_inv,
                    mu_in_inv,
                    rhs_,
                    dw_aug_,
                    _err,
                    current_bijection_map_,
                    inner_pb_dim,
                    dim,
                    n_eq,
                    n_in,
                    n_c,
					it_,
					str,
					chekNoAlias);
		++it;
		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm( err.head(inner_pb_dim)) << std::endl;
		}
		while (infty_norm( err.head(inner_pb_dim)) >= eps) {
			if (it >= nb_iterative_refinement) {
				break;
			}
			++it;
			ldl.solve_in_place( err.head(inner_pb_dim) );
			dw_aug.head(inner_pb_dim).noalias()  +=  err.head(inner_pb_dim);
  
			err.head(inner_pb_dim).setZero();
			qp::detail::oldNew_iterative_residual<T>(
					qp_scaled,
                    rho,
                    mu_eq_inv,
                    mu_in_inv,
                    rhs_,
                    dw_aug_,
                    _err,
                    current_bijection_map_,
                    inner_pb_dim,
                    dim,
                    n_eq,
                    n_in,
                    n_c,
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
		T& primal_feasibility_lhs,
		VectorViewMut<T> primal_residual_in_scaled_u,
		VectorViewMut<T> primal_residual_in_scaled_l,
		VectorViewMut<T> primal_residual_eq_scaled,
		Preconditioner& precond,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		isize& n_mu_updates,
		T& bcl_mu_in,
		T& bcl_mu_eq,
		T& bcl_mu_in_inv,
		T& bcl_mu_eq_inv,

		VectorViewMut<T> ye,
		VectorViewMut<T> ze,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		VectorViewMut<T> dw_aug_,
		ldlt::Ldlt<T>& ldl,
		isize dim,
		isize n_eq,
		isize n_in,
		isize& n_c,
		T mu_max_in,
		T mu_max_eq,
		T mu_max_in_inv,
		T mu_max_eq_inv,
		T mu_fact_update,
		T mu_fact_update_inv,
		T bcl_eta_ext_init,
		T beta,
		T alpha
		){
		precond.scale_primal_residual_in_place_eq(primal_residual_eq_scaled);
		precond.scale_primal_residual_in_place_in(primal_residual_in_scaled_l);
		precond.scale_primal_residual_in_place_in(primal_residual_in_scaled_u);
		T primal_feasibility_eq_lhs = infty_norm(primal_residual_eq_scaled.to_eigen());
		T primal_feasibility_in_lhs = max2(infty_norm(primal_residual_in_scaled_l.to_eigen()),infty_norm(primal_residual_in_scaled_u.to_eigen()));
		T tmp = max2(primal_feasibility_eq_lhs,primal_feasibility_in_lhs);
		//std::cout << "tmp for BCL " << tmp << std::endl;
		if (tmp <= bcl_eta_ext) {
			std::cout << "good step"<< std::endl;
			bcl_eta_ext = bcl_eta_ext * pow(bcl_mu_in_inv, beta);
			bcl_eta_in = max2(bcl_eta_in * bcl_mu_in_inv,eps_abs);
		} else {
			std::cout << "bad step"<< std::endl; 
			y.to_eigen() = ye.to_eigen();
			z.to_eigen() = ze.to_eigen();
			T new_bcl_mu_in(std::min(bcl_mu_in * mu_fact_update, mu_max_in));
			T new_bcl_mu_eq(std::min(bcl_mu_eq * mu_fact_update, mu_max_eq));
			T new_bcl_mu_in_inv(max2(bcl_mu_in_inv * mu_fact_update_inv, mu_max_in_inv)); // mu stores the inverse of mu
			T new_bcl_mu_eq_inv(max2(bcl_mu_eq_inv * mu_fact_update_inv, mu_max_eq_inv)); // mu stores the inverse of mu


			if (bcl_mu_in != new_bcl_mu_in || bcl_mu_eq != new_bcl_mu_eq) {
					{
					++n_mu_updates;
					}
			}	
			qp::detail::oldNew_mu_update(
				ldl,
				dw_aug_,
				bcl_mu_eq_inv,
				bcl_mu_in_inv,
				new_bcl_mu_eq_inv,
				new_bcl_mu_in_inv,
				dim,
				n_eq,
				n_in,
				n_c);
			bcl_mu_eq = new_bcl_mu_eq;
			bcl_mu_in = new_bcl_mu_in;
			bcl_mu_eq_inv = new_bcl_mu_eq_inv;
			bcl_mu_in_inv = new_bcl_mu_in_inv;
			bcl_eta_ext = bcl_eta_ext_init * pow(bcl_mu_in_inv, alpha);
			bcl_eta_in = max2(  bcl_mu_in_inv  ,eps_abs);
	}
}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
void oldNew_global_primal_residual(
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
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  residual_scaled_tmp,
			qp::QpViewBoxMut<T> qp_scaled,
			Preconditioner& precond,
			VectorViewMut<T> x,
			isize dim,
			isize n_eq,
			isize n_in
		){
				LDLT_DECL_SCOPE_TIMER("in solver", "primal residual", T);
				auto A_ = qp_scaled.A.as_const().to_eigen();
				auto C_ = qp_scaled.C.as_const().to_eigen();
				auto x_ = x.as_const().to_eigen();
				auto b_ = qp_scaled.b.as_const().to_eigen();
				auto l_ = qp_scaled.l.as_const().to_eigen();
				auto u_ = qp_scaled.u.as_const().to_eigen();

				residual_scaled_tmp.setZero();
				// A×x - b and Cx - u and Cx - l  /!\ optimization surely possible
				primal_residual_eq_scaled.setZero();
				primal_residual_eq_scaled.noalias() += A_ * x_;
    
				primal_residual_in_scaled_u.setZero();
				primal_residual_in_scaled_u.noalias() += C_ * x_;
				primal_residual_in_scaled_l.setZero();
				primal_residual_in_scaled_l.noalias() += C_ * x_;

				{
					residual_scaled_tmp.middleRows(dim,n_eq) = primal_residual_eq_scaled;
                    residual_scaled_tmp.bottomRows(n_in) = primal_residual_in_scaled_u;
					precond.unscale_primal_residual_in_place_eq(
							VectorViewMut<T>{from_eigen, residual_scaled_tmp.middleRows(dim,n_eq)});
                    precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, residual_scaled_tmp.bottomRows(n_in)});
					primal_feasibility_eq_rhs_0 = infty_norm(residual_scaled_tmp.middleRows(dim,n_eq));
                    primal_feasibility_in_rhs_0 = infty_norm(residual_scaled_tmp.bottomRows(n_in));
				}
				primal_residual_eq_scaled -= b_;
                primal_residual_in_scaled_u -= u_ ;
                primal_residual_in_scaled_l -= l_ ;

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
			T& dual_feasibility_lhs,
			T& dual_feasibility_rhs_0,
			T& dual_feasibility_rhs_1,
        	T& dual_feasibility_rhs_3,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  dual_residual_scaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  dual_residual_unscaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  residual_scaled_tmp,
			qp::QpViewBoxMut<T> qp_scaled,
			Preconditioner& precond,
			VectorViewMut<T> x,
			VectorViewMut<T> y,
			VectorViewMut<T> z,
			isize dim,
			isize n_eq,
			isize n_in
		){
			auto H_ = qp_scaled.H.as_const().to_eigen();
			auto A_ = qp_scaled.A.as_const().to_eigen();
            auto C_ = qp_scaled.C.as_const().to_eigen();
			auto x_ = x.as_const().to_eigen();
			auto y_ = y.as_const().to_eigen();
            auto z_ = z.as_const().to_eigen();
			auto g_ = qp_scaled.g.as_const().to_eigen();
			residual_scaled_tmp.setZero();

			dual_residual_scaled = g_;
			{
				residual_scaled_tmp.topRows(dim).noalias() = (H_ * x_);
				{ dual_residual_scaled += residual_scaled_tmp.topRows(dim); }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, residual_scaled_tmp.topRows(dim)});
				dual_feasibility_rhs_0 = infty_norm(residual_scaled_tmp.topRows(dim));

				residual_scaled_tmp.topRows(dim).noalias() = A_.transpose() * y_;
				{ dual_residual_scaled += residual_scaled_tmp.topRows(dim); }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, residual_scaled_tmp.topRows(dim)});
				dual_feasibility_rhs_1 = infty_norm(residual_scaled_tmp.topRows(dim));

				residual_scaled_tmp.topRows(dim).noalias() = C_.transpose() * z_;
				{ dual_residual_scaled += residual_scaled_tmp.topRows(dim); }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, residual_scaled_tmp.topRows(dim)});
				dual_feasibility_rhs_3 = infty_norm(residual_scaled_tmp.topRows(dim));
			}

			dual_residual_unscaled = dual_residual_scaled;
			precond.unscale_dual_residual_in_place(
					VectorViewMut<T>{from_eigen, dual_residual_unscaled});

			dual_feasibility_lhs = infty_norm(dual_residual_unscaled);
        };


template<typename T> 
T oldNew_SaddlePoint(
			qp::QpViewBox<T> qp_scaled,
			VectorViewMut<T> x,
			VectorViewMut<T> y,
        	VectorViewMut<T> z,
			VectorViewMut<T> xe,
			VectorViewMut<T> ye,
        	VectorViewMut<T> ze,
			T mu_in_inv,
			T rho,
			isize n_in,
			VectorViewMut<T> prim_in_u,
			VectorViewMut<T> prim_in_l,
			VectorViewMut<T> prim_eq,
			VectorViewMut<T> dual_eq
			){
			
			auto H_ = qp_scaled.H.to_eigen();
			auto g_ = qp_scaled.g.to_eigen();
			auto A_ = qp_scaled.A.to_eigen();
			auto C_ = qp_scaled.C.to_eigen();
			auto x_ = x.as_const().to_eigen();
			auto x_e = xe.to_eigen();
			auto y_ = y.as_const().to_eigen();
			auto y_e = ye.to_eigen();
			auto z_ = z.as_const().to_eigen();
			auto z_e = ze.to_eigen();
			auto b_ = qp_scaled.b.to_eigen();
			auto l_ = qp_scaled.l.to_eigen();
			auto u_ = qp_scaled.u.to_eigen();

			prim_in_u.to_eigen().noalias()-=  (z_*mu_in_inv); 
			prim_in_l.to_eigen().noalias() -= (z_*mu_in_inv) ; 
			T prim_eq_e = infty_norm(prim_eq.to_eigen()) ; 
			dual_eq.to_eigen().noalias() += (C_.transpose()*z_);
			T dual_e = infty_norm(dual_eq.to_eigen());
			T err = max2(prim_eq_e,dual_e);

			T prim_in_e(0);

			for (isize i = 0 ; i< n_in ; i=i+1){
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
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		T mu_eq,
		T mu_in,
		T mu_eq_inv,
		T mu_in_inv,
		T rho,
		T eps,
		isize dim,
		isize n_eq,
		isize n_in,
		Eigen::Matrix<T,Eigen::Dynamic,1>& z_pos,
		Eigen::Matrix<T,Eigen::Dynamic,1>& z_neg,
		Eigen::Matrix<T,Eigen::Dynamic,1>& res_y, 
		Eigen::Matrix<T,Eigen::Dynamic,1>& dual_for_eq,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& l_active_set_n_u,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& l_active_set_n_l,
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& active_inequalities,

        Eigen::Matrix<T,Eigen::Dynamic,1>& dw_aug,
        isize& n_c,
        Eigen::Matrix<isize, Eigen::Dynamic, 1>& current_bijection_map,
		Eigen::Matrix<isize, Eigen::Dynamic, 1>& new_bijection_map,
        ldlt::Ldlt<T>& ldl,
        const bool VERBOSE,
        isize nb_iterative_refinement,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
		VectorViewMut<T> err_,
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& kkt,
		T eps_refact,
		std::string str,
		const bool chekNoAlias
	){
		
		auto H_ = qp_scaled.H.to_eigen();
		auto A_ = qp_scaled.A.to_eigen();
		auto C_ = qp_scaled.C.to_eigen();
		
		l_active_set_n_u.noalias() = (z_pos.array() > 0).matrix();
		l_active_set_n_l.noalias() = (z_neg.array() < 0).matrix();

		active_inequalities.noalias() = l_active_set_n_u || l_active_set_n_l ; 

		isize num_active_inequalities = active_inequalities.count();
		isize inner_pb_dim = dim + n_eq + num_active_inequalities;

		rhs.setZero();
		dw_aug.setZero();
		
		{
				rhs.topRows(dim).noalias() -=  dual_for_eq ;
				for (isize j = 0 ; j < n_in; ++j){
					rhs.topRows(dim).noalias() -= mu_in*(max2(z_pos(j),T(0.)) + std::min(z_neg(j),T(0.))) * C_.row(j) ; 
				}

		}
        qp::line_search::oldNew_active_set_change(
                    qp_scaled,
                    VectorView<bool>{from_eigen,active_inequalities},
                    VectorViewMut<isize>{from_eigen,current_bijection_map},
					new_bijection_map,
                    ldl,
                    dw_aug,
					rho,
                    mu_in_inv,
                    n_c,
                    n_in,
                    n_eq,
                    dim);

		auto err = err_.to_eigen();
        oldNew_iterative_solve_with_permut_fact_new( //
                    qp_scaled,
                    rho,
                    mu_eq,
                    mu_in,
					mu_eq_inv,
					mu_in_inv,
                    VectorViewMut<T>{from_eigen,rhs.head(inner_pb_dim)},
                    VectorViewMut<T>{from_eigen,dw_aug},
                    VectorViewMut<isize>{from_eigen,current_bijection_map},
                    eps,
                    inner_pb_dim,
                    dim,
                    n_eq,
                    n_in,
                    n_c,
                    ldl,
                    VERBOSE,
                    nb_iterative_refinement,
					VectorViewMut<T>{from_eigen,err.head(inner_pb_dim)},
					kkt,
					eps_refact,
					isize(1),
					str,
					chekNoAlias
					);

}

template<typename T,typename Preconditioner>
T oldNew_initial_guess(
		VectorViewMut<T> xe,
		VectorViewMut<T> ye,
        VectorViewMut<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
        VectorViewMut<T> z,
		qp::QpViewBoxMut<T> qp_scaled,
		T mu_in,
		T mu_eq,
		T mu_in_inv,
		T mu_eq_inv,
		T rho,
		T eps_int,
		Preconditioner precond,
		isize dim,
		isize n_eq,
		isize n_in,
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

        Eigen::Matrix<isize, Eigen::Dynamic, 1>& current_bijection_map,
        ldlt::Ldlt<T>& ldl,
        isize& n_c,
        const bool VERBOSE,
        isize nb_iterative_refinement,
		T R,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& active_part_z,
		Eigen::Matrix<isize, Eigen::Dynamic, 1>& new_bijection_map,

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
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& kkt,
		T eps_refact,

		std::vector<T>& alphas,

		std::string str,

		const bool chekNoAlias

		){
			
			auto H_ = qp_scaled.H.as_const().to_eigen();
			auto g_ = qp_scaled.g.as_const().to_eigen();
			auto A_ = qp_scaled.A.as_const().to_eigen();
			auto C_ = qp_scaled.C.as_const().to_eigen();
			auto x_ = x.to_eigen();
			auto z_ = z.to_eigen();
			auto z_e = ze.to_eigen().eval();
			auto l_ = qp_scaled.l.as_const().to_eigen();
			auto u_ = qp_scaled.u.as_const().to_eigen();

			prim_in_u.noalias() =  (C_*x_-u_) ; 
			prim_in_l.noalias() = (C_*x_-l_) ; 

			precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_u});
			precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_l});
			precond.unscale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, z_e}); 
			
			prim_in_u.noalias() += (z_e*mu_in_inv) ; 

			prim_in_l.noalias() += (z_e*mu_in_inv) ; 

			l_active_set_n_u = (prim_in_u.array() >= 0.).matrix();
			l_active_set_n_l = (prim_in_l.array() <= 0.).matrix();

			active_inequalities = l_active_set_n_u || l_active_set_n_l ;   

			prim_in_u.noalias() -= (z_e*mu_in_inv) ; 
			prim_in_l.noalias() -= (z_e*mu_in_inv) ; 

			precond.scale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_u});
			precond.scale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_l});
			precond.scale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, z_e});

			isize num_active_inequalities = active_inequalities.count();
			isize inner_pb_dim = dim + n_eq + num_active_inequalities;

			rhs.setZero();
			active_part_z.setZero();
            qp::line_search::oldNew_active_set_change(
                                qp_scaled.as_const(),
                                VectorView<bool>{from_eigen,active_inequalities},
                                VectorViewMut<isize>{from_eigen,current_bijection_map},
								new_bijection_map,
                                ldl,
                                dw_aug,
								rho,
								mu_in_inv,
                                n_c,
                                n_in,
                                n_eq,
                                dim);
			
			rhs.head(dim).noalias() = -dual_for_eq ;
			rhs.segment(dim,n_eq).noalias() = -primal_residual_eq ;
			for (isize i = 0; i < n_in; i++) {
				isize j = current_bijection_map(i);
				if (j < n_c) {
					if (l_active_set_n_u(i)) {
						rhs(j + dim + n_eq) = -prim_in_u(i);
					} else if (l_active_set_n_l(i)) {
						rhs(j + dim + n_eq) = -prim_in_l(i);
					}
				} else {
					rhs.head(dim).noalias() += z_(i) * C_.row(i); // unactive unrelevant columns
				}
			}	

			auto err_iter = _err.to_eigen();
            oldNew_iterative_solve_with_permut_fact_new( //
                    qp_scaled.as_const(),
                    rho,
                    mu_eq,
                    mu_in,
					mu_eq_inv,
					mu_in_inv,
                    VectorViewMut<T>{from_eigen,rhs.head(inner_pb_dim)},
                    VectorViewMut<T>{from_eigen,dw_aug},
                    VectorViewMut<isize>{from_eigen,current_bijection_map},
                    eps_int,
                    inner_pb_dim,
                    dim,
                    n_eq,
                    n_in,
                    n_c,
                    ldl,
                    VERBOSE,
                    nb_iterative_refinement,
					VectorViewMut<T>{from_eigen,err_iter.head(inner_pb_dim)},
					kkt,
					eps_refact,
					isize(2),
					str,
					chekNoAlias
					);
			
			// use active_part_z as a temporary variable to permut back dw_aug newton step
			for (isize j = 0; j < n_in; ++j) {
				isize i = current_bijection_map(j);
				if (i < n_c) {
					//dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
					
					active_part_z(j) = dw_aug(dim + n_eq + i);
					
				} else {
					//dw_aug_(j + dim + n_eq) = -z_(j);
					active_part_z(j) = -z_(j);
				}
			}
			dw_aug.tail(n_in) = active_part_z ;

			prim_in_u.noalias() += (z_e*mu_in_inv) ; 
			prim_in_l.noalias() += (z_e*mu_in_inv) ; 

			d_primal_residual_eq.noalias() = (A_*dw_aug.topRows(dim)- dw_aug.middleRows(dim,n_eq) * mu_eq_inv).eval() ;
			d_dual_for_eq.noalias() = (H_*dw_aug.topRows(dim)+A_.transpose()*dw_aug.middleRows(dim,n_eq)+rho*dw_aug.topRows(dim)).eval() ;
			Cdx_.noalias() = C_*dw_aug.topRows(dim) ;
			dual_for_eq.noalias() -= C_.transpose()*z_e ; 
			T alpha_step = qp::line_search::oldNew_initial_guess_LS(
									  ze.as_const(),
						VectorView<T>{from_eigen,dw_aug.tail(n_in)},
						VectorView<T>{from_eigen,prim_in_l},
						VectorView<T>{from_eigen,prim_in_u},
						VectorView<T>{from_eigen,Cdx_},
						VectorView<T>{from_eigen,d_dual_for_eq},
						VectorView<T>{from_eigen,dual_for_eq},
						VectorView<T>{from_eigen,d_primal_residual_eq},
						VectorView<T>{from_eigen,primal_residual_eq},
						qp_scaled.C.as_const(),
						mu_in_inv,
						rho,
						dim,
						n_eq,
						n_in,
						R,

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

			x.to_eigen().noalias() += (alpha_step * dw_aug.topRows(dim)) ; 
			y.to_eigen().noalias() += (alpha_step * dw_aug.middleRows(dim,n_eq)) ; 

			for (isize i = 0; i< n_in ; ++i){
				if (l_active_set_n_u(i)){
					z_(i) = std::max(z_(i)+alpha_step*dw_aug(dim+n_eq+i),T(0.)) ; 
				}else if (l_active_set_n_l(i)){
					z_(i) = std::min(z_(i)+alpha_step*dw_aug(dim+n_eq+i),T(0.)) ; 
				} else{
					z_(i) += alpha_step*dw_aug(dim+n_eq+i) ; 
				}
			}
			primal_residual_eq.noalias() += (alpha_step*d_primal_residual_eq);
			dual_for_eq.noalias() += alpha_step* (d_dual_for_eq) ;
			dw_aug.setZero();
			T err_saddle_point = oldNew_SaddlePoint(
				qp_scaled.as_const(),
				x,
				y,
				z,
				xe,
				ye,
				ze,
				mu_in_inv,
				rho,
				n_in,
				VectorViewMut<T>{from_eigen,prim_in_u},
				VectorViewMut<T>{from_eigen,prim_in_l},
				VectorViewMut<T>{from_eigen,primal_residual_eq},
				VectorViewMut<T>{from_eigen,dual_for_eq}
			);
			
			return err_saddle_point;
}


template<typename T>
T oldNew_correction_guess(
		VectorView<T> xe,
		VectorView<T> ye, 
        VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
        VectorViewMut<T> z,
		qp::QpViewBoxMut<T> qp_scaled,
		T mu_in,
		T mu_eq,
		T mu_in_inv,
		T mu_eq_inv,

		T rho,
		T eps_int,
		isize dim,
		isize n_eq,
		isize n_in,
		isize max_iter_in,
		isize& n_tot,
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
        Eigen::Matrix<isize, Eigen::Dynamic, 1>& current_bijection_map,
		Eigen::Matrix<isize, Eigen::Dynamic, 1>& new_bijection_map,
        ldlt::Ldlt<T>& ldl,
        isize& n_c,
        const bool VERBOSE,
        isize nb_iterative_refinement,
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
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& kkt,
		T eps_refact,

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

		for (i64 iter = 0; iter <= max_iter_in; ++iter) {

			if (iter == max_iter_in) {
				n_tot +=max_iter_in;
				break;
			}
			
			qp::detail::oldNew_newton_step_new<T>(
											qp_scaled.as_const(),
											x.as_const(),
											xe,
											ye,
											ze,
											mu_eq,
											mu_in,
											mu_eq_inv,
											mu_in_inv,
											rho,
											eps_int,
											dim,
											n_eq,
											n_in,
											z_pos,
											z_neg,
											residual_in_y,
											dual_for_eq,
											l_active_set_n_u,
											l_active_set_n_l,
											active_inequalities,

                                            dw_aug,
                                            n_c,
                                            current_bijection_map,
											new_bijection_map,
                                            ldl,
                                            VERBOSE,
                                            nb_iterative_refinement,
											rhs,
											err_,
											kkt,
											eps_refact,
											str,
											checkNoAlias
			);
			T alpha_step = 1;
			Hdx.noalias() = (qp_scaled.H).to_eigen() * dw_aug.head(dim) ; 
			Adx.noalias() = (qp_scaled.A).to_eigen() * dw_aug.head(dim) ; 
			Cdx.noalias() = (qp_scaled.C).to_eigen() * dw_aug.head(dim) ; 
			if (n_in > 0){
				alpha_step = qp::line_search::oldNew_correction_guess_LS(
										Hdx,
									 	VectorView<T>{from_eigen,dw_aug.head(dim)},
										(qp_scaled.g).as_const(),
										Adx,  
										Cdx,
										residual_in_y,
										z_pos,
										z_neg,
										x.as_const(),
										xe,
										ye,
										ze,
										mu_eq,
										mu_in,

										rho,
										n_in,

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
			if (infty_norm(alpha_step * dw_aug.head(dim))< 1.E-11){
				n_tot += iter+1;
				std::cout << "infty_norm(alpha_step * dx) " << infty_norm(alpha_step * dw_aug.head(dim)) << std::endl;
				break;
			}
			
			x.to_eigen().noalias()+= (alpha_step *dw_aug.head(dim)) ; 
			z_pos.noalias() += (alpha_step *Cdx) ;
			z_neg.noalias() += (alpha_step *Cdx); 
			residual_in_y.noalias() += (alpha_step * Adx);
 			y.to_eigen().noalias() = mu_eq *  residual_in_y  ;
			dual_for_eq.noalias() += (alpha_step * ( mu_eq * (qp_scaled.A).to_eigen().transpose() * Adx + rho * dw_aug.head(dim) + Hdx  )) ;
			for (isize j = 0 ; j < n_in; ++j){
				z(j) = mu_in*(max2(z_pos(j),T(0)) + std::min(z_neg(j),T(0))); 
			}

			tmp1.noalias() = (qp_scaled.H).to_eigen() *x.to_eigen() ;
			tmp2.noalias() = (qp_scaled.A.to_eigen().transpose()) * ( y.to_eigen() );
			tmp3.noalias() = (qp_scaled.C.to_eigen().transpose()) * ( z.to_eigen() )   ; 
			grad_n.noalias() = tmp1 + tmp2 + tmp3  + (qp_scaled.g).to_eigen() + rho* (x.to_eigen()-xe.to_eigen()) ;

			err_in = infty_norm(  grad_n  );
			std::cout << "---it in " << iter << " projection norm " << err_in << " alpha " << alpha_step << " rhs " << eps_int * (1 + max2(max2(max2(infty_norm(tmp1), infty_norm(tmp2)), infty_norm(tmp3)), infty_norm((qp_scaled.g).to_eigen())) )   <<  std::endl;

			if (err_in<= eps_int * (1. + max2(max2(max2(infty_norm(tmp1), infty_norm(tmp2)), infty_norm(tmp3)), infty_norm((qp_scaled.g).to_eigen())) )  ){
				n_tot +=iter+1;
				break;
			}
		}
	
		return err_in;

}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpSolveStats oldNew_qpSolve( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
        VectorViewMut<T> z,
		qp::QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		std::string str,
		Preconditioner precond = Preconditioner{},
		T mu_max_eq = 1.e8,
		T mu_max_in = 1.e8,
		T R = 5,
		T eps_IG = 1.e-4,
		T eps_refact = 1.e-6,
		isize nb_iterative_refinement = 5,
		const bool chekNoAlias = false) {

	using namespace ldlt::tags;
    static constexpr Layout layout = rowmajor;
    static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
    const bool VERBOSE = true;
	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
    isize n_in = qp.C.rows;
	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;
    isize n_c = 0;
	T alpha(0.1); 
	T beta(0.9);
	T mu_max_eq_inv(1./mu_max_eq);
	T mu_max_in_inv(1./mu_max_in);

	T machine_eps = std::numeric_limits<T>::epsilon();
	T rho(1e-6);
	T bcl_mu_eq(1e3);
    T bcl_mu_in(1e1);
	T bcl_mu_eq_inv(1.e-3);
	T bcl_mu_in_inv(1.e-1);

	T bcl_eta_ext_init = pow(T(0.1),alpha);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	
	LDLT_MULTI_WORKSPACE_MEMORY(
	     	(_g_scaled,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
	     	(_b_scaled,Init, Vec(n_eq),LDLT_CACHELINE_BYTES, T),
         	(_u_scaled,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
         	(_l_scaled,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
	     	(_residual_scaled,Init, Vec(dim + n_eq+n_in),LDLT_CACHELINE_BYTES, T),
	     	(_residual_scaled_tmp,Init, Vec(dim + n_eq+n_in),LDLT_CACHELINE_BYTES, T),
	     	(_residual_unscaled,Init, Vec(dim + n_eq+n_in),LDLT_CACHELINE_BYTES, T),
	     	(_y,Init, Vec(n_eq),LDLT_CACHELINE_BYTES, T), 
         	(_z,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(xe_,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
			
			(_dw_aug,Init, Vec(dim+n_eq+n_in),LDLT_CACHELINE_BYTES, T),
			(_rhs,Init, Vec(dim+n_eq+n_in),LDLT_CACHELINE_BYTES, T),
			(_active_part_z,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(d_dual_for_eq_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(Cdx__,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(d_primal_residual_eq_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(l_active_set_n_u_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, bool),
			(l_active_set_n_l_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, bool),
			(active_inequalities_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, bool),
            (_current_bijection_map,Init, Vec(n_in),LDLT_CACHELINE_BYTES, isize),
			(new_bijection_map_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, isize)
		);
	
	LDLT_MULTI_WORKSPACE_MEMORY(
			(_tmp_u,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp_l,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_aux_u,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
			(_aux_l,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
			(_dz_p,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp1,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
			(_tmp2,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
			(_tmp3,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
			(_tmp4,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),

			(_tmp_d2_u,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp_d2_l,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp_d3,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp2_u,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp2_l,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_tmp3_local_saddle_point,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),

			(_err,Init, Vec(dim+n_eq+n_in),LDLT_CACHELINE_BYTES, T),

			(_active_set_l,Init, Vec(n_in),LDLT_CACHELINE_BYTES, isize),
			(_active_set_u,Init, Vec(n_in),LDLT_CACHELINE_BYTES, isize),
			(_inactive_set,Init, Vec(n_in),LDLT_CACHELINE_BYTES, isize)
			


	);

	RowMat test(2,2); // test it is full of nan
	std::cout << "test " << test << std::endl;

	std::vector<T> alphas;
	alphas.reserve( 3*n_in );

	auto err = _err.to_eigen();
	auto tmp_u = _tmp_u.to_eigen().eval();
	auto tmp_l = _tmp_l.to_eigen().eval();
	auto aux_u = _aux_u.to_eigen().eval();
	auto aux_l = _aux_l.to_eigen().eval();
	auto dz_p = _dz_p.to_eigen().eval();

	auto rhs = _rhs.to_eigen().eval();
	auto active_part_z =_active_part_z.to_eigen().eval();
    auto current_bijection_map = _current_bijection_map.to_eigen().eval();

	auto new_bijection_map = new_bijection_map_.to_eigen().eval();
    for (isize i = 0; i < n_in; i++) {
        current_bijection_map(i) = i;
		new_bijection_map(i) = i;
    } 
    

    RowMat H_copy(dim,dim);
    RowMat A_copy(n_eq,dim);
    RowMat C_copy(n_in,dim);
    RowMat kkt(dim+n_eq,dim+n_eq);
    H_copy.setZero();
    A_copy.setZero();
    C_copy.setZero();

	auto d_dual_for_eq = d_dual_for_eq_.to_eigen().eval();
	auto Cdx_ = Cdx__.to_eigen().eval();
	auto d_primal_residual_eq = d_primal_residual_eq_.to_eigen().eval();
	auto l_active_set_n_u = l_active_set_n_u_.to_eigen().eval();
	auto l_active_set_n_l = l_active_set_n_l_.to_eigen().eval();
	auto active_inequalities = active_inequalities_.to_eigen().eval();	
	auto dw_aug = _dw_aug.to_eigen().eval(); 

	auto q_copy = _g_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();
    auto u_copy = _u_scaled.to_eigen();
    auto l_copy = _l_scaled.to_eigen();

	H_copy = qp.H.to_eigen();
	q_copy = qp.g.to_eigen();
	A_copy = qp.A.to_eigen();
	b_copy = qp.b.to_eigen();
    C_copy = qp.C.to_eigen();
    u_copy = qp.u.to_eigen();
    l_copy = qp.l.to_eigen();

    qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen,H_copy},
			_g_scaled,
			{from_eigen,A_copy},
			_b_scaled,
			{from_eigen,C_copy},
			_u_scaled,
            _l_scaled
	};
    
	precond.scale_qp_in_place(qp_scaled,_dw_aug);
    dw_aug.setZero();
	////

	kkt.topLeftCorner(dim, dim) = H_copy ;
	kkt.topLeftCorner(dim, dim).diagonal().array() += rho;	
	kkt.block(0, dim, dim, n_eq) = A_copy.transpose();
	kkt.block(dim, 0, n_eq, dim) = A_copy;
	kkt.bottomRightCorner(n_eq, n_eq).setZero();
	kkt.diagonal().segment(dim, n_eq).setConstant(-bcl_mu_eq_inv); // mu stores the inverse of mu

	ldlt::Ldlt<T> ldl{decompose, kkt};
	rhs.head(dim) = -qp_scaled.g.to_eigen();
	rhs.segment(dim,n_eq) = qp_scaled.b.to_eigen();
	
    oldNew_iterative_solve_with_permut_fact_new( //
        qp_scaled.as_const(),
		rho,
        bcl_mu_eq,
        bcl_mu_in,
        bcl_mu_eq_inv,
        bcl_mu_in_inv,
		VectorViewMut<T>{from_eigen,rhs.head(dim+n_eq)},
        VectorViewMut<T>{from_eigen,dw_aug},
        VectorViewMut<isize>{from_eigen,current_bijection_map},
		bcl_eta_in,
		dim+n_eq,
        dim,
        n_eq,
        n_in,
        n_c,
        ldl,
        VERBOSE,
        nb_iterative_refinement,
		VectorViewMut<T>{from_eigen,err.head(dim+n_eq)},
		kkt,
		eps_refact,
		isize(0),
		str,
		chekNoAlias
        );
	x.to_eigen() = dw_aug.head(dim);
	y.to_eigen() = dw_aug.segment(dim,n_eq);
	
	dw_aug.setZero();

	auto residual_scaled = _residual_scaled.to_eigen().eval();
	auto residual_scaled_tmp = _residual_scaled_tmp.to_eigen().eval();
	auto residual_unscaled = _residual_unscaled.to_eigen().eval();

	auto ye = _y.to_eigen();
    auto ze = _z.to_eigen();
	auto xe = xe_.to_eigen();

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
    T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
    T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());
	
	auto dual_residual_scaled = residual_scaled.topRows(dim).eval();
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim,n_eq).eval();
	auto primal_residual_in_scaled_u = residual_scaled.bottomRows(n_in).eval();
	auto primal_residual_in_scaled_l = residual_scaled.bottomRows(n_in).eval();
	auto dual_residual_unscaled = residual_unscaled.topRows(dim).eval();
	auto primal_residual_eq_unscaled = residual_unscaled.middleRows(dim,n_eq).eval();
	auto primal_residual_in_u_unscaled = residual_unscaled.bottomRows(n_in).eval();
	auto primal_residual_in_l_unscaled = residual_unscaled.bottomRows(n_in).eval();

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
	
	for (i64 iter = 0; iter <= max_iter; ++iter) {
		n_ext +=1;
		if (iter == max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::oldNew_global_primal_residual(
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
				residual_scaled_tmp,
				qp_scaled,
				precond,
				x,
				dim,
				n_eq,
				n_in
		);
		qp::detail::oldNew_global_dual_residual(
			dual_feasibility_lhs,
			dual_feasibility_rhs_0,
			dual_feasibility_rhs_1,
        	dual_feasibility_rhs_3,
			dual_residual_scaled,
			dual_residual_unscaled,
			residual_scaled_tmp,
			qp_scaled,
			precond,
			x,
			y,
			z,
			dim,
			n_eq,
			n_in
		);
		
		std::cout << "---------------it : " << iter << " primal residual : " << primal_feasibility_lhs << " dual residual : " << dual_feasibility_lhs << std::endl;
		std::cout << "bcl_eta_ext : " << bcl_eta_ext << " bcl_eta_in : " << bcl_eta_in <<  " rho : " << rho << " bcl_mu_eq : " << bcl_mu_eq << " bcl_mu_in : " << bcl_mu_in <<std::endl;

		bool is_primal_feasible =
				primal_feasibility_lhs <=
				(eps_abs + eps_rel * max2(
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
				(eps_abs + eps_rel * max2(                      
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
			
			if (dual_feasibility_lhs >= 1.e-2 && rho != 1.e-7){

				T rho_new(1.e-7);
				oldNew_refactorize(
						qp_scaled.as_const(),
						rho_new,
						rho,
						ldl,
						kkt,
						bcl_mu_eq,
						bcl_mu_in,
						bcl_mu_eq_inv,
						bcl_mu_in_inv,
						dim,
						n_eq,
						n_in,
						n_c,
						VectorViewMut<isize>{from_eigen,current_bijection_map},
						VectorViewMut<T>{from_eigen,dw_aug}
						);

				rho = rho_new;
			}
			if (is_dual_feasible){
				{
				LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
				precond.unscale_primal_in_place(x); 
				precond.unscale_dual_in_place_eq(y);
				precond.unscale_dual_in_place_in(z);
				}
				return {n_ext, n_mu_updates,n_tot};
			}
		}
		
		xe = x.to_eigen().eval(); 
		ye = y.to_eigen().eval(); 
		ze = z.to_eigen().eval(); 

		const bool do_initial_guess_fact = (primal_feasibility_lhs < eps_IG || n_in == 0 ) ;

		T err_in(0.);

		if (do_initial_guess_fact){

			err_in = qp::detail::oldNew_initial_guess<T,Preconditioner>(
							VectorViewMut<T>{from_eigen,xe},
							VectorViewMut<T>{from_eigen,ye},
							VectorViewMut<T>{from_eigen,ze},
							x,
							y,
							z,
							qp_scaled,
							bcl_mu_in,
							bcl_mu_eq,
							bcl_mu_in_inv,
							bcl_mu_eq_inv,

							rho,
							bcl_eta_in,
							precond,
							dim,
							n_eq,
							n_in,
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

							current_bijection_map,
							ldl,
							n_c,
							VERBOSE,
							nb_iterative_refinement,
							R,
							rhs,
							active_part_z,
							new_bijection_map,

							tmp_u,
							tmp_l,
							aux_u,
							aux_l,
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

							_err,
							kkt,
							eps_refact,

							alphas,
							
							str,
							chekNoAlias

			);
			n_tot +=1;

		}

		bool do_correction_guess = (!do_initial_guess_fact && n_in != 0) ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in && n_in != 0) ;
		
		std::cout << " error from initial guess : " << err_in << " bcl_eta_in " << bcl_eta_in << std::endl;
		
		if ((do_initial_guess_fact && err_in >= bcl_eta_in && n_in != 0)){

			dual_residual_scaled.noalias() += (-(qp_scaled.C).to_eigen().transpose()*z.to_eigen() + bcl_mu_eq * (qp_scaled.A).to_eigen().transpose()*primal_residual_eq_scaled );
			primal_residual_eq_scaled.noalias()  += (y.to_eigen()*bcl_mu_eq_inv);
			primal_residual_in_scaled_u.noalias()  += (z.to_eigen()*bcl_mu_in_inv);
			primal_residual_in_scaled_l.noalias() += (z.to_eigen()*bcl_mu_in_inv); //

		}
		if (!do_initial_guess_fact && n_in != 0){ // y=ye, x=xe, 

			primal_residual_eq_scaled.noalias()  += (ye*bcl_mu_eq_inv); // Axe-b+ye/mu_eq
			primal_residual_eq_scaled.noalias()  -= (y.to_eigen()*bcl_mu_eq_inv);

			primal_residual_in_scaled_u.noalias()  = (qp_scaled.C).to_eigen() * x.to_eigen() + ze*bcl_mu_in_inv;
			primal_residual_in_scaled_l.noalias()  = primal_residual_in_scaled_u ;
			primal_residual_in_scaled_u.noalias()  -= qp_scaled.u.to_eigen();
			primal_residual_in_scaled_l.noalias()  -= qp_scaled.l.to_eigen();

			dual_residual_scaled.noalias() += -(qp_scaled.C).to_eigen().transpose()*z.to_eigen();
			dual_residual_scaled.noalias() += bcl_mu_eq * (qp_scaled.A).to_eigen().transpose()*primal_residual_eq_scaled ; // no need of rho * (x-xe) as x=xe
			primal_residual_eq_scaled.noalias()  += (y.to_eigen()*bcl_mu_eq_inv);

		}

		if (do_correction_guess){
			
			err_in = qp::detail::oldNew_correction_guess(
						VectorView<T>{from_eigen,xe},
						VectorView<T>{from_eigen,ye},
						VectorView<T>{from_eigen,ze},
						x,
						y,
						z,
						qp_scaled,
						bcl_mu_in,
						bcl_mu_eq,
						bcl_mu_in_inv,
						bcl_mu_eq_inv,

						rho,
						bcl_eta_in,
						dim,
						n_eq,
						n_in,
						max_iter_in,
						n_tot,
						primal_residual_eq_scaled,
						primal_residual_in_scaled_u,
						primal_residual_in_scaled_l,
						dual_residual_scaled,
						d_primal_residual_eq,
						Cdx_,
						d_dual_for_eq,
						l_active_set_n_u,
						l_active_set_n_l,
						active_inequalities,


                        dw_aug,
                        current_bijection_map,
						new_bijection_map,
                        ldl,
                        n_c,
                        VERBOSE,
                        nb_iterative_refinement,
						rhs,

						_tmp1,
						_tmp2,
						_tmp3,
						_tmp4,

						VectorViewMut<T>{from_eigen,tmp_u},
						VectorViewMut<T>{from_eigen,tmp_l},
						_active_set_u,
						_active_set_l,

						_tmp_d2_u,
						_tmp2_u,
						_tmp_d2_l,
						_tmp2_l,

						_err,
						kkt,
						eps_refact,

						alphas,

						aux_u,

						str,
						chekNoAlias


			);
			std::cout << " error from correction guess : " << err_in << std::endl;
		}
		
		T primal_feasibility_lhs_new(primal_feasibility_lhs) ; 

		qp::detail::oldNew_global_primal_residual(
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
						residual_scaled_tmp,
						qp_scaled,
						precond,
						x,
						dim,
						n_eq,
						n_in
		);

		qp::detail::oldNew_BCL_update(
					primal_feasibility_lhs_new,
					VectorViewMut<T>{from_eigen,primal_residual_in_scaled_u},
					VectorViewMut<T>{from_eigen,primal_residual_in_scaled_l},
					VectorViewMut<T>{from_eigen,primal_residual_eq_scaled},
					precond,
					bcl_eta_ext,
					bcl_eta_in,
					eps_abs,
					n_mu_updates,
					bcl_mu_in,
					bcl_mu_eq,
					bcl_mu_in_inv,
					bcl_mu_eq_inv,

					VectorViewMut<T>{from_eigen,ye},
					VectorViewMut<T>{from_eigen,ze},
					y,
					z,
					VectorViewMut<T>{from_eigen,dw_aug},
					ldl,
					dim,
					n_eq,
					n_in,
					n_c,
					mu_max_eq,
					mu_max_in,
					mu_max_eq_inv,
					mu_max_in_inv,
					mu_fact_update,
					mu_fact_update_inv,
					bcl_eta_ext_init,
					beta,
					alpha
		);

		// COLD RESTART
		
		T dual_feasibility_lhs_new(dual_feasibility_lhs) ; 
		
		qp::detail::oldNew_global_dual_residual(
			dual_feasibility_lhs_new,
			dual_feasibility_rhs_0,
			dual_feasibility_rhs_1,
        	dual_feasibility_rhs_3,
			dual_residual_scaled,
			dual_residual_unscaled,
			residual_scaled_tmp,
			qp_scaled,
			precond,
			x,
			y,
			z,
			dim,
			n_eq,
			n_in
		);

		if ((primal_feasibility_lhs_new / max2(primal_feasibility_lhs,machine_eps) >= 1.) && (dual_feasibility_lhs_new / max2(primal_feasibility_lhs,machine_eps) >= 1.) && bcl_mu_in >= 1.E5){
			std::cout << "cold restart" << std::endl;
			T bcl_mu_in_new(1.1); 
			T bcl_mu_eq_new(1.1);
			T bcl_mu_in_inv_new(1./bcl_mu_in_new); 
			T bcl_mu_eq_inv_new(1./bcl_mu_eq_new);



			qp::detail::oldNew_mu_update(
				ldl,
				_dw_aug,
				bcl_mu_eq_inv,
				bcl_mu_in_inv,
				bcl_mu_eq_inv_new,
				bcl_mu_eq_inv_new,
				dim,
				n_eq,
				n_in,
				n_c);
			
			bcl_mu_in = bcl_mu_in_new;
			bcl_mu_eq = bcl_mu_eq_new;
			bcl_mu_in_inv = bcl_mu_in_inv_new;
			bcl_mu_eq_inv = bcl_mu_eq_inv_new;

		}
			
	}
	
	return {max_iter, n_mu_updates, n_tot};
}


} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS */
