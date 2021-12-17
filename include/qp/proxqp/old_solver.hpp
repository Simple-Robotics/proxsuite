#ifndef INRIA_LDLT_OLD_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_OLD_SOLVER_HPP_HDWGZKCLS

#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/proxqp/old_line_search.hpp"
#include <cmath>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}


namespace detail {

template <typename T>
struct TypeIdentityImpl {
	using Type = T;
};

template <typename T>
using DoNotDeduce = typename TypeIdentityImpl<T>::Type;

template <typename Dst, typename Lhs, typename Rhs>
void mul_add_no_alias(Dst& dst, Lhs const& lhs, Rhs const& rhs) {
	dst.noalias().operator+=(lhs.operator*(rhs));
}
template <typename Dst, typename Lhs, typename Rhs>
void mul_no_alias(Dst& dst, Lhs const& lhs, Rhs const& rhs) {
	dst.setZero();
	mul_add_no_alias(dst, lhs, rhs);
}

struct EqSolverTimer {};
struct QpSolveStats {
	isize n_ext;
	isize n_mu_updates;
	isize n_tot;
};

template <typename T>
void old_iterative_solve_with_permut_fact( //
		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& sol,
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>&  mat,
		T eps,
		isize max_it) {

	
	//LDLT_DECL_SCOPE_TIMER("in solver", "factorization", T);
	ldlt::Ldlt<T> ldl{decompose, mat};

	i32 it = 0;
	sol = rhs;
	ldl.solve_in_place(sol);
	auto res = (mat * sol - rhs).eval();
	std::cout <<"infty_norm(res) " << qp::infty_norm(res) << std::endl;
	while (qp::infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;
		res = (mat * sol - rhs);
	}
}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
void old_BCL_update(
		T& primal_feasibility_lhs,
		VectorViewMut<T> primal_residual_in_scaled_u,
		VectorViewMut<T> primal_residual_in_scaled_l,
		VectorViewMut<T> primal_residual_eq_scaled,
		Preconditioner precond,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		isize& n_mu_updates,
		T& bcl_mu_in,
		T& bcl_mu_eq,
		VectorViewMut<T> ye,
		VectorViewMut<T> ze,
		VectorViewMut<T> y,
		VectorViewMut<T> z
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
			bcl_eta_ext = bcl_eta_ext / pow(bcl_mu_in, T(0.9));
			bcl_eta_in = max2(bcl_eta_in /bcl_mu_in,eps_abs);
		} else {
			std::cout << "bad step"<< std::endl;
			y.to_eigen() = ye.to_eigen();
			z.to_eigen() = ze.to_eigen();
			T new_bcl_mu_in = std::min(bcl_mu_in * T(10), T(1e8));
			T new_bcl_mu_eq = std::min(bcl_mu_eq * T(10), T(1e10));
			if (bcl_mu_in != new_bcl_mu_in || bcl_mu_eq != new_bcl_mu_eq) {
					{
					++n_mu_updates;
					}
			}	
			bcl_mu_eq = new_bcl_mu_eq;
			bcl_mu_in = new_bcl_mu_in;
			bcl_eta_ext = (T(1)/pow(T(10),T(0.1))) / pow(bcl_mu_in, T(0.1));
			bcl_eta_in = max2(  T(1) / bcl_mu_in  ,eps_abs);
	}
}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
void old_global_primal_residual(
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
			Preconditioner precond,
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
					auto w_eq = residual_scaled_tmp.middleRows(dim,n_eq);
                    auto w_in = residual_scaled_tmp.bottomRows(n_in);
					w_eq = primal_residual_eq_scaled;
                    w_in = primal_residual_in_scaled_u;
					precond.unscale_primal_residual_in_place_eq(
							VectorViewMut<T>{from_eigen, w_eq});
                    precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, w_in});
					primal_feasibility_eq_rhs_0 = infty_norm(w_eq);
                    primal_feasibility_in_rhs_0 = infty_norm(w_in);
				}
				primal_residual_eq_scaled -= b_;
                primal_residual_in_scaled_u -= u_ ;
                primal_residual_in_scaled_l -= l_ ;

                primal_residual_in_scaled_u = ( primal_residual_in_scaled_u.array() > T(0)).select(primal_residual_in_scaled_u, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)); 
				primal_residual_in_scaled_l = ( primal_residual_in_scaled_l.array() < T(0)).select(primal_residual_in_scaled_l, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)); 
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
void old_global_dual_residual(
			T& dual_feasibility_lhs,
			T& dual_feasibility_rhs_0,
			T& dual_feasibility_rhs_1,
        	T& dual_feasibility_rhs_3,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  dual_residual_scaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  dual_residual_unscaled,
			Eigen::Matrix<T, Eigen::Dynamic, 1>&  residual_scaled_tmp,
			qp::QpViewBoxMut<T> qp_scaled,
			Preconditioner precond,
			VectorViewMut<T> x,
			VectorViewMut<T> y,
			VectorViewMut<T> z,
			isize dim,
			isize n_eq,
			isize n_in
		){
			LDLT_DECL_SCOPE_TIMER("in solver", "dual residual", T);
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
				auto w = residual_scaled_tmp.topRows(dim);
				w.setZero();
				w.array() = (H_ * x_).array();
				{ dual_residual_scaled += w; }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, w});
				dual_feasibility_rhs_0 = infty_norm(w);

				w.setZero();
				w.noalias() = A_.transpose() * y_;
				{ dual_residual_scaled += w; }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, w});
				dual_feasibility_rhs_1 = infty_norm(w);

				w.setZero();
				w.noalias() = C_.transpose() * z_;
				{ dual_residual_scaled += w; }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, w});
				dual_feasibility_rhs_3 = infty_norm(w);
			}

			dual_residual_unscaled = dual_residual_scaled;
			precond.unscale_dual_residual_in_place(
					VectorViewMut<T>{from_eigen, dual_residual_unscaled});

			dual_feasibility_lhs = infty_norm(dual_residual_unscaled);
        };


template<typename T> 
T old_SaddlePoint(
			qp::QpViewBox<T> qp_scaled,
			VectorViewMut<T> x,
			VectorViewMut<T> y,
        	VectorViewMut<T> z,
			VectorView<T> xe,
			VectorView<T> ye,
        	VectorView<T> ze,
			T mu_eq,
			T mu_in,
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

			prim_in_u.to_eigen().array() -=  (z_/mu_in).array() ; 
			prim_in_l.to_eigen().array() -= (z_/mu_in).array() ; 
			T prim_eq_e = infty_norm(prim_eq.to_eigen()) ; 
			dual_eq.to_eigen().array() += (C_.transpose()*z_).array();
			T dual_e = infty_norm(dual_eq.to_eigen());
			T err = max2(prim_eq_e,dual_e);

			T prim_in_e(0);

			for (isize i = 0 ; i< n_in ; i=i+1){
				if (z_(i) >0){
					prim_in_e = max2(prim_in_e,std::abs(prim_in_u(i)));
				}else if (z_(i) < 0){
					prim_in_e = max2(prim_in_e,std::abs(prim_in_l(i)));
				}else{
					prim_in_e = max2(prim_in_e,max2(prim_in_u(i),T(0))) ;
					prim_in_e = max2(prim_in_e, std::abs(std::min(prim_in_l(i),T(0))) );
				}
			}
			err = max2(err,prim_in_e);
			return err;
}


template<typename T>
void old_newton_step_new(
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> dx,
		T mu_eq,
		T mu_in,
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
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& active_inequalities
	){
		
		auto H_ = qp_scaled.H.to_eigen();
		auto A_ = qp_scaled.A.to_eigen();
		auto C_ = qp_scaled.C.to_eigen();
		
		l_active_set_n_u = (z_pos.array() > T(0)).matrix();
		l_active_set_n_l = (z_neg.array() < T(0)).matrix();

		active_inequalities = l_active_set_n_u || l_active_set_n_l ; 

		isize num_active_inequalities = active_inequalities.count();
		isize inner_pb_dim = dim + n_eq + num_active_inequalities;
		
		LDLT_MULTI_WORKSPACE_MEMORY(
			(_htot,Uninit, Mat(inner_pb_dim, inner_pb_dim),LDLT_CACHELINE_BYTES, T),
			(_rhs,Init, Vec(inner_pb_dim),LDLT_CACHELINE_BYTES, T),
			(_dw,Init, Vec(inner_pb_dim),LDLT_CACHELINE_BYTES, T)
		);
		
		auto Htot = _htot.to_eigen().eval();
		auto rhs = _rhs.to_eigen().eval();
		rhs.setZero();
		Htot.setZero();

		auto dw = _dw.to_eigen().eval();
		dw.setZero();
		
		{
				rhs.topRows(dim) -=  dual_for_eq ;
				for (isize j = 0 ; j < n_in; ++j){
					rhs.topRows(dim) -= mu_in*(max2(z_pos(j),T(0)) + std::min(z_neg(j),T(0))) * C_.row(j) ; 
				}
				LDLT_DECL_SCOPE_TIMER("in solver", "set H", T);
				Htot.topLeftCorner(dim, dim) = H_;
				for (isize i = 0; i < dim; ++i) {
					Htot(i, i) += rho;
				}
				
				Htot.block(0,dim,dim,n_eq) = A_.transpose();
				Htot.block(dim,0,n_eq,dim) = A_;
				Htot.bottomRightCorner(n_eq+num_active_inequalities, n_eq+num_active_inequalities).setZero();
				{
					T tmp_eq = -T(1) / mu_eq;
					T tmp_in = -T(1) / mu_in;
					for (isize i = 0; i < n_eq; ++i) {
						Htot(dim + i, dim + i) = tmp_eq;
					}
					for (isize i = 0; i < num_active_inequalities; ++i) {
						Htot(dim + n_eq + i, dim + n_eq + i) = tmp_in;
					}
				}
				isize j = 0;
				for (isize i = 0; i< n_in ; ++i){
					if (l_active_set_n_u(i)){
						Htot.block(j+dim+n_eq,0,1,dim) = C_.row(i) ; 
						Htot.block(0,j+dim+n_eq,dim,1) = C_.transpose().col(i) ; 
						j+=1 ;
					}else if (l_active_set_n_l(i)){
						Htot.block(j+dim+n_eq,0,1,dim) = C_.row(i) ; 
						Htot.block(0,j+dim+n_eq,dim,1) = C_.transpose().col(i) ; 
						j+=1;
					}	
				}
		}

		old_iterative_solve_with_permut_fact( //
			rhs,
			dw,
			Htot,
			eps,
			isize(5));
		
		dx.to_eigen() = dw.topRows(dim);
}

template<typename T,typename Preconditioner>
T old_initial_guess(
		VectorView<T> xe,
		VectorView<T> ye,
        VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
        VectorViewMut<T> z,
		qp::QpViewBoxMut<T> qp_scaled,
		T mu_in,
		T mu_eq,
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
		Eigen::Matrix<T, Eigen::Dynamic, 1>& dw_aug
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

			prim_in_u =  (C_*x_-u_) ; 
			prim_in_l = (C_*x_-l_) ; 

			precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_u});
			precond.unscale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_l});
			precond.unscale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, z_e}); 

			prim_in_u.array() += (z_e/mu_in).array() ; 

			prim_in_l.array() += (z_e/mu_in).array() ; 

			l_active_set_n_u = (prim_in_u.array() >= T(0)).matrix();
			l_active_set_n_l = (prim_in_l.array() <= T(0)).matrix();

			active_inequalities = l_active_set_n_u || l_active_set_n_l ; 

			prim_in_u.array() -= (z_e/mu_in).array() ; 
			prim_in_l.array() -= (z_e/mu_in).array() ; 

			precond.scale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_u});
			precond.scale_primal_residual_in_place_in(
							VectorViewMut<T>{from_eigen, prim_in_l});
			precond.scale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, z_e});
			// rescale value
			isize num_active_inequalities = active_inequalities.count();
			isize inner_pb_dim = dim + n_eq + num_active_inequalities;

			std::cout << " active_inequalities " << active_inequalities << std::endl;

			LDLT_MULTI_WORKSPACE_MEMORY(
			(_htot,Uninit, Mat(inner_pb_dim, inner_pb_dim),LDLT_CACHELINE_BYTES, T),
			(_rhs,Init, Vec(inner_pb_dim),LDLT_CACHELINE_BYTES, T),
			(_dw,Init, Vec(inner_pb_dim),LDLT_CACHELINE_BYTES, T)
			);
			auto Htot = _htot.to_eigen().eval();
			auto rhs = _rhs.to_eigen().eval();
			auto dw = _dw.to_eigen().eval();
			
			rhs.topRows(dim) = -dual_for_eq ;
			rhs.middleRows(dim,n_eq) = -primal_residual_eq ;

			LDLT_DECL_SCOPE_TIMER("in solver", "set H", T);
			Htot.topLeftCorner(dim, dim) = H_;
			for (isize i = 0; i < dim; ++i) {
					Htot(i, i) += rho;
			}
			
			Htot.block(0,dim,dim,n_eq) = A_.transpose();
			Htot.block(dim,0,n_eq,dim) = A_;
			Htot.bottomRightCorner(n_eq+num_active_inequalities, n_eq+num_active_inequalities).setZero();
			{
					T tmp_eq = -T(1) / mu_eq;
					T tmp_in = -T(1) / mu_in;
					for (isize i = 0; i < n_eq; ++i) {
						Htot(dim + i, dim + i) = tmp_eq;
					}
					for (isize i = 0; i < num_active_inequalities; ++i) {
						Htot(dim + n_eq + i, dim + n_eq + i) = tmp_in;
					}
			}
			isize j = 0;
			for (isize i = 0; i< n_in ; ++i){
					if (l_active_set_n_u(i)){
						Htot.block(j+dim+n_eq,0,1,dim) = C_.row(i) ; 
						Htot.block(0,j+dim+n_eq,dim,1) = C_.transpose().col(i) ; 
						rhs(j+dim+n_eq) = -prim_in_u(i);
						//rhs.topRows(dim) -= C_.transpose().col(i) * z_(i);
						j+=1 ;
					}else if (l_active_set_n_l(i)){
						Htot.block(j+dim+n_eq,0,1,dim) = C_.row(i) ; 
						Htot.block(0,j+dim+n_eq,dim,1) = C_.transpose().col(i) ; 
						rhs(j+dim+n_eq) = -prim_in_l(i);
						//rhs.topRows(dim) -= C_.transpose().col(i) * z_(i);
						j+=1;
					}else{
						rhs.topRows(dim) += C_.transpose().col(i) * z_(i); //unactive unrelevant columns
					}
			}
			std::cout << "rhs " << rhs << std::endl;
			old_iterative_solve_with_permut_fact( //
				rhs,
				dw,
				Htot,
				eps_int,
				isize(5));
			dw_aug.setZero();
			dw_aug.topRows(dim+n_eq) = dw.topRows(dim+n_eq) ;
			isize j_aug = 0;
			for (isize i =0 ; i<n_in ; ++i){
				if (l_active_set_n_u(i)){
					dw_aug(dim+n_eq+i) = dw(dim+n_eq+j_aug) ; 
					j_aug +=1;
				}else if (l_active_set_n_l(i)){
					dw_aug(dim+n_eq+i) = dw(dim+n_eq+j_aug) ; 
					j_aug +=1;
				} else{
					dw_aug(dim+n_eq+i) -= z(i) ; 
				}
			}
			std::cout << "dw_aug " << dw_aug << std::endl;
			prim_in_u.array() += (z_e/mu_in).array() ;  
			prim_in_l.array() += (z_e/mu_in).array() ; 

			d_primal_residual_eq = (A_*dw_aug.topRows(dim)- dw_aug.middleRows(dim,n_eq) / mu_eq).eval() ;
			d_dual_for_eq = (H_*dw_aug.topRows(dim)+A_.transpose()*dw_aug.middleRows(dim,n_eq)+rho*dw_aug.topRows(dim)).eval() ;
			Cdx_ = C_*dw_aug.topRows(dim) ;
			dual_for_eq -= C_.transpose()*z_e ; 
			T R = 1.e6;
			T alpha_step = qp::line_search::old_initial_guess_LS(
								ze,
								VectorView<T>{from_eigen,dw_aug.tail(n_in)},
								VectorView<T>{from_eigen,prim_in_l},
								VectorView<T>{from_eigen,prim_in_u},
								VectorView<T>{from_eigen,Cdx_},
								VectorView<T>{from_eigen,d_dual_for_eq},
								VectorView<T>{from_eigen,dual_for_eq},
								VectorView<T>{from_eigen,d_primal_residual_eq},
								VectorView<T>{from_eigen,primal_residual_eq},
								qp_scaled.C.as_const(),
								mu_eq,
								mu_in,
								rho,
								dim,
								n_eq,
								n_in,
								R
			);
			
			std::cout << "alpha from initial guess " << alpha_step << std::endl;

			prim_in_u.array() += (alpha_step*Cdx_).array();
			prim_in_l.array() += (alpha_step*Cdx_).array();
			l_active_set_n_u = (prim_in_u.array() >= T(0)).matrix();
			l_active_set_n_l = (prim_in_l.array() <= T(0)).matrix();
			active_inequalities = l_active_set_n_u || l_active_set_n_l ; 

			x.to_eigen().array() += (alpha_step * dw_aug.topRows(dim).array()) ; 
			y.to_eigen().array() += (alpha_step * dw_aug.middleRows(dim,n_eq).array()) ; 

			for (isize i = 0; i< n_in ; ++i){
				if (l_active_set_n_u(i)){
					z(i) = max2(z(i)+alpha_step*dw_aug(dim+n_eq+i),T(0)) ; 
				}else if (l_active_set_n_l(i)){
					z(i) = std::min(z(i)+alpha_step*dw_aug(dim+n_eq+i),T(0)) ; 
				} else{
					z(i) += alpha_step*dw_aug(dim+n_eq+i) ; 
				}
			}
			primal_residual_eq.array() += (alpha_step*d_primal_residual_eq).array();
			dual_for_eq.array() += alpha_step* (d_dual_for_eq.array()) ;

			T err = old_SaddlePoint(
				qp_scaled.as_const(),
				x,
				y,
				z,
				xe,
				ye,
				ze,
				mu_eq, 
				mu_in,
				rho,
				n_in,
				VectorViewMut<T>{from_eigen,prim_in_u},
				VectorViewMut<T>{from_eigen,prim_in_l},
				VectorViewMut<T>{from_eigen,primal_residual_eq},
				VectorViewMut<T>{from_eigen,dual_for_eq}
			);
			std::cout << "x " << x.to_eigen() << std::endl;
			std::cout << "y " << y.to_eigen() << std::endl;
			std::cout << "z " << z.to_eigen() << std::endl;
			return err;
}


template<typename T>
T old_correction_guess(
		VectorView<T> xe,
		VectorView<T> ye,
        VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
        VectorViewMut<T> z,
		qp::QpViewBoxMut<T> qp_scaled,
		T mu_in,
		T mu_eq,
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
		Eigen::Matrix<bool,Eigen::Dynamic,1>& active_inequalities
		){

		T err_in = T(0);

		for (i64 iter = 0; iter <= max_iter_in; ++iter) {

			if (iter == max_iter_in) {
				n_tot +=max_iter_in;
				break;
			}

			Eigen::Matrix<T, Eigen::Dynamic, 1> dx(dim);
			dx.setZero();
			qp::detail::old_newton_step_new<T>(
											qp_scaled.as_const(),
											x.as_const(),
											xe,
											ye,
											ze,
											VectorViewMut<T>{from_eigen,dx},
											mu_eq,
											mu_in,
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
											active_inequalities
			);
			T alpha_step = T(1);
			Hdx = (qp_scaled.H).to_eigen() * dx ; 
			Adx = (qp_scaled.A).to_eigen() * dx ; 
			Cdx = (qp_scaled.C).to_eigen() * dx ; 
			if (n_in > isize(0)){
				alpha_step = qp::line_search::old_correction_guess_LS(
										Hdx,
									 	dx,
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
										n_in
						
				) ;
			}
			if (infty_norm(alpha_step * dx)< T(1.E-11)){
				n_tot += iter+1;
				break;
			}
			
			x.to_eigen().array() += (alpha_step *dx.array()) ; 
			z_pos.array() += (alpha_step *Cdx).array() ;
			z_neg.array() += (alpha_step *Cdx).array() ; 
			residual_in_y.array() += (alpha_step * Adx).array();
 			y.to_eigen() = mu_eq *  residual_in_y  ;
			dual_for_eq.array() += (alpha_step * ( mu_eq * (qp_scaled.A).to_eigen().transpose() * Adx + rho * dx + Hdx  )).array() ;
			for (isize j = 0 ; j < n_in; ++j){
				z(j) = mu_in*(max2(z_pos(j),T(0)) + std::min(z_neg(j),T(0))) ; 
			}

			auto tmp1 = (qp_scaled.H).to_eigen() *x.to_eigen() ;
			auto tmp2 = (qp_scaled.A.to_eigen().transpose()) * ( y.to_eigen() );
			auto tmp3 = (qp_scaled.C.to_eigen().transpose()) * ( z.to_eigen() )   ; 
			auto grad_n = tmp1 + tmp2 + tmp3  + (qp_scaled.g).to_eigen() ;

			err_in = infty_norm(  grad_n + rho* (x.to_eigen()-xe.to_eigen()) );
			std::cout << "---it in " << iter << " projection norm " << err_in << " alpha " << alpha_step << std::endl;

			if (err_in<= eps_int * (1 + max2(max2(max2(infty_norm(tmp1), infty_norm(tmp2)), infty_norm(tmp3)), infty_norm((qp_scaled.g).to_eigen())) )  ){
				n_tot +=iter+1;
				break;
			}
		}
	
		return err_in;

}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpSolveStats old_qpSolve( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
        VectorViewMut<T> z,
		qp::QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		Preconditioner precond = Preconditioner{}) {

	using namespace ldlt::tags;
    static constexpr Layout layout = rowmajor;
    static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
    isize n_in = qp.C.rows;
	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6);
	auto bcl_mu_eq = T(1e3);
    auto bcl_mu_in = T(1e1);
	T bcl_eta_ext = 1 / pow(bcl_mu_in, T(0.1));
    T bcl_eta_in = T(1);
	
	LDLT_MULTI_WORKSPACE_MEMORY(
			//(_h_scaled,Uninit, Mat(dim, dim),LDLT_CACHELINE_BYTES, T), /!\ creates by default a column major based matrix
			//(_h_ws,Uninit, Mat(dim, dim),LDLT_CACHELINE_BYTES, T),
	     	(_g_scaled,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
	     	//(_a_scaled,Uninit, Mat(n_eq, dim),LDLT_CACHELINE_BYTES, T),
		 	//(_c_scaled,Uninit, Mat(n_in, dim),LDLT_CACHELINE_BYTES, T),
	     	(_b_scaled,Init, Vec(n_eq),LDLT_CACHELINE_BYTES, T),
         	(_u_scaled,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
         	(_l_scaled,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
	     	(_residual_scaled,Init, Vec(dim + n_eq+n_in),LDLT_CACHELINE_BYTES, T),
	     	(_residual_scaled_tmp,Init, Vec(dim + n_eq+n_in),LDLT_CACHELINE_BYTES, T),
	     	(_residual_unscaled,Init, Vec(dim + n_eq+n_in),LDLT_CACHELINE_BYTES, T),
	     	(_y,Init, Vec(n_eq),LDLT_CACHELINE_BYTES, T),
         	(_z,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(xe_,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
	     	(_diag_diff_eq,Init, Vec(n_eq),LDLT_CACHELINE_BYTES, T),
         	(_diag_diff_in,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			
			(_dw_aug,Init, Vec(dim+n_eq+n_in),LDLT_CACHELINE_BYTES, T),
			(d_dual_for_eq_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(Cdx__,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(d_primal_residual_eq_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(l_active_set_n_u_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, bool),
			(l_active_set_n_l_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, bool),
			(active_inequalities_,Init, Vec(n_in),LDLT_CACHELINE_BYTES, bool)
		);


    RowMat H_copy(dim,dim);
    RowMat H_ws(dim,dim);
    RowMat A_copy(n_eq,dim);
    RowMat C_copy(n_in,dim);
    H_copy.setZero();
    H_ws.setZero();
    A_copy.setZero();
    C_copy.setZero();

	Eigen::Matrix<T, Eigen::Dynamic, 1> d_dual_for_eq = d_dual_for_eq_.to_eigen();
	Eigen::Matrix<T, Eigen::Dynamic, 1> Cdx_ = Cdx__.to_eigen();
	Eigen::Matrix<T, Eigen::Dynamic, 1> d_primal_residual_eq = d_primal_residual_eq_.to_eigen();
	Eigen::Matrix<bool, Eigen::Dynamic, 1> l_active_set_n_u = l_active_set_n_u_.to_eigen();
	Eigen::Matrix<bool, Eigen::Dynamic, 1> l_active_set_n_l = l_active_set_n_l_.to_eigen();
	Eigen::Matrix<bool, Eigen::Dynamic, 1> active_inequalities = active_inequalities_.to_eigen();	
	Eigen::Matrix<T, Eigen::Dynamic, 1> dw_aug = _dw_aug.to_eigen(); 

	//auto H_copy = _h_scaled.to_eigen();
    //_h_scaled = {from_eigen,H_copy};
	//auto c = _h_ws.to_eigen();
    //_h_ws = {from_eigen,H_ws};
	auto q_copy = _g_scaled.to_eigen();
	//auto A_copy = _a_scaled.to_eigen();
    // _a_scaled = {from_eigen,A_copy};
	//auto C_copy = _c_scaled.to_eigen();
    //_c_scaled = {from_eigen,C_copy};
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

    /*
	qp::QpViewBoxMut<T> qp_scaled{
			ldlt::MatrixViewMut<T,layout>{ldlt::from_eigen, H_copy},
			ldlt::VectorViewMut<T>{ldlt::from_eigen, q_copy},
			ldlt::MatrixViewMut<T,layout>{ldlt::from_eigen, A_copy},
			ldlt::VectorViewMut<T>{ldlt::from_eigen, b_copy},
			ldlt::MatrixViewMut<T,layout>{ldlt::from_eigen, C_copy},
			ldlt::VectorViewMut<T>{ldlt::from_eigen, u_copy},
            ldlt::VectorViewMut<T>{ldlt::from_eigen, l_copy}
	};
    */
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
	/*
	H_ws = H_copy;
	for (isize i = 0;i< dim ; ++i){
		H_ws(i,i) += rho ;
	}

	ldlt::Ldlt<T> ldl{decompose, H_ws};
	x.to_eigen().array() = -(qp_scaled.g).to_eigen().array();
	ldl.solve_in_place(x.to_eigen());
	*/
	///
	RowMat kkt(dim+n_eq,dim+n_eq);
	Eigen::Matrix<T, Eigen::Dynamic, 1> rhs_ws(dim+n_eq);
	rhs_ws.setZero();
	kkt.setZero();

	kkt.topLeftCorner(dim, dim) = H_copy ;
	kkt.topLeftCorner(dim, dim).diagonal().array() += rho;	
	kkt.block(0, dim, dim, n_eq) = A_copy.transpose();
	kkt.block(dim, 0, n_eq, dim) = A_copy;
	kkt.bottomRightCorner(n_eq, n_eq).setZero();
	kkt.diagonal().segment(dim, n_eq).setConstant(-T(1)/bcl_mu_eq); // mu stores the inverse of mu

	ldlt::Ldlt<T> ldl{decompose, kkt};
	rhs_ws.head(dim) = -qp_scaled.g.to_eigen();
	rhs_ws.segment(dim,n_eq) = qp_scaled.b.to_eigen();

	ldl.solve_in_place(rhs_ws.head(dim+n_eq));
	x.to_eigen() = rhs_ws.head(dim);
	y.to_eigen() = rhs_ws.segment(dim,n_eq);

	///
	auto residual_scaled = _residual_scaled.to_eigen();
	Eigen::Matrix<T, Eigen::Dynamic, 1> residual_scaled_tmp = _residual_scaled_tmp.to_eigen();
	auto residual_unscaled = _residual_unscaled.to_eigen();

	auto ye = _y.to_eigen();
    auto ze = _z.to_eigen();
	auto xe = xe_.to_eigen();
	auto diag_diff_in = _diag_diff_in.to_eigen();
	auto diag_diff_eq = _diag_diff_eq.to_eigen();

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
    T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
    T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());
	
	Eigen::Matrix<T, Eigen::Dynamic, 1> dual_residual_scaled = residual_scaled.topRows(dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_eq_scaled = residual_scaled.middleRows(dim,n_eq);
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_in_scaled_u = residual_scaled.bottomRows(n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_in_scaled_l = residual_scaled.bottomRows(n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> dual_residual_unscaled = residual_unscaled.topRows(dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_eq_unscaled = residual_unscaled.middleRows(dim,n_eq);
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_in_u_unscaled = residual_unscaled.bottomRows(n_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_in_l_unscaled = residual_unscaled.bottomRows(n_in);

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);


	std::cout << "x " << x.to_eigen() << std::endl;
	std::cout << "y " << y.to_eigen() << std::endl;
	std::cout << "z " << z.to_eigen() << std::endl;
	for (i64 iter = 0; iter <= max_iter; ++iter) {
		n_ext +=1;
		if (iter == max_iter) {
			break;
		}

		// compute primal residual
		
		qp::detail::old_global_primal_residual(
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
		qp::detail::old_global_dual_residual(
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

			rho = max2(rho / T(10), T(1e-7));
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

		T err_in = qp::detail::old_initial_guess<T,Preconditioner>(
						VectorView<T>{from_eigen,xe},
						VectorView<T>{from_eigen,ye},
						VectorView<T>{from_eigen,ze},
						x,
						y,
						z,
						qp_scaled,
						bcl_mu_in,
						bcl_mu_eq,
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
						dw_aug
		);
		n_tot +=1;
		std::cout << " error from initial guess : " << err_in << " bcl_eta_in " << bcl_eta_in << std::endl;
		
		if (err_in >= bcl_eta_in){
			
			dual_residual_scaled.array() += (-(qp_scaled.C).to_eigen().transpose()*z.to_eigen() + bcl_mu_eq * (qp_scaled.A).to_eigen().transpose()*primal_residual_eq_scaled ).array();
			primal_residual_eq_scaled.array() += (y.to_eigen()/bcl_mu_eq).array();
			primal_residual_in_scaled_u.array() += (z.to_eigen()/bcl_mu_in).array();
			primal_residual_in_scaled_l.array() += (z.to_eigen()/bcl_mu_in).array();

			err_in = qp::detail::old_correction_guess(
						VectorView<T>{from_eigen,xe},
						VectorView<T>{from_eigen,ye},
						VectorView<T>{from_eigen,ze},
						x,
						y,
						z,
						qp_scaled,
						bcl_mu_in,
						bcl_mu_eq,
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
						active_inequalities
			);
			//std::cout << " error from correction guess : " << err_in << std::endl;
		}
		
		T primal_feasibility_lhs_new(primal_feasibility_lhs) ; 

		qp::detail::old_global_primal_residual(
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

		qp::detail::old_BCL_update(
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
					VectorViewMut<T>{from_eigen,ye},
					VectorViewMut<T>{from_eigen,ze},
					y,
					z
		);

		// COLD RESTART
		
		T dual_feasibility_lhs_new(dual_feasibility_lhs) ; 

		
		qp::detail::old_global_dual_residual(
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

		if ((primal_feasibility_lhs_new / max2(primal_feasibility_lhs,machine_eps) >= T(1)) && (dual_feasibility_lhs_new / max2(primal_feasibility_lhs,machine_eps) >= T(1)) && bcl_mu_in >= T(1.E5)){
			std::cout << "cold restart" << std::endl;
			bcl_mu_in = T(1.1);
			bcl_mu_eq = T(1.1);
		}
		
	}
	
	return {max_iter, n_mu_updates, n_tot};
}


} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_SOLVER_HPP_HDWGZKCLS */
