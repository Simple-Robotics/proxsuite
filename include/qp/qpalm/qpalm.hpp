#ifndef INRIA_LDLT_QPALM_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_QPALM_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "qp/utils.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/qpalm/line_search_qpalm.hpp"
#include "qp/precond/identity.hpp"
#include <cmath>
#include <type_traits>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

struct QpalmSolveStats{
	double n_ext;
	double n_mu_updates;
	double n_tot;
};

#define LDLT_DEDUCE_RET(...)                                                   \
	noexcept(noexcept(__VA_ARGS__))                                              \
			->typename std::remove_const<decltype(__VA_ARGS__)>::type {              \
		return __VA_ARGS__;                                                        \
	}                                                                            \
	static_assert(true, ".")

template <typename T>
void iterative_residual_QPALM(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> err_,
		VectorView<T> mu_,
		T rho) {
	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto err = err_.to_eigen();
	auto mu = mu_.to_eigen();
	auto current_bijection_map = current_bijection_map_.to_eigen();

	err = (-rhs).eval();
	err.topRows(dim) +=
			(qp_scaled.H).to_eigen() * sol.topRows(dim) + rho * sol.topRows(dim) +
			(qp_scaled.A).to_eigen().transpose() * sol.middleRows(dim, n_eq);

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			err.topRows(dim) += sol(dim + n_eq + j) * qp_scaled.C.to_eigen().row(i);
			err(dim + n_eq + j) +=
					(qp_scaled.C.to_eigen().row(i)).dot(sol.topRows(dim)) -
					sol(dim + n_eq + j) / mu(n_eq + i);
		}
	}

	err.middleRows(dim, n_eq).array() +=
			((qp_scaled.A).to_eigen() * sol.topRows(dim)).array() -
			sol.middleRows(dim, n_eq).array() / mu.topRows(n_eq).array();
}

template <typename T>
void iterative_solve_with_permut_fact_QPALM( //
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> res_,
		ldlt::Ldlt<T>& ldl,
		T eps,
		isize max_it,
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		VectorView<T> mu_,
		T rho,
		bool& VERBOSE) {

	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto res = res_.to_eigen();
	auto mu = mu_.to_eigen();

	i32 it = 0;
	sol = rhs;
	ldl.solve_in_place(sol);

	qp::detail::iterative_residual_QPALM<T>(
			qp_scaled,
			current_bijection_map_,
			dim,
			n_eq,
			n_c,
			n_in,
			{from_eigen, rhs},
			{from_eigen, sol},
			{from_eigen, res},
			{from_eigen, mu},
			rho);
	if (VERBOSE){
		std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;
	}
	while (qp::infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;

		res.setZero();
		qp::detail::iterative_residual_QPALM<T>(
				qp_scaled,
				current_bijection_map_,
				dim,
				n_eq,
				n_c,
				n_in,
				{from_eigen, rhs},
				{from_eigen, sol},
				{from_eigen, res},
				{from_eigen, mu},
				rho);
		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;
		}
	}
}


 template <typename T>
 void QPALM_mu_update(
 		T& primal_feasibility_lhs,
 		VectorView<T> primal_residual_eq_scaled_,
 		VectorView<T> primal_residual_in_scaled_l_,
 		VectorView<T> primal_residual_eq_scaled_old_,
 		VectorView<T> primal_residual_in_scaled_in_old_,
		VectorView<isize> current_bijection_map_,
 		isize& n_mu_updates,
 		VectorViewMut<T> mu_,
 		isize dim,
 		isize n_eq,
		isize n_in,
 		isize& n_c,
 		ldlt::Ldlt<T>& ldl,
 		qp::QpViewBox<T> qp_scaled,
 		T rho,
 		T theta,
 		T sigmaMax,
 		T Delta,
		isize& mu_update,
		T max_rank_update,
		T max_rank_update_fraction,
		bool& VERBOSE) {

	auto current_bijection_map = current_bijection_map_.to_eigen();
	auto primal_residual_eq_scaled = primal_residual_eq_scaled_.to_eigen();
	auto primal_residual_in_scaled_l = primal_residual_in_scaled_l_.to_eigen();
	auto primal_residual_eq_scaled_old = primal_residual_eq_scaled_old_.to_eigen();
	auto primal_residual_in_scaled_in_old = primal_residual_in_scaled_in_old_.to_eigen();
	auto mu = mu_.to_eigen();

 	for (isize i = 0; i < n_eq; ++i) {
 		if (primal_residual_eq_scaled(i) >=
 		    theta * primal_residual_eq_scaled_old(i)) {
 			T mu_eq_new = min2(
 					sigmaMax,
 					max2(
 							mu(i) * Delta * primal_residual_eq_scaled(i) /
 									primal_feasibility_lhs,
 							mu(i)));
			n_mu_updates +=1;
 		}
 	}

 	for (isize i = 0; i < n_in; ++i) {
		isize j = current_bijection_map(i) ;
		if (j < n_c){
			if (primal_residual_in_scaled_l(i) >=
				theta * primal_residual_in_scaled_in_old(i)) {
				T mu_in_new = min2( sigmaMax, max2( mu(n_eq + i) * Delta * primal_residual_in_scaled_l(i) / primal_feasibility_lhs, mu(n_eq + i) ) );
				n_mu_updates +=1;
			}
		}
 	}

	if (T(n_mu_updates) <= min2(max_rank_update * T(0.25), max_rank_update_fraction * (dim+n_eq+n_in)  )) {

		for (isize i = 0; i < n_eq; ++i) {
			if (primal_residual_eq_scaled(i) >=
				theta * primal_residual_eq_scaled_old(i)) {
				T mu_eq_new = min2(
						sigmaMax,
						max2(
								mu(i) * Delta * primal_residual_eq_scaled(i) /
										primal_feasibility_lhs,
								mu(i)));
				LDLT_MULTI_WORKSPACE_MEMORY(
						(e_k_, Init, Vec(dim + n_eq + n_c), LDLT_CACHELINE_BYTES, T));
				auto e_k = e_k_.to_eigen().eval();
				T diff = T(1) / mu(i) - T(1) / mu_eq_new;
				e_k(dim + i) = T(1);
				ldl.rank_one_update(e_k, diff);
				e_k(dim + i) = T(0);
				mu(i) = mu_eq_new;
			}
		}

		for (isize i = 0; i < n_in; ++i) {
			isize j = current_bijection_map(i) ;
			if (j < n_c){
				if (primal_residual_in_scaled_l(i) >=
					theta * primal_residual_in_scaled_in_old(i)) {
					T mu_in_new = min2( sigmaMax, max2( mu(n_eq + i) * Delta * primal_residual_in_scaled_l(i) / primal_feasibility_lhs, mu(n_eq + i) ) );
					LDLT_MULTI_WORKSPACE_MEMORY(
							(e_k_, Init, Vec(dim + n_eq + n_c), LDLT_CACHELINE_BYTES, T));
					auto e_k = e_k_.to_eigen().eval();
					T diff = T(1) / mu(n_eq + i) - T(1) / mu_in_new;
					e_k(dim + n_eq + j) = T(1);
					ldl.rank_one_update(e_k, diff);
					e_k(dim + n_eq + j) = T(0);
					mu(n_eq + i) = mu_in_new;
				}
			}
		}

		n_mu_updates = 0;

	}else{

		[&]{
			LDLT_MULTI_WORKSPACE_MEMORY(
				(_htot,Uninit, Mat(dim+n_eq+n_c, dim+n_eq+n_c),LDLT_CACHELINE_BYTES, T)
			);
			
			auto Htot = _htot.to_eigen().eval();
			Htot.setZero();
			
			Htot.topLeftCorner(dim, dim) = qp_scaled.H.to_eigen();
			for (isize i = 0; i < dim; ++i) {
				Htot(i, i) += rho; 
			}
			
			Htot.block(0,dim,dim,n_eq) = qp_scaled.A.to_eigen().transpose();
			Htot.block(dim,0,n_eq,dim) = qp_scaled.A.to_eigen();
			{
				for (isize i = 0; i < n_eq; ++i) {

					if (primal_residual_eq_scaled(i) >=
						theta * primal_residual_eq_scaled_old(i)) {
							T mu_eq_new = min2(
								sigmaMax,
								max2(
										mu(i) * Delta * primal_residual_eq_scaled(i) /
												primal_feasibility_lhs,
										mu(i)));
							mu(i) = mu_eq_new ;
						}

					Htot(dim + i, dim + i) = -T(1) / mu( i);
				}
			}
			for (isize i = 0; i< n_in ; ++i){ 
				isize j = current_bijection_map(i);
				if (j<n_c){
					
					if (primal_residual_in_scaled_l(i) >=
					theta * primal_residual_in_scaled_in_old(i)) {
						mu(n_eq+i) = min2( sigmaMax, max2( mu(n_eq + i) * Delta * primal_residual_in_scaled_l(i) / primal_feasibility_lhs, mu(n_eq + i) ) );
					}

					Htot.block(j+dim+n_eq,0,1,dim) = qp_scaled.C.to_eigen().row(i) ; 
					Htot.block(0,j+dim+n_eq,dim,1) = qp_scaled.C.to_eigen().transpose().col(i)  ; 
					Htot(dim + n_eq + j, dim + n_eq + j) = -T(1) / mu(n_eq + i);
				}
			}
			ldl.factorize(Htot);
			if (VERBOSE){
				std::cout << "error " <<  qp::infty_norm(Htot-ldl.reconstructed_matrix()) << std::endl; 
			}
		}();

		n_mu_updates = 0;
	}

}

template <typename T>
void QPALM_update_fact(
		T& primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		VectorViewMut<T> xe,
		VectorViewMut<T> ye,
		VectorViewMut<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		bool& VERBOSE) {

	if (primal_feasibility_lhs <= bcl_eta_ext) {
		if (VERBOSE){
			std::cout << "good step" << std::endl;
		}
		bcl_eta_ext = bcl_eta_ext / T(10);
		bcl_eta_in = max2(bcl_eta_in / T(10), eps_abs);
		ye.to_eigen() = y.to_eigen();
		ze.to_eigen() = z.to_eigen();
		xe.to_eigen() = x.to_eigen();
	} else {
		if (VERBOSE){
			std::cout << "bad step" << std::endl;
		}
		bcl_eta_in = max2(bcl_eta_in / T(10), eps_abs);
		// In convex case only for QPALM
		ye.to_eigen() = y.to_eigen();
		ze.to_eigen() = z.to_eigen();
		xe.to_eigen() = x.to_eigen();
	}
}

template <typename T>
void newton_step_QPALM(
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> dx,
		VectorView<T> mu_,
		T rho,
		T eps,
		isize dim,
		isize n_eq,
		isize n_in,
		VectorView<T> z_pos_,
		VectorView<T> z_neg_,
		VectorView<T> dual_for_eq_,
		VectorViewMut<bool> l_active_set_n_u_,
		VectorViewMut<bool> l_active_set_n_l_,
		VectorViewMut<bool> active_inequalities_,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		T max_rank_update,
		T max_rank_update_fraction,
		isize& nb_enter,
		isize& nb_leave,
		bool& VERBOSE
		) {

	/* NB
	* dual_for_eq_ = Hx + rho (x-xe) + AT( mu(Ax-b) + ye )
	* z_pos_ = Cx-u + ze / mu
	* z_neg_ = Cx-l + ze / mu
	*/

	auto H_ = qp_scaled.H.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();

	auto mu = mu_.to_eigen();
	auto z_pos = z_pos_.to_eigen();
	auto z_neg = z_neg_.to_eigen();
	auto dual_for_eq = dual_for_eq_.to_eigen();
	auto l_active_set_n_u = l_active_set_n_u_.to_eigen();
	auto l_active_set_n_l = l_active_set_n_l_.to_eigen();
	auto active_inequalities = active_inequalities_.to_eigen();

	l_active_set_n_u = (z_pos.array() > T(0)).matrix();
	l_active_set_n_l = (z_neg.array() < T(0)).matrix();

	active_inequalities = l_active_set_n_u || l_active_set_n_l;

	isize num_active_inequalities = active_inequalities.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_rhs, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_err, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T));

	auto rhs = _rhs.to_eigen();
	auto dw = _dw.to_eigen();
	auto err = _err.to_eigen();
	qp::line_search_qpalm::active_set_change_QPALM(
			VectorView<bool>{from_eigen, active_inequalities},
			current_bijection_map,
			n_c,
			n_in,
			dim,
			n_eq,
			ldl,
			qp_scaled,
			mu_,
			rho,
			max_rank_update,
			max_rank_update_fraction,
			nb_enter,
			nb_leave,
			VERBOSE
			);
	rhs.topRows(dim) -= dual_for_eq;
	for (isize j = 0; j < n_in; ++j) {
		rhs.topRows(dim) -= mu(n_eq + j) *
							(max2(z_pos(j), T(0)) + min2(z_neg(j), T(0))) *
							C_.row(j);
	}

	iterative_solve_with_permut_fact_QPALM( //
			VectorViewMut<T>{from_eigen,rhs},
			VectorViewMut<T>{from_eigen,dw},
			VectorViewMut<T>{from_eigen,err},
			ldl,
			eps,
			isize(3),
			qp_scaled,
			current_bijection_map,
			dim,
			n_eq,
			n_c,
			n_in,
			mu_,
			rho,
			VERBOSE);

	dx.to_eigen() = dw.topRows(dim);
}

template <typename T,typename Preconditioner = qp::preconditioner::IdentityPrecond>
T correction_guess_QPALM(
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> mu_,
		T rho,
		T eps_int,
		isize dim,
		isize n_eq,
		isize n_in,
		isize max_iter_in,
		isize& n_tot,
		VectorViewMut<T> residual_in_y_,
		VectorViewMut<T> z_pos_,
		VectorViewMut<T> z_neg_,
		VectorViewMut<T> dual_for_eq_,
		VectorViewMut<T> Hdx_,
		VectorViewMut<T> Adx_,
		VectorViewMut<T> Cdx_,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		VectorViewMut<T> dw_aug_,
		T& correction_guess_rhs_g,
		T max_rank_update,
		T max_rank_update_fraction,
		isize& nb_enter,
		isize& nb_leave,
		isize& no_change_in_active_constraints,
		Preconditioner& precond,
		isize& prev_iter,
		bool& VERBOSE) {

	/* NB
	* dual_for_eq_ = Hx + rho (x-xe) + AT( mu(Ax-b) + ye )
	* residual_in_y_ = Ax-b + ye/mu
	* z_pos_ = Cx-u + ze / mu
	* z_neg_ = Cx-l + ze / mu
	*/
	auto mu = mu_.to_eigen();
	auto residual_in_y = residual_in_y_.to_eigen();
	auto z_pos = z_pos_.to_eigen();
	auto z_neg = z_neg_.to_eigen();
	auto dual_for_eq = dual_for_eq_.to_eigen();
	auto Hdx = Hdx_.to_eigen();
	auto Adx = Adx_.to_eigen();
	auto Cdx = Cdx_.to_eigen();
	auto dw_aug = dw_aug_.to_eigen();
	
	T err_in = T(1.e6);

	for (i64 iter = 0; iter <= max_iter_in; ++iter) {
		if (iter == max_iter_in+prev_iter) {
			n_tot += max_iter_in;
			no_change_in_active_constraints = isize(0);
			prev_iter = iter ;
			break;
		}
		dw_aug.topRows(dim).setZero();
		
		qp::detail::newton_step_QPALM<T>(
				qp_scaled,
				x.as_const(),
				xe,
				ye,
				ze,
				VectorViewMut<T>{from_eigen, dw_aug.topRows(dim)},
				mu_,
				rho,
				eps_int,
				dim,
				n_eq,
				n_in,
				z_pos_.as_const(),
				z_neg_.as_const(),
				dual_for_eq_.as_const(),
				l_active_set_n_u,
				l_active_set_n_l,
				active_inequalities,
				ldl,
				current_bijection_map,
				n_c,
				max_rank_update,
				max_rank_update_fraction,
				nb_enter,
				nb_leave,
				VERBOSE);
		T alpha_step = T(1);

		Hdx = (qp_scaled.H).to_eigen() * dw_aug.topRows(dim);
		Adx = (qp_scaled.A).to_eigen() * dw_aug.topRows(dim);
		Cdx = (qp_scaled.C).to_eigen() * dw_aug.topRows(dim);
		if (n_in > isize(0)) {
			alpha_step = qp::line_search_qpalm::correction_guess_LS_QPALM(
					VectorView<T>{from_eigen,Hdx},
					VectorView<T>{from_eigen, dw_aug.topRows(dim)},
					(qp_scaled.g),
					VectorView<T>{from_eigen,Adx},
					VectorView<T>{from_eigen,Cdx},
					residual_in_y_.as_const(),
					z_pos_.as_const(),
					z_neg_.as_const(),
					x.as_const(),
					xe,
					ye,
					ze,
					mu_,
					rho,
					n_in,
					n_eq);
		}

		x.to_eigen() += alpha_step * dw_aug.topRows(dim);
		z_pos += alpha_step * Cdx;
		z_neg += alpha_step * Cdx;
		residual_in_y += alpha_step * Adx;
		y.to_eigen().array() = mu.topRows(n_eq).array() * residual_in_y.array();
		dual_for_eq.array() += alpha_step * (qp_scaled.A.to_eigen().transpose() * (mu.topRows(n_eq).array() * Adx.array()).matrix()).array() +
							   (rho * dw_aug.topRows(dim) + Hdx).array();
		for (isize j = 0; j < n_in; ++j) {
			z(j) = mu(n_eq + j) * (max2(z_pos(j), T(0)) + min2(z_neg(j), T(0)));
		}

		

		Hdx = (qp_scaled.H).to_eigen() * x.to_eigen();
		precond.unscale_dual_residual_in_place(
							VectorViewMut<T>{from_eigen,Hdx}) ;
		T rhs_c = max2(correction_guess_rhs_g, infty_norm(Hdx));

		dw_aug.topRows(dim) = (qp_scaled.A.to_eigen().transpose()) * (y.to_eigen());
		precond.unscale_dual_residual_in_place(
							VectorViewMut<T>{from_eigen,dw_aug.topRows(dim)}) ;
		rhs_c = max2(rhs_c, infty_norm(dw_aug.topRows(dim)));
		Hdx += dw_aug.topRows(dim);

		dw_aug.topRows(dim) = (qp_scaled.C.to_eigen().transpose()) * (z.to_eigen());
		precond.unscale_dual_residual_in_place(
							VectorViewMut<T>{from_eigen,dw_aug.topRows(dim)}) ;
		rhs_c = max2(rhs_c, infty_norm(dw_aug.topRows(dim)));
		Hdx += dw_aug.topRows(dim);

		dw_aug.topRows(dim) = (qp_scaled.g).to_eigen() + rho * (x.to_eigen() - xe.to_eigen()) ; 
		precond.unscale_dual_residual_in_place(
							VectorViewMut<T>{from_eigen,dw_aug.topRows(dim)}) ;
		Hdx += dw_aug.topRows(dim);
		
		err_in = infty_norm(Hdx);
		if (VERBOSE){
			std::cout << "---it in " << iter << " projection norm " << err_in << " alpha " << alpha_step << std::endl;
		}
		if (err_in <= eps_int * (1 + rhs_c) || no_change_in_active_constraints == isize(3)) {
			n_tot += iter + 1;
			no_change_in_active_constraints = isize(0);
			prev_iter = iter;
			break;
		} else if (nb_enter + nb_leave == 0){
			no_change_in_active_constraints +=isize(1);
		} else if (nb_enter + nb_leave > 0){
			no_change_in_active_constraints = isize(0);
		}
	}

	return err_in;
}

template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpalmSolveStats QPALMSolve( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		T max_rank_update,
		T max_rank_update_fraction,
		Preconditioner precond = Preconditioner{},
		bool VERBOSE = false) {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_c = 0; 

	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;
	isize n_max = dim + n_eq + n_in ;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6); // in QPALM 1.e-7
	T eta_ext = T(1);
	T eta_in = T(1);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_h_scaled, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_h_ws, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_g_scaled, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_a_scaled, Uninit, Mat(n_eq, dim), LDLT_CACHELINE_BYTES, T),
			(_c_scaled, Uninit, Mat(n_in, dim), LDLT_CACHELINE_BYTES, T),
			(_b_scaled, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_u_scaled, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_l_scaled, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_residual_scaled, Init, Vec(n_max + 2*n_eq + 3*n_in), LDLT_CACHELINE_BYTES, T),
			(_y, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_z, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(xe_, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_eq, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_in, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),

			(_kkt,
	     Uninit,
	     Mat(dim + n_eq + n_c, dim + n_eq + n_c),
	     LDLT_CACHELINE_BYTES,
	     T),

			(current_bijection_map_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, isize),
			(mu_, Init, Vec(n_eq+n_in), LDLT_CACHELINE_BYTES, T),

			(_dw_aug, Init, Vec(n_max), LDLT_CACHELINE_BYTES, T),
			(d_dual_for_eq_, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_cdx, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(d_primal_residual_eq_, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(l_active_set_n_u_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(l_active_set_n_l_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(active_inequalities_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool));

	auto residual_scaled = _residual_scaled.to_eigen();

	auto ye = _y.to_eigen();
	auto ze = _z.to_eigen();
	auto xe = xe_.to_eigen();
	auto diag_diff_in = _diag_diff_in.to_eigen();
	auto diag_diff_eq = _diag_diff_eq.to_eigen();
	auto dual_residual_scaled = residual_scaled.topRows(dim);
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim, n_eq);
	auto primal_residual_in_scaled_u = residual_scaled.middleRows(dim + n_eq,n_in);
	auto primal_residual_in_scaled_l = residual_scaled.middleRows(n_max,n_in);
	auto primal_residual_eq_scaled_new = residual_scaled.middleRows(n_max + n_in, n_eq);
	auto primal_residual_in_scaled_l_new = residual_scaled.middleRows(n_max + n_in + n_eq, n_in);
	auto primal_residual_in_scaled_l_old = residual_scaled.middleRows(n_max + 2*n_in + n_eq,n_in);
	auto primal_residual_eq_scaled_old = residual_scaled.tail(n_eq);
	auto mu = mu_.to_eigen();
	auto d_dual_for_eq = d_dual_for_eq_.to_eigen();
	auto Cdx = _cdx.to_eigen();
	auto d_primal_residual_eq = d_primal_residual_eq_.to_eigen();
	auto l_active_set_n_u = l_active_set_n_u_.to_eigen();
	auto l_active_set_n_l = l_active_set_n_l_.to_eigen();
	auto active_inequalities = active_inequalities_.to_eigen();
	auto dw_aug = _dw_aug.to_eigen();

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);
	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());
	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);
	T theta(0.25);
	T Delta(100);
	T sigmaMax(1.e6); // in QPALM 1.e9
	isize nb_enter(0);
	isize nb_leave(0);
	isize no_change_in_active_constraints(0);
	isize mu_update(0);
	isize prev_iter(0);

	auto current_bijection_map = current_bijection_map_.to_eigen();
	for (isize i = 0; i < n_in; i++) {
		current_bijection_map(i) = i;
	}

	auto H_copy = _h_scaled.to_eigen();
	auto kkt = _kkt.to_eigen();
	auto H_ws = _h_ws.to_eigen();
	auto q_copy = _g_scaled.to_eigen();
	auto A_copy = _a_scaled.to_eigen();
	auto C_copy = _c_scaled.to_eigen();
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

	auto qp_scaled = qp::QpViewBoxMut<T>{
			{from_eigen, H_copy},
			{from_eigen, q_copy},
			{from_eigen, A_copy},
			{from_eigen, b_copy},
			{from_eigen, C_copy},
			{from_eigen, u_copy},
			{from_eigen, l_copy}};
	precond.scale_qp_in_place(qp_scaled);

	//T correction_guess_rhs_g = infty_norm((qp_scaled.g).to_eigen());

	H_ws = H_copy;
	for (isize i = 0; i < dim; ++i) {
		H_ws(i, i) += rho;
	}
	//ldlt::Ldlt<T> ldl_ws{decompose, H_ws};
	kkt.topLeftCorner(dim, dim) = H_ws;
	kkt.block(0, dim, dim, n_eq) = qp_scaled.A.to_eigen().transpose();
	kkt.block(dim, 0, n_eq, dim) = qp_scaled.A.to_eigen();
	kkt.bottomRightCorner(n_eq + n_c, n_eq + n_c).setZero();

	//x.to_eigen() = -(qp_scaled.g).to_eigen();
	//ldl_ws.solve_in_place(x.to_eigen());

	primal_residual_eq_scaled = qp_scaled.A.to_eigen() * x.to_eigen() - qp_scaled.b.to_eigen() ;
	primal_residual_in_scaled_u = qp_scaled.C.to_eigen() *x.to_eigen() ; 
	primal_residual_in_scaled_l =
			((primal_residual_in_scaled_u - qp.u.to_eigen()).array() > T(0))
					.select(
							primal_residual_in_scaled_u - qp.u.to_eigen(),
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_in)) +
			((primal_residual_in_scaled_u - qp.l.to_eigen()).array() < T(0))
					.select(
							primal_residual_in_scaled_u - qp.l.to_eigen(),
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_in));


	T mu_init = max2( T(1e-4) , min2(  T(20) * max2( T(1) , T(std::abs(0.5 * x.to_eigen().dot(H_copy*x.to_eigen()) + q_copy.dot(x.to_eigen()))) )  / max2( T(1) , max2(
			infty_norm(primal_residual_in_scaled_l),
			infty_norm(primal_residual_eq_scaled)) )     , T(1e4) ) ) ; 
	T divide_mu_init = - T(1) / mu_init ; 
	for (isize i = 0; i < n_eq; ++i) {
		
		mu(i) = mu_init ; 
		kkt(dim + i, dim + i) = divide_mu_init;
	}
	for (isize i = 0; i < n_in; ++i) {
		mu(n_eq+i) = mu_init ;
	}
	ldlt::Ldlt<T> ldl{decompose, kkt};
	xe = x.to_eigen();
	ye = y.to_eigen();
	ze = z.to_eigen();
	for (i64 iter = 0; iter <= max_iter; ++iter) {
		n_ext += 1;
		if (iter == max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());
		primal_residual_in_scaled_l_old = primal_residual_in_scaled_l ; 
		primal_residual_eq_scaled_old = primal_residual_eq_scaled ; 
		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				{from_eigen, dual_residual_scaled},
				{from_eigen, dw_aug},
				qp_scaled.as_const(),
				precond,
				x.as_const(),
				y.as_const(),
				z.as_const());

		if (VERBOSE){
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "eta_ext : " << eta_ext
								<< " bcl_eta_in : " << eta_in << " rho : " << rho
								<< std::endl;
		}
		bool is_primal_feasible =
				primal_feasibility_lhs <=
				(eps_abs +
		     eps_rel *
		         max2(
								 max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 max2(
										 max2(
												 primal_feasibility_rhs_1_eq,
												 primal_feasibility_rhs_1_in_u),
										 primal_feasibility_rhs_1_in_l)

										 ));

		bool is_dual_feasible =
				dual_feasibility_lhs <=
				(eps_abs +
		     eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 dual_feasibility_rhs_2)));

		if (is_primal_feasible) {

			if (is_dual_feasible) {
				{
					//LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
					precond.unscale_primal_in_place(x);
					precond.unscale_dual_in_place_eq(y);
					precond.unscale_dual_in_place_in(z);


				}
				return {double(n_ext), double(n_mu_updates), double(n_tot)};
			}
		}
		/* NB 
		* primal_residual_eq_scaled = scaled(Ax-b)
		* dual_residual_scaled = scaled(Hx+g+ATy+CTz)
		*/
		primal_residual_eq_scaled.array() += (ye.array() / mu.topRows(n_eq).array());
		{
		auto Cx_ze_mu = qp_scaled.C.to_eigen() * x.to_eigen() + (ze.array() / mu.tail(n_in).array()).matrix() ;  
		
		dual_residual_scaled +=
				(rho * (x.to_eigen() - xe)
				-(qp_scaled.A).to_eigen().transpose() * y.to_eigen()
				-(qp_scaled.C).to_eigen().transpose() * z.to_eigen()
				+(qp_scaled.A).to_eigen().transpose() * (mu.topRows(n_eq).array() * primal_residual_eq_scaled.array()).matrix()   ) ;
		
		primal_residual_in_scaled_u = Cx_ze_mu - qp_scaled.u.to_eigen() ;
		primal_residual_in_scaled_l = Cx_ze_mu - qp_scaled.l.to_eigen() ;
		}
		T err_in = qp::detail::correction_guess_QPALM(
					VectorView<T>{from_eigen, xe},
					VectorView<T>{from_eigen, ye},
					VectorView<T>{from_eigen, ze},
					x,
					y,
					z,
					qp_scaled.as_const(),
					mu_.as_const(),
					rho,
					eta_in, 
					dim,
					n_eq,
					n_in,
					max_iter_in,
					n_tot,
					VectorViewMut<T>{from_eigen, primal_residual_eq_scaled},
					VectorViewMut<T>{from_eigen, primal_residual_in_scaled_u},
					VectorViewMut<T>{from_eigen, primal_residual_in_scaled_l},
					VectorViewMut<T>{from_eigen, dual_residual_scaled},
					VectorViewMut<T>{from_eigen, d_dual_for_eq},
					VectorViewMut<T>{from_eigen, d_primal_residual_eq},
					VectorViewMut<T>{from_eigen, Cdx},
					VectorViewMut<bool>{from_eigen, l_active_set_n_u},
					VectorViewMut<bool>{from_eigen, l_active_set_n_l},
					VectorViewMut<bool>{from_eigen, active_inequalities},
					ldl,
					VectorViewMut<isize>{from_eigen, current_bijection_map},
					n_c,
					VectorViewMut<T>{from_eigen, dw_aug},
					dual_feasibility_rhs_2,
					max_rank_update,
					max_rank_update_fraction,
					nb_enter,
					nb_leave,
					no_change_in_active_constraints,
					precond,
					prev_iter,
					VERBOSE);
		if (VERBOSE){
			std::cout << "primal_feasibility_lhs " << primal_feasibility_lhs
								<< " error from inner loop : " << err_in 
								<< std::endl;
		}
		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		qp::detail::global_primal_residual(
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled_new},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l_new},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());
		qp::detail::QPALM_update_fact(
				primal_feasibility_lhs_new,
				eta_ext, 
				eta_in,
				eps_abs,
				VectorViewMut<T>{from_eigen, xe},
				VectorViewMut<T>{from_eigen, ye},
				VectorViewMut<T>{from_eigen, ze},
				x,
				y,
				z,
				VERBOSE);
		
		qp::detail::QPALM_mu_update(
		      primal_feasibility_lhs,
		      VectorView<T>{from_eigen,primal_residual_eq_scaled_new},
		      VectorView<T>{from_eigen,primal_residual_in_scaled_l_new},
		      VectorView<T>{from_eigen,primal_residual_eq_scaled_old},
		      VectorView<T>{from_eigen,primal_residual_in_scaled_l_old},
			  VectorView<isize>{from_eigen, current_bijection_map},
		      n_mu_updates,
		      VectorViewMut<T>{from_eigen,mu},
		      dim,
		      n_eq,
			  n_in,
		      n_c,
		      ldl,
		      qp_scaled.as_const(),
		      rho,
		      theta,
		      sigmaMax,
		      Delta,
			  mu_update,
			  max_rank_update,
			  max_rank_update_fraction,
			  VERBOSE
		);
		
	}

	return {double(n_ext), double(n_mu_updates), double(n_tot)};
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_QPALM_SOLVER_HPP_HDWGZKCLS */