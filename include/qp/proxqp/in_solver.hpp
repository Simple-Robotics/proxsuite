#ifndef INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "qp/utils.hpp"
#include <qp/QPData.hpp>
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/proxqp/line_search.hpp"
#include "qp/precond/identity.hpp"
#include <cmath>
#include <type_traits>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

struct QpSolveStats {
	isize n_ext;
	isize n_mu_updates;
	isize n_tot;
	isize activeSetChange;
	isize SolveLS;
	isize deletion;
	isize adding;
	/*
	double equilibration_tmp ;
	double fact_tmp;
	double ws_tmp;
	double residuals_tmp;
	double IG_tmp;
	double CG_tmp;
	double BCL_tmp;
	double cold_restart_tmp;
	*/
};

#define LDLT_DEDUCE_RET(...)                                                   \
	noexcept(noexcept(__VA_ARGS__))                                              \
			->typename std::remove_const<decltype(__VA_ARGS__)>::type {              \
		return __VA_ARGS__;                                                        \
	}                                                                            \
	static_assert(true, ".")

template <typename T>
void iterative_residual(
		qp::Qpdata<T>& qpdata,
		qp::QpViewBox<T> qp_scaled,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		isize inner_pb_dim,
		T mu_eq,
		T mu_in,
		T rho) {
	
	qpdata._err.head(inner_pb_dim) = qpdata._rhs.head(inner_pb_dim);

	qpdata._err.head(dim).noalias() -=
			(qp_scaled.H).to_eigen() * qpdata._dw_aug.head(dim) + rho * qpdata._dw_aug.head(dim) +
			(qp_scaled.A).to_eigen().transpose() * qpdata._dw_aug.segment(dim, n_eq);

	for (isize i = 0; i < n_in; i++) {
		isize j = qpdata._current_bijection_map(i);
		if (j < n_c) {
			qpdata._err.head(dim).noalias() -= qpdata._dw_aug(dim + n_eq + j) * qp_scaled.C.to_eigen().row(i);
			qpdata._err(dim + n_eq + j) -=
					(qp_scaled.C.to_eigen().row(i)).dot(qpdata._dw_aug.head(dim)) -
					qpdata._dw_aug(dim + n_eq + j) / mu_in;
		}
	}

	qpdata._err.segment(dim, n_eq).noalias() -= (qp_scaled.A).to_eigen() * qpdata._dw_aug.head(dim) -
	                              qpdata._dw_aug.segment(dim, n_eq) / mu_eq;

}

template <typename T>
void iterative_solve_with_permut_fact_new( //
		qp::Qpdata<T>& qpdata,
		ldlt::Ldlt<T> const& ldl,
		T eps,
		isize max_it,
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		isize inner_pb_dim,
		T mu_eq,
		T mu_in,
		T rho,
		bool& VERBOSE) {

	i32 it = 0;
	qpdata._dw_aug.head(inner_pb_dim) = qpdata._rhs.head(inner_pb_dim);
	ldl.solve_in_place(qpdata._dw_aug.head(inner_pb_dim));

	auto compute_iterative_residual = [&] {
		qp::detail::iterative_residual<T>(
				qpdata,
				qp_scaled,
				dim,
				n_eq,
				n_c,
				n_in,
				inner_pb_dim,
				mu_eq,
				mu_in,
				rho);
	};

	compute_iterative_residual();
	++it;
	if (VERBOSE){
		std::cout << "infty_norm(res) " << qp::infty_norm( qpdata._err.head(inner_pb_dim)) << std::endl;
	}
	while (infty_norm( qpdata._err.head(inner_pb_dim)) >= eps) {
		if (it >= max_it) {
			break;
		}
		++it;
		ldl.solve_in_place( qpdata._err.head(inner_pb_dim));
		qpdata._dw_aug.head(inner_pb_dim) +=  qpdata._err.head(inner_pb_dim);

		qpdata._err.head(inner_pb_dim).setZero();
		compute_iterative_residual();
		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(qpdata._err.head(inner_pb_dim)) << std::endl;
		}
	}
}

template <typename T>
void BCL_update_fact(
		qp::Qpdata<T>& qpdata,
		T& primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		isize& n_mu_updates,
		T& bcl_mu_in,
		T& bcl_mu_eq,
		isize dim,
		isize n_eq,
		isize n_c,
		ldlt::Ldlt<T>& ldl,
		T& beta,
		T& exponent,
		T& bcl_eta_ext_init,
		T& cold_reset_bcl_mu_max,
		bool& VERBOSE) {

	if (primal_feasibility_lhs <= bcl_eta_ext) {
		if (VERBOSE){
			std::cout << "good step" << std::endl;
		}
		bcl_eta_ext = bcl_eta_ext / pow(bcl_mu_in, beta);
		bcl_eta_in = max2(bcl_eta_in / bcl_mu_in, eps_abs);
	} else {
		if (VERBOSE){
			std::cout << "bad step" << std::endl;
		}
		qpdata._y = qpdata._ye;
		qpdata._z = qpdata._ze;
		T new_bcl_mu_in(min2(bcl_mu_in * 10, cold_reset_bcl_mu_max));
		T new_bcl_mu_eq(min2(bcl_mu_eq * (10), cold_reset_bcl_mu_max * 100));
		if (bcl_mu_in != new_bcl_mu_in || bcl_mu_eq != new_bcl_mu_eq) {
			{ ++n_mu_updates; }
		}
		qp::detail::mu_update(
				qpdata,
				bcl_mu_eq,
				new_bcl_mu_eq,
				bcl_mu_in,
				new_bcl_mu_in,
				dim,
				n_eq,
				n_c,
				ldl);
		bcl_mu_eq = new_bcl_mu_eq;
		bcl_mu_in = new_bcl_mu_in;
		bcl_eta_ext = bcl_eta_ext_init/ pow(bcl_mu_in, exponent);
		bcl_eta_in = max2(1 / bcl_mu_in, eps_abs);
	}
}

template <typename T>
auto saddle_point(
		qp::QpViewBox<T> qp_scaled,
		qp::Qpdata<T>& qpdata,
		T mu_in,
		isize n_in,
		T& zero) -> T {

	auto C_ = qp_scaled.C.to_eigen();
	qpdata._primal_residual_in_scaled_u -= qpdata._z / mu_in;
	qpdata._primal_residual_in_scaled_l -= qpdata._z / mu_in;
	T prim_eq_e = infty_norm(qpdata._primal_residual_eq_scaled);

	qpdata._dual_residual_scaled.noalias() += C_.transpose() * qpdata._z ;

	T dual_e = infty_norm(qpdata._dual_residual_scaled);
	T err = max2(prim_eq_e, dual_e);

	T prim_in_e(0);

	for (isize i = 0; i < n_in; ++i) {
		using std::fabs;

		if (qpdata._z(i) > 0) {
			prim_in_e = max2(prim_in_e, fabs(qpdata._primal_residual_in_scaled_u(i)));
		} else if (qpdata._z(i) < 0) {
			prim_in_e = max2(prim_in_e, fabs(qpdata._primal_residual_in_scaled_l(i)));
		} else {
			prim_in_e = max2(prim_in_e, max2(qpdata._primal_residual_in_scaled_u(i), zero));
			prim_in_e = max2(prim_in_e, fabs(min2(qpdata._primal_residual_in_scaled_l(i), zero)));
		}
	}

	err = max2(err, prim_in_e);
	return err;
}

template <typename T>
void newton_step_fact(
		qp::Qpdata<T>& qpdata,
		qp::QpViewBox<T> qp_scaled,
		T mu_eq,
		T mu_in,
		T rho,
		T eps,
		isize dim,
		isize n_eq,
		isize n_in,
		ldlt::Ldlt<T>& ldl,
		isize& n_c,
		isize& deletion,
		isize& adding,
		bool& VERBOSE,
		T& zero) {
	
	qpdata._l_active_set_n_u = (qpdata._primal_residual_in_scaled_u.array() > 0).matrix();
	qpdata._l_active_set_n_l = (qpdata._primal_residual_in_scaled_l.array() < 0).matrix();

	qpdata._active_inequalities =qpdata._l_active_set_n_u || qpdata._l_active_set_n_l;

	isize num_active_inequalities = qpdata._active_inequalities.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	qpdata._err.head(inner_pb_dim).setZero();
	qpdata._rhs.head(inner_pb_dim) = qpdata._err.head(inner_pb_dim);
	qpdata._rhs.head(dim) = -qpdata._dual_residual_scaled;

	{
		//LDLT_DECL_SCOPE_TIMER("in solver", "activeSetChange", T);
		qp::line_search::active_set_change_new(
				qpdata,
				n_c,
				n_in,
				dim,
				n_eq,
				ldl,
				qp_scaled,
				mu_in,
				mu_eq,
				rho,
				deletion,
				adding);
	}

	{
	//LDLT_DECL_SCOPE_TIMER("in solver", "SolveLS", T);

	detail::iterative_solve_with_permut_fact_new( //
			qpdata,
			ldl,
			eps,
			3,
			qp_scaled,
			VectorViewMut<isize>{from_eigen,qpdata._current_bijection_map},
			dim,
			n_eq,
			n_c,
			n_in,
			inner_pb_dim,
			mu_eq,
			mu_in,
			rho,
			VERBOSE);
	}

}

template <typename T, typename Preconditioner>
auto initial_guess_fact(
		qp::Qpdata<T>& qpdata,
		qp::QpViewBox<T> qp_scaled,
		qp::QpViewBox<T> qp,
		T mu_in,
		T mu_eq,
		T rho,
		T eps_int,
		Preconditioner& precond,
		isize dim,
		isize n_eq,
		isize n_in,
		ldlt::Ldlt<T>& ldl,
		isize& n_c,
		T R,
		isize& deletion,
		isize& adding,
		bool& VERBOSE,
		T& zero) -> T {

	auto C_ = qp_scaled.C.to_eigen();
	
	qpdata._primal_residual_in_scaled_l = qpdata._primal_residual_in_scaled_u;
	qpdata._primal_residual_in_scaled_u -= qp.u.to_eigen();
	qpdata._primal_residual_in_scaled_l -= qp.l.to_eigen();

	precond.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpdata._ze});
	qpdata._primal_residual_in_scaled_u += qpdata._ze / mu_in;
	qpdata._primal_residual_in_scaled_l += qpdata._ze / mu_in;

	qpdata._l_active_set_n_u.array() = (qpdata._primal_residual_in_scaled_u .array() >= 0);
	qpdata._l_active_set_n_l.array() = (qpdata._primal_residual_in_scaled_l.array() <= 0);

	qpdata._active_inequalities = qpdata._l_active_set_n_u || qpdata._l_active_set_n_l;

	qpdata._primal_residual_in_scaled_u -= qpdata._ze / mu_in;
	qpdata._primal_residual_in_scaled_l -= qpdata._ze / mu_in;

	precond.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpdata._primal_residual_in_scaled_u});
	precond.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpdata._primal_residual_in_scaled_l});
	precond.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpdata._ze});
	// rescale value
	isize num_active_inequalities = qpdata._active_inequalities.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	//LDLT_DECL_SCOPE_TIMER("in solver", "activeSetChange", T);

	qp::line_search::active_set_change_new(
			qpdata,
			n_c,
			n_in,
			dim,
			n_eq,
			ldl,
			qp_scaled,
			mu_in,
			mu_eq,
			rho,
			deletion,
			adding);

	qpdata._err.head(inner_pb_dim).setZero();

	for (isize i = 0; i < n_in; i++) {
		isize j = qpdata._current_bijection_map(i);
		if (j < n_c) {
			if (qpdata._l_active_set_n_u(i)) {
				qpdata._rhs(j + dim + n_eq) = -qpdata._primal_residual_in_scaled_u(i);
			} else if (qpdata._l_active_set_n_l(i)) {
				qpdata._rhs(j + dim + n_eq) = -qpdata._primal_residual_in_scaled_l(i);
			}
		} else {
			qpdata._rhs.head(dim).noalias() += qpdata._z(i) * C_.row(i);
		}
	}
	
	qpdata._rhs.head(dim) = -qpdata._dual_residual_scaled;
	qpdata._rhs.segment(dim, n_eq) = -qpdata._primal_residual_eq_scaled;

	{
	//LDLT_DECL_SCOPE_TIMER("in solver", "SolveLS", T);
	detail::iterative_solve_with_permut_fact_new( //
			qpdata,
			ldl,
			eps_int,
			3,
			qp_scaled,
			VectorViewMut<isize>{from_eigen, qpdata._current_bijection_map},
			dim,
			n_eq,
			n_c,
			n_in,
			inner_pb_dim,
			mu_eq,
			mu_in,
			rho,
			VERBOSE);
	}

	qpdata._d_dual_for_eq = qpdata._rhs.topRows(dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz

	for (isize i = 0; i < n_in; i++) {
		isize j = qpdata._current_bijection_map(i);
		if (j < n_c) {
			if (qpdata._l_active_set_n_u(i)) {
				qpdata._d_dual_for_eq.noalias() -= qpdata._dw_aug(j + dim + n_eq) * C_.row(i);
			} else if (qpdata._l_active_set_n_l(i)) {
				qpdata._d_dual_for_eq.noalias() -= qpdata._dw_aug(j + dim + n_eq) * C_.row(i);
			}
		}
	}

	// use active_part_z as a temporary variable to permut back dw_aug newton step
	for (isize j = 0; j < n_in; ++j) {
		isize i = qpdata._current_bijection_map(j);
		if (i < n_c) {
			//dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
			//cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;
			
			qpdata._active_part_z(j) = qpdata._dw_aug(dim + n_eq + i);
			qpdata._Cdx(j) = qpdata._rhs(i + dim + n_eq) + qpdata._dw_aug(dim + n_eq + i) / mu_in;
			
		} else {
			//dw_aug_(j + dim + n_eq) = -z_(j);
			qpdata._active_part_z(j) = -qpdata._z(j);
			qpdata._Cdx(j) = C_.row(j).dot(qpdata._dw_aug.head(dim));
		}
	}
	qpdata._dw_aug.tail(n_in) = qpdata._active_part_z ;

	qpdata._primal_residual_in_scaled_u += qpdata._ze / mu_in;
	qpdata._primal_residual_in_scaled_l += qpdata._ze / mu_in;

	qpdata._d_primal_residual_eq = qpdata._rhs.segment(dim, n_eq); // By definition of linear system solution

	qpdata._dual_residual_scaled.noalias() -= qpdata._CTz;

	T alpha_step = qp::line_search::initial_guess_LS(
			qpdata,
			qp_scaled.C,
			mu_eq,
			mu_in,
			rho,
			dim,
			n_eq,
			n_in,
			R);
	
	if (VERBOSE){
		std::cout << "alpha from initial guess " << alpha_step << std::endl;
	}
	
	qpdata._primal_residual_in_scaled_u += (alpha_step * qpdata._Cdx);
	qpdata._primal_residual_in_scaled_l += (alpha_step * qpdata._Cdx);
	qpdata._l_active_set_n_u = (qpdata._primal_residual_in_scaled_u.array() >= 0).matrix();
	qpdata._l_active_set_n_l = (qpdata._primal_residual_in_scaled_l.array() <= 0).matrix();
	qpdata._active_inequalities = qpdata._l_active_set_n_u || qpdata._l_active_set_n_l;

	qpdata._x.noalias() += alpha_step * qpdata._dw_aug.head(dim);
	qpdata._y.noalias()  += alpha_step * qpdata._dw_aug.segment(dim, n_eq);

	
	qpdata._active_part_z = qpdata._z + alpha_step * qpdata._dw_aug.tail(n_in) ;

	qpdata._residual_in_z_u_plus_alpha = (qpdata._active_part_z.array() > 0).select(qpdata._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in));
	qpdata._residual_in_z_l_plus_alpha = (qpdata._active_part_z.array() < 0).select(qpdata._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in));

	qpdata._z = (qpdata._l_active_set_n_u).select(qpdata._residual_in_z_u_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)) +
				   (qpdata._l_active_set_n_l).select(qpdata._residual_in_z_l_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)) +
				   (!qpdata._l_active_set_n_l.array() && !qpdata._l_active_set_n_u.array()).select(qpdata._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(n_in)) ;
	
	qpdata._primal_residual_eq_scaled += alpha_step * qpdata._d_primal_residual_eq;
	qpdata._dual_residual_scaled += alpha_step * qpdata._d_dual_for_eq;

	T err = detail::saddle_point(
			qp_scaled,
			qpdata,
			mu_in,
			n_in,
			zero);
	
	return err;
}

template <typename T>
auto correction_guess(
		qp::Qpdata<T>& qpdata,
		qp::QpViewBox<T> qp_scaled,
		T mu_in,
		T mu_eq,
		T rho,
		T eps_int,
		isize dim,
		isize n_eq,
		isize n_in,
		isize max_iter_in,
		isize& n_tot,
		ldlt::Ldlt<T>& ldl,
		isize& n_c,
		T& correction_guess_rhs_g,
		isize& deletion,
		isize& adding,
		bool& VERBOSE,
		T& zero) -> T {

	auto g_ = (qp_scaled.g).to_eigen() ;

	T err_in(0.);

	qpdata._dual_residual_scaled.noalias() = qpdata._Hx + qpdata._ATy + g_ + rho * (qpdata._x - qpdata._xe) + mu_in * qp_scaled.C.to_eigen().transpose() * (qp::detail::positive_part(qpdata._primal_residual_in_scaled_u) + qp::detail::negative_part(qpdata._primal_residual_in_scaled_l));  // used for newton step at first iteration

	for (i64 iter = 0; iter <= max_iter_in; ++iter) {

		if (iter == max_iter_in) {
			n_tot += max_iter_in;
			break;
		}

		qpdata._dw_aug.head(dim).setZero();
		qp::detail::newton_step_fact(
				qpdata,
				qp_scaled,
				mu_eq,
				mu_in,
				rho,
				eps_int,
				dim,
				n_eq,
				n_in,
				ldl,
				n_c,
				deletion,
				adding,
				VERBOSE,
				zero);

		T alpha_step(1);
		qpdata._d_dual_for_eq.noalias() = (qp_scaled.H).to_eigen() * qpdata._dw_aug.head(dim);
		qpdata._d_primal_residual_eq.noalias() = qpdata._dw_aug.middleRows(dim,n_eq) / mu_eq; // by definition Adx = dy / mu
		qpdata._Cdx.noalias() = (qp_scaled.C).to_eigen() * qpdata._dw_aug.head(dim);
		if (n_in > isize(0)) {

			alpha_step = qp::line_search::correction_guess_LS(
					qpdata,
					qp_scaled.g,
					mu_eq,
					mu_in,
					rho,
					dim,
					n_in
			);
		}

		if (infty_norm(alpha_step * qpdata._dw_aug.head(dim)) < 1.E-11) {
			n_tot += iter + 1;
			break;
		}

		qpdata._x.noalias() += alpha_step * qpdata._dw_aug.head(dim);
		qpdata._primal_residual_in_scaled_u += alpha_step * qpdata._Cdx;
		qpdata._primal_residual_in_scaled_l += alpha_step * qpdata._Cdx;
		qpdata._primal_residual_eq_scaled.noalias() += alpha_step * qpdata._d_primal_residual_eq;
		qpdata._y.noalias() = mu_eq * qpdata._primal_residual_eq_scaled;

		qpdata._Hx.noalias() += alpha_step * qpdata._d_dual_for_eq ; // stores Hx
		qpdata._ATy.noalias() += (alpha_step * mu_eq) * (qp_scaled.A).to_eigen().transpose() * qpdata._d_primal_residual_eq ; // stores ATy
		
		qpdata._z = mu_in * (qp::detail::positive_part(qpdata._primal_residual_in_scaled_u) + qp::detail::negative_part(qpdata._primal_residual_in_scaled_l)) ; 
		T rhs_c = max2(correction_guess_rhs_g, infty_norm( qpdata._d_dual_for_eq));
		rhs_c = max2(rhs_c, infty_norm(qpdata._ATy));
		qpdata._dual_residual_scaled.noalias() = qp_scaled.C.to_eigen().transpose() * qpdata._z ; 
		rhs_c = max2(rhs_c, infty_norm(qpdata._dual_residual_scaled));
		qpdata._dual_residual_scaled.noalias() += qpdata._Hx + g_ + qpdata._ATy + rho * (qpdata._x - qpdata._xe);

		err_in = infty_norm(qpdata._dual_residual_scaled);
		if (VERBOSE){
			std::cout << "---it in " << iter << " projection norm " << err_in
							<< " alpha " << alpha_step << std::endl;
		}
		if (err_in <= eps_int * (1 + rhs_c)) {
			n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}


template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpSolveStats qpSolve( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp,
		qp::Qpdata<T>& qpdata,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		T err_IG,
		T beta,
		T R,
		Preconditioner & precond = Preconditioner{},
		bool VERBOSE = false) {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_tot_max = dim + n_eq + n_in;
	isize n_c = 0;

	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;

	constexpr T machine_eps = std::numeric_limits<T>::epsilon();
	T rho = 1e-6;
	T bcl_mu_eq = 1e3;
	T bcl_mu_in = 1e1;
	T exponent(0.1);
	T bcl_eta_ext_init = T(1) / pow(bcl_mu_in, exponent);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in = 1;
	T zero(0);
	T square(2);

	qpdata._h_scaled = qp.H.to_eigen();
	qpdata._g_scaled = qp.g.to_eigen();
	qpdata._a_scaled = qp.A.to_eigen();
	qpdata._b_scaled = qp.b.to_eigen();
	qpdata._c_scaled = qp.C.to_eigen();
	qpdata._u_scaled = qp.u.to_eigen();
	qpdata._l_scaled = qp.l.to_eigen();

	auto qp_scaled = qp::QpViewBoxMut<T>{
			{from_eigen, qpdata._h_scaled},
			{from_eigen, qpdata._g_scaled},
			{from_eigen, qpdata._a_scaled},
			{from_eigen, qpdata._b_scaled},
			{from_eigen, qpdata._c_scaled},
			{from_eigen, qpdata._u_scaled},
			{from_eigen, qpdata._l_scaled}};
	//{
	//LDLT_DECL_SCOPE_TIMER("in solver", "equilibration", T);
	//::Eigen::internal::set_is_malloc_allowed(false);
	precond.scale_qp_in_place(qp_scaled);

	//}
	//{
	//LDLT_DECL_SCOPE_TIMER("in solver", "setting kkt", T);

	qpdata._kkt.topLeftCorner(dim, dim) = qp_scaled.H.to_eigen();
	qpdata._kkt.topLeftCorner(dim, dim).diagonal().array() += rho;	
	qpdata._kkt.block(0, dim, dim, n_eq) = qp_scaled.A.to_eigen().transpose();
	qpdata._kkt.block(dim, 0, n_eq, dim) = qp_scaled.A.to_eigen();
	qpdata._kkt.bottomRightCorner(n_eq, n_eq).setZero();
	qpdata._kkt.diagonal().segment(dim, n_eq).setConstant(-T(1) / bcl_mu_eq);

	ldlt::Ldlt<T> ldl{decompose, qpdata._kkt};

	qpdata._rhs.head(dim) = -qp_scaled.g.to_eigen();
	qpdata._rhs.middleRows(dim,n_eq) = qp_scaled.b.to_eigen();
	ldl.solve_in_place(qpdata._rhs.head(dim+n_eq));
	qpdata._x = qpdata._rhs.head(dim);
	qpdata._y = qpdata._rhs.segment(dim,n_eq);
	//{
	//LDLT_DECL_SCOPE_TIMER("in solver", "warm starting", T);
	//}

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	T correction_guess_rhs_g = infty_norm(qp_scaled.g.to_eigen());

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);

	T refactor_dual_feasibility_threshold(1e-2);
	T refactor_rho_threshold(1e-7);
	T refactor_rho_update_factor(10);

	T cold_reset_bcl_mu_max(1e8);
	T cold_reset_primal_test_factor(1);
	T cold_reset_dual_test_factor(1);
	T cold_reset_mu_eq(1.1);
	T cold_reset_mu_in(1.1);
	isize deletion(0);
	isize adding(0);
	
	for (i64 iter = 0; iter <= max_iter; ++iter) {
		if (iter == max_iter) {
			break;
		}
		n_ext += 1;

		// compute primal residual

		// LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);

		qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				qpdata,
				qp,
				qp_scaled.as_const(),
				precond
				);
		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				qpdata,
				qp_scaled.as_const(),
				precond,
				VectorView<T>{from_eigen,qpdata._x},
				VectorView<T>{from_eigen,qpdata._y},
				VectorView<T>{from_eigen,qpdata._z}
				);
		//}
		//Freturn {0,0,0};
		if (VERBOSE){
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "bcl_eta_ext : " << bcl_eta_ext
								<< " bcl_eta_in : " << bcl_eta_in << " rho : " << rho
								<< " bcl_mu_eq : " << bcl_mu_eq << " bcl_mu_in : " << bcl_mu_in
								<< std::endl;
		}
		const bool is_primal_feasible =
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

		const bool is_dual_feasible =
				dual_feasibility_lhs <=
				(eps_abs +
		     eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 dual_feasibility_rhs_2)));

		if (is_primal_feasible) {
			if (dual_feasibility_lhs > refactor_dual_feasibility_threshold && //
			    rho > refactor_rho_threshold) {
				T rho_new = max2( //
						(rho / refactor_rho_update_factor),
						refactor_rho_threshold);

				qp::detail::refactorize(
						qp_scaled.as_const(),
						VectorViewMut<isize>{from_eigen, qpdata._current_bijection_map},
						MatrixViewMut<T, colmajor>{from_eigen, qpdata._kkt},
						n_c,
						bcl_mu_in,
						rho,
						rho_new,
						ldl);

				rho = rho_new;
			}

			if (is_dual_feasible) {

				//LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
					::Eigen::internal::set_is_malloc_allowed(false);
				precond.unscale_primal_in_place({from_eigen,qpdata._x});
				precond.unscale_dual_in_place_eq({from_eigen,qpdata._y});
				precond.unscale_dual_in_place_in({from_eigen,qpdata._z});

					::Eigen::internal::set_is_malloc_allowed(true);

				/*
				auto timing_map = LDLT_GET_MAP(T)["in solver"];
				double activeSetChange_tmp(0);
				double SolveLS_tmp(0);
				
				auto it = timing_map.find("activeSetChange");
				if (it != timing_map.end()) {
					auto& duration_vec = (*it).second.ref;
					auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
					activeSetChange_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
				}
				it = timing_map.find("SolveLS");
				if (it != timing_map.end()) {
					auto& duration_vec = (*it).second.ref;
					auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
					SolveLS_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
				}
				*/
				return {n_ext, n_mu_updates, n_tot};
			}
		} 

		qpdata._xe = qpdata._x;
		qpdata._ye = qpdata._y;
		qpdata._ze = qpdata._z;
		const bool do_initial_guess_fact = primal_feasibility_lhs < err_IG;

		T err_in(0.);

		if (do_initial_guess_fact) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "initial guess", T);

			err_in = qp::detail::initial_guess_fact(
					qpdata,
					qp_scaled.as_const(),
					qp,
					bcl_mu_in,
					bcl_mu_eq,
					rho,
					bcl_eta_in,
					precond,
					dim,
					n_eq,
					n_in,
					ldl,
					n_c,
					R,
					deletion,
					adding,
					VERBOSE,
					zero);
			n_tot += 1;
		}

		bool do_correction_guess = !do_initial_guess_fact ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in);

		if (do_correction_guess) {
			qpdata._dual_residual_scaled.noalias() -= qp_scaled.C.to_eigen().transpose() * qpdata._z;

			qpdata._ATy.noalias() += bcl_mu_eq * (qp_scaled.A.trans().to_eigen() * qpdata._primal_residual_eq_scaled);
		}

		if (do_initial_guess_fact && err_in >= bcl_eta_in) {
			qpdata._primal_residual_eq_scaled.noalias() += qpdata._y / bcl_mu_eq;

			qpdata._primal_residual_in_scaled_u.noalias() += qpdata._z / bcl_mu_in;
			qpdata._primal_residual_in_scaled_l.noalias() += qpdata._z / bcl_mu_in;
		}
		if (!do_initial_guess_fact) {

			precond.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen, qpdata._primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains Cx unscaled from global primal residual

			qpdata._primal_residual_eq_scaled.noalias()  += qpdata._ye / bcl_mu_eq;
			qpdata._primal_residual_in_scaled_u.noalias()  += qpdata._ze / bcl_mu_in;

			qpdata._primal_residual_in_scaled_l = qpdata._primal_residual_in_scaled_u;

			qpdata._primal_residual_in_scaled_u -= qp_scaled.u.to_eigen();
			qpdata._primal_residual_in_scaled_l -= qp_scaled.l.to_eigen();
		}

		if (do_correction_guess) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "correction guess", T);

			err_in = qp::detail::correction_guess(
					qpdata,
					qp_scaled.as_const(),
					bcl_mu_in,
					bcl_mu_eq,
					rho,
					bcl_eta_in,
					dim,
					n_eq,
					n_in,
					max_iter_in,
					n_tot,
					ldl,
					n_c,
					correction_guess_rhs_g,
					deletion,
					adding,
					VERBOSE,
					zero);
			if (VERBOSE){
				std::cout << "primal_feasibility_lhs " << primal_feasibility_lhs
									<< " error from initial guess : " << err_in << " bcl_eta_in "
									<< bcl_eta_in << std::endl;
			}
		}

		// LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);
		T primal_feasibility_lhs_new(primal_feasibility_lhs);
		//{
		//LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);
		
		qp::detail::global_primal_residual(
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				qpdata,
				qp,
				qp_scaled.as_const(),
				precond
				);

		// LDLT_DECL_SCOPE_TIMER("in solver", "BCL", T);
		qp::detail::BCL_update_fact(
				qpdata,
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				eps_abs,
				n_mu_updates,
				bcl_mu_in,
				bcl_mu_eq,
				dim,
				n_eq,
				n_c,
				ldl,
				beta,
				exponent,
				bcl_eta_ext_init,
				cold_reset_bcl_mu_max,
				VERBOSE);

		// COLD RESTART

		// LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);

		T dual_feasibility_lhs_new(dual_feasibility_lhs);

		//{
		//LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);

		qp::detail::global_dual_residual(
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				qpdata,
				qp_scaled.as_const(),
				precond,
				VectorView<T>{from_eigen,qpdata._x},
				VectorView<T>{from_eigen,qpdata._y},
				VectorView<T>{from_eigen,qpdata._z});

		if ((primal_feasibility_lhs_new >=
		     cold_reset_primal_test_factor *
		         max2(primal_feasibility_lhs, machine_eps)) &&

		    (dual_feasibility_lhs_new >=
		     cold_reset_dual_test_factor *
		         max2(primal_feasibility_lhs, machine_eps)) &&

		    max2(bcl_mu_eq, bcl_mu_in) >= cold_reset_bcl_mu_max) {
			if (VERBOSE){
				std::cout << "cold restart" << std::endl;
			}

			T new_bcl_mu_eq = cold_reset_mu_eq;
			T new_bcl_mu_in = cold_reset_mu_in;
			
			//{
			//LDLT_DECL_SCOPE_TIMER("in solver", "cold restart", T);
			qp::detail::mu_update(
					qpdata,
					bcl_mu_eq,
					new_bcl_mu_eq,
					bcl_mu_in,
					new_bcl_mu_in,
					dim,
					n_eq,
					n_c,
					ldl);

			//}
			bcl_mu_in = new_bcl_mu_in;
			bcl_mu_eq = new_bcl_mu_eq;
		}
	}

	return {max_iter, n_mu_updates, n_tot};
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS */
