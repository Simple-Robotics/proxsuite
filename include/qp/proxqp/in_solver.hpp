#ifndef INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "qp/utils.hpp"
#include <qp/QPWorkspace.hpp>
#include <qp/QPResults.hpp>
#include <qp/QPData.hpp>
#include <qp/QPSettings.hpp>
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

#define LDLT_DEDUCE_RET(...)                                                   \
	noexcept(noexcept(__VA_ARGS__))                                              \
			->typename std::remove_const<decltype(__VA_ARGS__)>::type {              \
		return __VA_ARGS__;                                                        \
	}                                                                            \
	static_assert(true, ".")

template <typename T>
void iterative_residual(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::QpViewBox<T> qp_scaled,
		isize inner_pb_dim) {
	
	qpwork._err.head(inner_pb_dim) = qpwork._rhs.head(inner_pb_dim);

	qpwork._err.head(qpmodel._dim).noalias() -= (qp_scaled.H).to_eigen() * qpwork._dw_aug.head(qpmodel._dim);
  qpwork._err.head(qpmodel._dim).noalias() -= qpresults._rho * qpwork._dw_aug.head(qpmodel._dim);
  qpwork._err.head(qpmodel._dim).noalias() -= (qp_scaled.A).to_eigen().transpose() * qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq);

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			qpwork._err.head(qpmodel._dim).noalias() -= qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + j) * qp_scaled.C.to_eigen().row(i);
			qpwork._err(qpmodel._dim + qpmodel._n_eq + j) -=
					(qp_scaled.C.to_eigen().row(i)).dot(qpwork._dw_aug.head(qpmodel._dim)) -
					qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + j) * qpresults._mu_in_inv; // mu stores the inverse of mu
		}
	}

	qpwork._err.segment(qpmodel._dim, qpmodel._n_eq).noalias() -= (qp_scaled.A).to_eigen() * qpwork._dw_aug.head(qpmodel._dim) -
	                              qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq) * qpresults._mu_eq_inv; // mu stores the inverse of mu

}

template <typename T>
void iterative_solve_with_permut_fact_new( //
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps,
		qp::QpViewBox<T> qp_scaled,
		isize inner_pb_dim) {

	i32 it = 0;
	qpwork._dw_aug.head(inner_pb_dim) = qpwork._rhs.head(inner_pb_dim);
	qpwork._ldl.solve_in_place(qpwork._dw_aug.head(inner_pb_dim));

	auto compute_iterative_residual = [&] {
		qp::detail::iterative_residual<T>(
				qpwork,
				qpresults,
				qpmodel,
				qp_scaled,
				inner_pb_dim);
	};

	compute_iterative_residual();
	++it;
	if (qpsettings._VERBOSE){
		std::cout << "infty_norm(res) " << qp::infty_norm( qpwork._err.head(inner_pb_dim)) << std::endl;
	}
	while (infty_norm( qpwork._err.head(inner_pb_dim)) >= eps) {
		if (it >= qpsettings._nb_iterative_refinement) {
			break;
		}
		++it;
		qpwork._ldl.solve_in_place( qpwork._err.head(inner_pb_dim));
		qpwork._dw_aug.head(inner_pb_dim) +=  qpwork._err.head(inner_pb_dim);

		qpwork._err.head(inner_pb_dim).setZero();
		compute_iterative_residual();
		if (qpsettings._VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(qpwork._err.head(inner_pb_dim)) << std::endl;
		}
	}
}

template <typename T>
void BCL_update_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T bcl_eta_ext_init) {

	if (primal_feasibility_lhs <= bcl_eta_ext) {
		if (qpsettings._VERBOSE){
			std::cout << "good step" << std::endl;
		}
		bcl_eta_ext = bcl_eta_ext * pow(qpresults._mu_in_inv, qpsettings._beta_bcl); // mu stores the inverse of mu
		bcl_eta_in = max2(bcl_eta_in * qpresults._mu_in_inv, qpsettings._eps_abs); // mu stores the inverse of mu
	} else {
		if (qpsettings._VERBOSE){
			std::cout << "bad step" << std::endl;
		}
		qpresults._y = qpwork._ye;
		qpresults._z = qpwork._ze;
		T new_bcl_mu_in_inv(max2(qpresults._mu_in_inv * qpsettings._mu_update_inv_factor, qpsettings._mu_max_in_inv)); // mu stores the inverse of mu
		T new_bcl_mu_eq_inv(max2(qpresults._mu_eq_inv * qpsettings._mu_update_inv_factor, qpsettings._mu_max_eq_inv)); // mu stores the inverse of mu
		T new_bcl_mu_in(min2(qpresults._mu_in * qpsettings._mu_update_factor, qpsettings._mu_max_in)); // mu stores mu
		T new_bcl_mu_eq(min2(qpresults._mu_eq * qpsettings._mu_update_factor, qpsettings._mu_max_eq)); // mu stores mu
		if (qpresults._mu_in != new_bcl_mu_in || qpresults._mu_eq != new_bcl_mu_eq) {
			{ ++qpresults._n_mu_change; }
		}
		qp::detail::mu_update(
				qpwork,
				qpresults,
				qpmodel,
				new_bcl_mu_eq_inv,
				new_bcl_mu_in_inv);
		qpresults._mu_eq_inv = new_bcl_mu_eq_inv;
		qpresults._mu_in_inv = new_bcl_mu_in_inv;
		qpresults._mu_eq = new_bcl_mu_eq;
		qpresults._mu_in = new_bcl_mu_in;
		bcl_eta_ext = bcl_eta_ext_init * pow(qpresults._mu_in_inv, qpsettings._alpha_bcl); // mu stores the inverse of mu
		bcl_eta_in = max2(qpresults._mu_in_inv, qpsettings._eps_abs); // mu stores the inverse of mu
	}
}

template <typename T>
auto saddle_point(
		qp::QpViewBox<T> qp_scaled,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel) -> T {

	auto C_ = qp_scaled.C.to_eigen();
	qpwork._primal_residual_in_scaled_u -= qpresults._z * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l -= qpresults._z * qpresults._mu_in_inv; // mu stores the inverse of mu
	T prim_eq_e = infty_norm(qpwork._primal_residual_eq_scaled);

	qpwork._dual_residual_scaled.noalias() += C_.transpose() * qpresults._z ;

	T dual_e = infty_norm(qpwork._dual_residual_scaled);
	T err = max2(prim_eq_e, dual_e);

	T prim_in_e(0);

	for (isize i = 0; i < qpmodel._n_in; ++i) {
		using std::fabs;

		if (qpresults._z(i) > 0) {
			prim_in_e = max2(prim_in_e, fabs(qpwork._primal_residual_in_scaled_u(i)));
		} else if (qpresults._z(i) < 0) {
			prim_in_e = max2(prim_in_e, fabs(qpwork._primal_residual_in_scaled_l(i)));
		} else {
			prim_in_e = max2(prim_in_e, max2(qpwork._primal_residual_in_scaled_u(i), T(0)));
			prim_in_e = max2(prim_in_e, fabs(min2(qpwork._primal_residual_in_scaled_l(i), T(0))));
		}
	}

	err = max2(err, prim_in_e);
	return err;
}

template <typename T>
void newton_step_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		qp::QpViewBox<T> qp_scaled,
		T eps) {
	
	qpwork._l_active_set_n_u.array() = (qpwork._primal_residual_in_scaled_u.array() > 0);
	qpwork._l_active_set_n_l.array() = (qpwork._primal_residual_in_scaled_l.array() < 0);

	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	isize num_active_inequalities = qpwork._active_inequalities.count();
	isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + num_active_inequalities;

	qpwork._err.head(inner_pb_dim).setZero();
	qpwork._rhs.head(inner_pb_dim) = qpwork._err.head(inner_pb_dim);
	qpwork._rhs.head(qpmodel._dim) = -qpwork._dual_residual_scaled;

	{
		qp::line_search::active_set_change_new(
				qpwork,
				qpresults,
				qpmodel,
				qp_scaled);
	}

	{

	detail::iterative_solve_with_permut_fact_new( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			eps,
			qp_scaled,
			inner_pb_dim);
	}

}

template <typename T>
auto initial_guess_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		qp::QpViewBox<T> qp_scaled,
		T eps_int) -> T {

	auto C_ = qp_scaled.C.to_eigen();
	
	qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
	qpwork._primal_residual_in_scaled_u -= qpmodel._u;
	qpwork._primal_residual_in_scaled_l -= qpmodel._l;

	qpwork._ruiz.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._ze});
	qpwork._primal_residual_in_scaled_u += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu

	qpwork._l_active_set_n_u.array() = (qpwork._primal_residual_in_scaled_u .array() >= 0);
	qpwork._l_active_set_n_l.array() = (qpwork._primal_residual_in_scaled_l.array() <= 0);

	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpwork._primal_residual_in_scaled_u -= qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l -= qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu

	qpwork._ruiz.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u});
	qpwork._ruiz.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_l});
	qpwork._ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._ze});
	// rescale value
	isize num_active_inequalities = qpwork._active_inequalities.count();
	isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + num_active_inequalities;


	qp::line_search::active_set_change_new(
			qpwork,
			qpresults,
			qpmodel,
			qp_scaled);

	qpwork._err.head(inner_pb_dim).setZero();

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_u(i);
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_l(i);
			}
		} else {
			qpwork._rhs.head(qpmodel._dim).noalias() += qpresults._z(i) * C_.row(i);
		}
	}
	
	qpwork._rhs.head(qpmodel._dim) = -qpwork._dual_residual_scaled;
	qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq) = -qpwork._primal_residual_eq_scaled;

	{
	detail::iterative_solve_with_permut_fact_new( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			eps_int,
			qp_scaled,
			inner_pb_dim);
	}

	qpwork._d_dual_for_eq = qpwork._rhs.head(qpmodel._dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork._dw_aug(j + qpmodel._dim + qpmodel._n_eq) * C_.row(i);
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork._dw_aug(j + qpmodel._dim + qpmodel._n_eq) * C_.row(i);
			}
		}
	}

	// use active_part_z as a temporary variable to permut back dw_aug newton step
	for (isize j = 0; j < qpmodel._n_in; ++j) {
		isize i = qpwork._current_bijection_map(j);
		if (i < qpresults._n_c) {
			//dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
			//cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;
			
			qpwork._active_part_z(j) = qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + i);
			qpwork._Cdx(j) = qpwork._rhs(i + qpmodel._dim + qpmodel._n_eq) + qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + i) * qpresults._mu_in_inv; // mu stores the inverse of mu
			
		} else {
			//dw_aug_(j + dim + n_eq) = -z_(j);
			qpwork._active_part_z(j) = -qpresults._z(j);
			qpwork._Cdx(j) = C_.row(j).dot(qpwork._dw_aug.head(qpmodel._dim));
		}
	}
	qpwork._dw_aug.tail(qpmodel._n_in) = qpwork._active_part_z ;

	qpwork._primal_residual_in_scaled_u += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu

	qpwork._d_primal_residual_eq = qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq); // By definition of linear system solution

	qpwork._dual_residual_scaled -= qpwork._CTz;

	T alpha_step = qp::line_search::initial_guess_LS(
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			qp_scaled.C);
	
	if (qpsettings._VERBOSE){
		std::cout << "alpha from initial guess " << alpha_step << std::endl;
	}
	
	qpwork._primal_residual_in_scaled_u += (alpha_step * qpwork._Cdx);
	qpwork._primal_residual_in_scaled_l += (alpha_step * qpwork._Cdx);
	qpwork._l_active_set_n_u = (qpwork._primal_residual_in_scaled_u.array() >= 0).matrix();
	qpwork._l_active_set_n_l = (qpwork._primal_residual_in_scaled_l.array() <= 0).matrix();
	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpresults._x.noalias() += alpha_step * qpwork._dw_aug.head(qpmodel._dim);
	qpresults._y.noalias() += alpha_step * qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq);

	
	qpwork._active_part_z = qpresults._z + alpha_step * qpwork._dw_aug.tail(qpmodel._n_in) ;

	qpwork._residual_in_z_u_plus_alpha = (qpwork._active_part_z.array() > 0).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	qpwork._residual_in_z_l_plus_alpha = (qpwork._active_part_z.array() < 0).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));

	qpresults._z = (qpwork._l_active_set_n_u).select(qpwork._residual_in_z_u_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (qpwork._l_active_set_n_l).select(qpwork._residual_in_z_l_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (!qpwork._l_active_set_n_l.array() && !qpwork._l_active_set_n_u.array()).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) ;
	
	qpwork._primal_residual_eq_scaled.noalias() += alpha_step * qpwork._d_primal_residual_eq;
	qpwork._dual_residual_scaled.noalias() += alpha_step * qpwork._d_dual_for_eq;

	T err = detail::saddle_point(
			qp_scaled,
			qpwork,
			qpresults,
			qpmodel);
	
	return err;
}

template <typename T>
auto correction_guess(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		qp::QpViewBox<T> qp_scaled,
		T eps_int,
		T& correction_guess_rhs_g) -> T {

	auto g_ = (qp_scaled.g).to_eigen() ;

	T err_in(0.);

	qpwork._dual_residual_scaled.noalias() = qpwork._Hx + qpwork._ATy + g_;
	qpwork._dual_residual_scaled.noalias() += qpresults._rho * (qpresults._x - qpwork._xe);
	qpwork._active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
	qpwork._dual_residual_scaled.noalias() +=  qp_scaled.C.to_eigen().transpose() * qpwork._active_part_z * qpresults._mu_in ; //mu stores mu  // used for newton step at first iteration


	for (i64 iter = 0; iter <= qpsettings._max_iter_in; ++iter) {

		if (iter == qpsettings._max_iter_in) {
			qpresults._n_tot += qpsettings._max_iter_in;
			break;
		}

		qpwork._dw_aug.head(qpmodel._dim).setZero();
		qp::detail::newton_step_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				qp_scaled,
				eps_int);

		T alpha_step(1);
		qpwork._d_dual_for_eq.noalias() = (qp_scaled.H).to_eigen() * qpwork._dw_aug.head(qpmodel._dim);
		qpwork._d_primal_residual_eq.noalias() = qpwork._dw_aug.segment(qpmodel._dim,qpmodel._n_eq) * qpresults._mu_eq_inv; // by definition Adx = dy / mu //mu stores the inverse of mu
		qpwork._Cdx.noalias() = (qp_scaled.C).to_eigen() * qpwork._dw_aug.head(qpmodel._dim);
		if (qpmodel._n_in > isize(0)) {

			alpha_step = qp::line_search::correction_guess_LS(
					qpwork,
					qpresults,
					qpmodel,
					qp_scaled.g
			);
		}

		if (infty_norm(alpha_step * qpwork._dw_aug.head(qpmodel._dim)) < 1.E-11) {
			qpresults._n_tot += iter + 1;
			break;
		}

		qpresults._x.noalias() += alpha_step * qpwork._dw_aug.head(qpmodel._dim);
		qpwork._primal_residual_in_scaled_u.noalias() += alpha_step * qpwork._Cdx;
		qpwork._primal_residual_in_scaled_l.noalias() += alpha_step * qpwork._Cdx;
		qpwork._primal_residual_eq_scaled.noalias() += alpha_step * qpwork._d_primal_residual_eq;
		qpresults._y.noalias() = qpwork._primal_residual_eq_scaled * qpresults._mu_eq; //mu stores mu

		qpwork._Hx.noalias() += alpha_step * qpwork._d_dual_for_eq ; // stores Hx
		qpwork._ATy.noalias() += (alpha_step * qpresults._mu_eq) * (qp_scaled.A).to_eigen().transpose() * qpwork._d_primal_residual_eq ; // stores ATy //mu stores mu
		
		qpresults._z =  (qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l)) *  qpresults._mu_in; //mu stores mu
		T rhs_c = max2(correction_guess_rhs_g, infty_norm( qpwork._d_dual_for_eq));
		rhs_c = max2(rhs_c, infty_norm(qpwork._ATy));
		qpwork._dual_residual_scaled.noalias() = qp_scaled.C.to_eigen().transpose() * qpresults._z ; 
		rhs_c = max2(rhs_c, infty_norm(qpwork._dual_residual_scaled));
		qpwork._dual_residual_scaled.noalias() += qpwork._Hx + g_ + qpwork._ATy + qpresults._rho * (qpresults._x - qpwork._xe);

		err_in = infty_norm(qpwork._dual_residual_scaled);
		if (qpsettings._VERBOSE){
			std::cout << "---it in " << iter << " projection norm " << err_in
							<< " alpha " << alpha_step << std::endl;
		}
		if (err_in <= eps_int * (1 + rhs_c)) {
			qpresults._n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}


template <typename T>
void qpSolve( //
		qp::Qpdata<T>& qpmodel,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpsettings<T>& qpsettings) {

	using namespace ldlt::tags;
	
	constexpr T machine_eps = std::numeric_limits<T>::epsilon();

	T bcl_eta_ext_init = pow(qpresults._mu_in_inv, qpsettings._alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in = 1;

	qpwork._h_scaled = qpmodel._H;
	qpwork._g_scaled = qpmodel._g;
	qpwork._a_scaled = qpmodel._A;
	qpwork._b_scaled = qpmodel._b;
	qpwork._c_scaled = qpmodel._C;
	qpwork._u_scaled = qpmodel._u;
	qpwork._l_scaled = qpmodel._l;

	auto qp_scaled = qp::QpViewBoxMut<T>{
			MatrixViewMut<T,rowmajor>{ldlt::from_eigen, qpwork._h_scaled},
			VectorViewMut<T>{ldlt::from_eigen, qpwork._g_scaled},
			MatrixViewMut<T,rowmajor>{ldlt::from_eigen, qpwork._a_scaled},
			VectorViewMut<T>{ldlt::from_eigen, qpwork._b_scaled},
			MatrixViewMut<T,rowmajor>{ldlt::from_eigen, qpwork._c_scaled},
			VectorViewMut<T>{ldlt::from_eigen, qpwork._u_scaled},
			VectorViewMut<T>{ldlt::from_eigen, qpwork._l_scaled}};

	qpwork._ruiz.scale_qp_in_place(qp_scaled,VectorViewMut<T>{from_eigen,qpwork._dw_aug}); // avoids temporary allocation in ruiz using another unused for the moment preallocated variable in qpwork

	qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim) = qp_scaled.H.to_eigen();
	qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim).diagonal().array() += qpresults._rho;	
	qpwork._kkt.block(0, qpmodel._dim, qpmodel._dim, qpmodel._n_eq) = qp_scaled.A.to_eigen().transpose();
	qpwork._kkt.block(qpmodel._dim, 0, qpmodel._n_eq, qpmodel._dim) = qp_scaled.A.to_eigen();
	qpwork._kkt.bottomRightCorner(qpmodel._n_eq, qpmodel._n_eq).setZero();
	qpwork._kkt.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults._mu_eq_inv); // mu stores the inverse of mu

	qpwork._ldl.factorize(qpwork._kkt);
	qpwork._rhs.head(qpmodel._dim) = -qp_scaled.g.to_eigen();
	qpwork._rhs.segment(qpmodel._dim,qpmodel._n_eq) = qp_scaled.b.to_eigen();
	qpwork._ldl.solve_in_place(qpwork._rhs.head(qpmodel._dim+qpmodel._n_eq));
	qpresults._x = qpwork._rhs.head(qpmodel._dim);
	qpresults._y = qpwork._rhs.segment(qpmodel._dim,qpmodel._n_eq);

	T primal_feasibility_rhs_1_eq = infty_norm(qpmodel._b);
	T primal_feasibility_rhs_1_in_u = infty_norm(qpmodel._u);
	T primal_feasibility_rhs_1_in_l = infty_norm(qpmodel._l);
	T dual_feasibility_rhs_2 = infty_norm(qpmodel._g);

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
	
	for (i64 iter = 0; iter <= qpsettings._max_iter; ++iter) {
		::Eigen::internal::set_is_malloc_allowed(false);
		if (iter == qpsettings._max_iter) {
			break;
		}
		qpresults._n_ext += 1;

		qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				qpwork,
				qpresults,
				qpmodel,
				qp_scaled.as_const()
				);
		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				qpwork,
				qpresults,
				qp_scaled.as_const()
				);
		if (qpsettings._VERBOSE){
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "bcl_eta_ext : " << bcl_eta_ext
								<< " bcl_eta_in : " << bcl_eta_in << " rho : " << qpresults._rho
								<< " mu_eq : " << qpresults._mu_eq << " mu_in : " << qpresults._mu_in
								<< std::endl;
		}
		const bool is_primal_feasible =
				primal_feasibility_lhs <=
				(qpsettings._eps_abs +
		     qpsettings._eps_rel *
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
				(qpsettings._eps_abs +
		     qpsettings._eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 dual_feasibility_rhs_2)));

		if (is_primal_feasible) {
			if (dual_feasibility_lhs > qpsettings._refactor_dual_feasibility_threshold && //
			    qpresults._rho > qpsettings._refactor_rho_threshold) {
				T rho_new = max2( //
						(qpresults._rho * qpsettings._refactor_rho_update_factor),
						qpsettings._refactor_rho_threshold);
				qp::detail::refactorize(
						qp_scaled.as_const(),
						qpwork,
						qpresults,
						qpmodel,
						rho_new
						);

				qpresults._rho = rho_new;
			}

			if (is_dual_feasible) {

				//LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
				qpwork._ruiz.unscale_primal_in_place({from_eigen,qpresults._x});
				qpwork._ruiz.unscale_dual_in_place_eq({from_eigen,qpresults._y});
				qpwork._ruiz.unscale_dual_in_place_in({from_eigen,qpresults._z});

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
				::Eigen::internal::set_is_malloc_allowed(true);
				break;
			}
		} 

		qpwork._xe = qpresults._x;
		qpwork._ye = qpresults._y;
		qpwork._ze = qpresults._z;
		const bool do_initial_guess_fact = primal_feasibility_lhs < qpsettings._err_IG;

		T err_in(0.);

		if (do_initial_guess_fact) {

			err_in = qp::detail::initial_guess_fact(
					qpwork,
					qpresults,
					qpmodel,
					qpsettings,
					qp_scaled.as_const(),
					bcl_eta_in);
			qpresults._n_tot += 1;
		}

		bool do_correction_guess = !do_initial_guess_fact ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in);

		if (do_correction_guess) {
			qpwork._dual_residual_scaled.noalias() -= qp_scaled.C.to_eigen().transpose() * qpresults._z;

			qpwork._ATy.noalias() +=  (qp_scaled.A.trans().to_eigen() * qpwork._primal_residual_eq_scaled) * qpresults._mu_eq ; //mu stores mu
		}

		if (do_initial_guess_fact && err_in >= bcl_eta_in) {
			qpwork._primal_residual_eq_scaled.noalias() += qpresults._y * qpresults._mu_eq_inv; //mu stores the inverse of mu

			qpwork._primal_residual_in_scaled_u.noalias() += qpresults._z * qpresults._mu_in_inv; //mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l.noalias() += qpresults._z * qpresults._mu_in_inv; //mu stores the inverse of mu
		}
		if (!do_initial_guess_fact) {

			qpwork._ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains Cx unscaled from global primal residual

			qpwork._primal_residual_eq_scaled.noalias()  += qpwork._ye * qpresults._mu_eq_inv;//mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_u.noalias()  += qpwork._ze * qpresults._mu_in_inv;//mu stores the inverse of mu

			qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;

			qpwork._primal_residual_in_scaled_u -= qp_scaled.u.to_eigen();
			qpwork._primal_residual_in_scaled_l -= qp_scaled.l.to_eigen();
		}

		if (do_correction_guess) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "correction guess", T);

			err_in = qp::detail::correction_guess(
					qpwork,
					qpresults,
					qpmodel,
					qpsettings,
					qp_scaled.as_const(),
					bcl_eta_in,
					correction_guess_rhs_g);
			if (qpsettings._VERBOSE){
				std::cout << "primal_feasibility_lhs " << primal_feasibility_lhs
									<< " inner loop error : " << err_in << " bcl_eta_in "
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
				qpwork,
				qpresults,
				qpmodel,
				qp_scaled.as_const()
				);

		// LDLT_DECL_SCOPE_TIMER("in solver", "BCL", T);
		qp::detail::BCL_update_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				bcl_eta_ext_init);

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
				qpwork,
				qpresults,
				qp_scaled.as_const()
				);

		if ((primal_feasibility_lhs_new >= std::max(primal_feasibility_lhs, machine_eps)) 
						 &&
		    (dual_feasibility_lhs_new >=  max2(primal_feasibility_lhs, machine_eps)) 
						 &&
		    	 qpresults._mu_eq_inv <= qpsettings._mu_max_eq_inv 
				 		&& 
				 qpresults._mu_in_inv <= qpsettings._mu_max_in_inv // stores the inverse of mu
				) {
			if (qpsettings._VERBOSE){
				std::cout << "cold restart" << std::endl;
			}
	
			//{
			//LDLT_DECL_SCOPE_TIMER("in solver", "cold restart", T);
			qp::detail::mu_update(
					qpwork,
					qpresults,
					qpmodel,
					qpsettings._cold_reset_mu_eq_inv,
					qpsettings._cold_reset_mu_in_inv);

			//}
			qpresults._mu_in_inv = qpsettings._cold_reset_mu_in_inv; 
			qpresults._mu_eq_inv = qpsettings._cold_reset_mu_eq_inv; 
			qpresults._mu_in = qpsettings._cold_reset_mu_in; 
			qpresults._mu_eq = qpsettings._cold_reset_mu_eq; 
		}
		::Eigen::internal::set_is_malloc_allowed(true);
	}
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS */
