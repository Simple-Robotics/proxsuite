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
void iterative_solve_with_permut_fact( //
		Eigen::Matrix<T, Eigen::Dynamic, 1>& rhs,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& sol,
		//Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>&  mat,
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&  mat,
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

template <typename T>
void iterative_residual(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		isize inner_pb_dim) {
	
	qpwork._err.head(inner_pb_dim) = qpwork._rhs.head(inner_pb_dim);

	qpwork._err.head(qpmodel._dim).noalias() -= qpwork._h_scaled * qpwork._dw_aug.head(qpmodel._dim);
  	qpwork._err.head(qpmodel._dim).noalias() -= qpresults._rho * qpwork._dw_aug.head(qpmodel._dim);
  	qpwork._err.head(qpmodel._dim).noalias() -= qpwork._a_scaled.transpose() * qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq);

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			qpwork._err.head(qpmodel._dim).noalias() -= qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + j) * qpwork._c_scaled.row(i);
			qpwork._err(qpmodel._dim + qpmodel._n_eq + j) -=
					(qpwork._c_scaled.row(i)).dot(qpwork._dw_aug.head(qpmodel._dim)) -
					qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + j) * qpresults._mu_in_inv; // mu stores the inverse of mu
		}
	}

	qpwork._err.segment(qpmodel._dim, qpmodel._n_eq).noalias() -= qpwork._a_scaled * qpwork._dw_aug.head(qpmodel._dim) -
	                              qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq) * qpresults._mu_eq_inv; // mu stores the inverse of mu

}

template <typename T>
void iterative_solve_with_permut_fact_new( //
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps,
		isize inner_pb_dim) {

	i32 it = 0;
	//std::cout << "qpwork._rhs entry for newton " << qpwork._rhs << std::endl;

	qpwork._dw_aug.head(inner_pb_dim) = qpwork._rhs.head(inner_pb_dim);
	
	// TEMP
	LDLT_MULTI_WORKSPACE_MEMORY(
		(_htot,Uninit, Mat(qpmodel._dim+qpmodel._n_eq+qpresults._n_c, qpmodel._dim+qpmodel._n_eq+qpresults._n_c),LDLT_CACHELINE_BYTES, T)
	);
	
	auto Htot = _htot.to_eigen().eval();
	Htot.setZero();

	Htot.topLeftCorner(qpmodel._dim, qpmodel._dim) = qpwork._h_scaled;
	for (isize i = 0; i < qpmodel._dim; ++i) {
		Htot(i, i) += qpresults._rho;
	}
	Htot.block(0,qpmodel._dim,qpmodel._dim,qpmodel._n_eq) = qpwork._a_scaled.transpose();
	Htot.block(qpmodel._dim,0,qpmodel._n_eq,qpmodel._dim) = qpwork._a_scaled;
	Htot.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults._mu_eq_inv);
	Htot.diagonal().segment(qpmodel._dim+qpmodel._n_eq, qpresults._n_c).setConstant(-qpresults._mu_in_inv);
	
	for (isize i = 0; i< qpmodel._n_in ; ++i){
		isize j = qpwork._current_bijection_map(i);
		if (j<qpresults._n_c){
			Htot.block(j+qpmodel._dim+qpmodel._n_eq,0,1,qpmodel._dim) = qpwork._c_scaled.row(i) ; 
			Htot.block(0,j+qpmodel._dim+qpmodel._n_eq,qpmodel._dim,1) = qpwork._c_scaled.transpose().col(i) ; 
		}
	}  
	qpwork._ldl.factorize(Htot);
	//
	qpwork._ldl.solve_in_place(qpwork._dw_aug.head(inner_pb_dim));

	auto compute_iterative_residual = [&] {
		qp::detail::iterative_residual<T>(
				qpwork,
				qpresults,
				qpmodel,
				inner_pb_dim);
	};

	//compute_iterative_residual();
	qpwork._err.head(inner_pb_dim) = Htot * qpwork._dw_aug.head(inner_pb_dim) - qpwork._rhs.head(inner_pb_dim);
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
		//compute_iterative_residual();
		qpwork._err.head(inner_pb_dim) = Htot * qpwork._dw_aug.head(inner_pb_dim) - qpwork._rhs.head(inner_pb_dim);
		if (qpsettings._VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(qpwork._err.head(inner_pb_dim)) << std::endl;
		}
	}
	qpwork._err.head(inner_pb_dim).setZero();
	qpwork._rhs.head(inner_pb_dim).setZero();
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
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings) -> T {

	qpwork._primal_residual_in_scaled_u -= qpresults._z * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l -= qpresults._z * qpresults._mu_in_inv; // mu stores the inverse of mu
	T prim_eq_e = infty_norm(qpwork._primal_residual_eq_scaled);

	if (qpsettings._VERBOSE){
		std::cout << "saddle prim_eq_e " << prim_eq_e << std::endl;
	}
	qpwork._dual_residual_scaled.noalias() += qpwork._c_scaled.transpose() * qpresults._z ;

	T dual_e = infty_norm(qpwork._dual_residual_scaled);

	if (qpsettings._VERBOSE){
		std::cout << "saddle dual_e " << dual_e << std::endl;
	}

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
	if (qpsettings._VERBOSE){
		std::cout << "saddle prim_in_e " << prim_in_e << std::endl;
	}
	err = max2(err, prim_in_e);
	if (qpsettings._VERBOSE){
		std::cout << "saddle point error " << err << std::endl;
	}
	return err;
}

template <typename T>
void newton_step_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps) {
	
	qpwork._dw_aug.head(qpmodel._dim).setZero();
	qpwork._l_active_set_n_u.array() = (qpwork._primal_residual_in_scaled_u.array() > 0);
	qpwork._l_active_set_n_l.array() = (qpwork._primal_residual_in_scaled_l.array() < 0);

	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	isize num_active_inequalities = qpwork._active_inequalities.count();
	isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + num_active_inequalities;

	//qpwork._err.head(inner_pb_dim).setZero();
	//qpwork._rhs.head(inner_pb_dim) = qpwork._err.head(inner_pb_dim);
	qpwork._err.setZero();
	qpwork._rhs = qpwork._err;
	qpwork._rhs.head(qpmodel._dim) = -qpwork._dual_residual_scaled;

	{
		qp::line_search::active_set_change_new(
				qpwork,
				qpresults,
				qpmodel);
	}

	{

	detail::iterative_solve_with_permut_fact_new( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			eps,
			//T(1.e-9),
			inner_pb_dim);
	}
	qpwork._dw_aug.tail(qpmodel._n_eq+qpmodel._n_in).setZero();

}



template <typename T>
auto initial_guess_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps_int) -> T {

	qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._dual_residual_scaled});
	qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._CTz});	

	qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u; // _primal_residual_in_scaled_u stores Cx unscaled
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
			qpmodel);

	qpwork._err.head(inner_pb_dim).setZero();
	qpwork._rhs.head(qpmodel._dim).setZero();

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_u(i);
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_l(i);
			}
		} else {
			qpwork._rhs.head(qpmodel._dim).noalias() += qpresults._z(i) * qpwork._c_scaled.row(i); // add CTze_inactif to rhs.head(dim)
		}
	}
	
	qpwork._rhs.head(qpmodel._dim) = -qpwork._dual_residual_scaled; // rhs.head(dim) contains now : -(Hxe + g + ATye + CTze_actif)
	qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq) = -qpwork._primal_residual_eq_scaled;

	{
	detail::iterative_solve_with_permut_fact_new( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			eps_int,
			inner_pb_dim);
	}

	qpwork._d_dual_for_eq = qpwork._rhs.head(qpmodel._dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz_actif by definition of the solution

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork._dw_aug(j + qpmodel._dim + qpmodel._n_eq) * qpwork._c_scaled.row(i); 
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork._dw_aug(j + qpmodel._dim + qpmodel._n_eq) * qpwork._c_scaled.row(i);
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
			qpwork._Cdx(j) = qpwork._c_scaled.row(j).dot(qpwork._dw_aug.head(qpmodel._dim));
		}
	}
	qpwork._dw_aug.tail(qpmodel._n_in) = qpwork._active_part_z ;

	//qpwork._primal_residual_in_scaled_u += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_u = qpwork._c_scaled * qpwork._xe -qpwork._u_scaled + qpwork._ze * qpresults._mu_in_inv;
	//qpwork._primal_residual_in_scaled_l += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l = qpwork._c_scaled * qpwork._xe -qpwork._l_scaled + qpwork._ze * qpresults._mu_in_inv;

	//qpwork._d_primal_residual_eq = qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq); // By definition of linear system solution // seems unprecise
	qpwork._d_primal_residual_eq.noalias() = qpwork._a_scaled * qpwork._dw_aug.head(qpmodel._dim) - qpwork._dw_aug.segment(qpmodel._dim,qpmodel._n_eq) * qpresults._mu_eq_inv;

	qpwork._dual_residual_scaled -= qpwork._CTz; // contains now Hxe+g+ATye
	
	qpwork._alpha = 1.;
	if (qpmodel._n_in >0 ){
		/*
		qp::line_search::initial_guess_LS(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings);
		*/
		qpwork._alpha = qp::line_search::initial_guess_line_search_box(
				VectorView<T>{from_eigen,qpresults._x}, 
				VectorView<T>{from_eigen,qpresults._y}, 
				VectorView<T>{from_eigen,qpwork._ze}, 
				VectorView<T>{from_eigen,qpwork._dw_aug}, 
				qpresults._mu_eq,
				qpresults._mu_in,
				qpresults._rho,
				qp::QpViewBox<T>{
					MatrixView<T,rowmajor>{ldlt::from_eigen, qpwork._h_scaled},
					VectorView<T>{ldlt::from_eigen, qpwork._g_scaled},
					MatrixView<T,rowmajor>{ldlt::from_eigen, qpwork._a_scaled},
					VectorView<T>{ldlt::from_eigen, qpwork._b_scaled},
					MatrixView<T,rowmajor>{ldlt::from_eigen, qpwork._c_scaled},
					VectorView<T>{ldlt::from_eigen, qpwork._u_scaled},
					VectorView<T>{ldlt::from_eigen, qpwork._l_scaled}});
	}
	if (qpsettings._VERBOSE){
		std::cout << "alpha from initial guess " << qpwork._alpha << std::endl;
	}
	
	qpwork._primal_residual_in_scaled_u += (qpwork._alpha * qpwork._Cdx);
	qpwork._primal_residual_in_scaled_l += (qpwork._alpha * qpwork._Cdx);
	qpwork._l_active_set_n_u = (qpwork._primal_residual_in_scaled_u.array() >= 0).matrix();
	qpwork._l_active_set_n_l = (qpwork._primal_residual_in_scaled_l.array() <= 0).matrix();
	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpresults._x.noalias() += qpwork._alpha * qpwork._dw_aug.head(qpmodel._dim);
	qpresults._y.noalias() += qpwork._alpha * qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq);

	qpwork._active_part_z = qpresults._z + qpwork._alpha * qpwork._dw_aug.tail(qpmodel._n_in) ;

	qpwork._residual_in_z_u_plus_alpha = (qpwork._active_part_z.array() > 0).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	qpwork._residual_in_z_l_plus_alpha = (qpwork._active_part_z.array() < 0).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));

	qpresults._z = (qpwork._l_active_set_n_u).select(qpwork._residual_in_z_u_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (qpwork._l_active_set_n_l).select(qpwork._residual_in_z_l_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (!qpwork._l_active_set_n_l.array() && !qpwork._l_active_set_n_u.array()).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) ;
	
	qpwork._primal_residual_eq_scaled.noalias() += qpwork._alpha * qpwork._d_primal_residual_eq;
	qpwork._dual_residual_scaled.noalias() += qpwork._alpha * qpwork._d_dual_for_eq;

	qpwork._ATy = qpwork._dual_residual_scaled ;  // will be used in correction guess if needed : contains Hx_new + rho*(x_new-xe) + g + ATynew
	T err = detail::saddle_point(
			qpwork,
			qpresults,
			qpmodel,
			qpsettings);
	
	return err;
}

template <typename T>
auto correction_guess(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps_int) -> T {

	T err_in(1.);
	isize iter_m(0);

	for (i64 iter = 0; iter <= qpsettings._max_iter_in; ++iter) {

		if (iter == qpsettings._max_iter_in) {
			qpresults._n_tot += qpsettings._max_iter_in;
			break;
		}

		
		qp::detail::newton_step_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				eps_int);

		qpwork._alpha = 1.;
		qpwork._d_dual_for_eq.noalias() = qpwork._h_scaled * qpwork._dw_aug.head(qpmodel._dim);
		//qpwork._d_primal_residual_eq.noalias() = qpwork._dw_aug.segment(qpmodel._dim,qpmodel._n_eq) * qpresults._mu_eq_inv; // by definition Adx = dy / mu : seems unprecise
		qpwork._d_primal_residual_eq.noalias() = qpwork._a_scaled * qpwork._dw_aug.head(qpmodel._dim);
		qpwork._Cdx.noalias() = qpwork._c_scaled * qpwork._dw_aug.head(qpmodel._dim);
		if ( true ) {
			//std::cout << "LS err_in " << err_in << std::endl;
			qp::line_search::correction_guess_LS(
					qpwork,
					qpresults,
					qpmodel);
		}else{
			//std::cout << "adding err_in " << err_in << std::endl;
			iter_m+=1;
		}

		if (infty_norm(qpwork._alpha * qpwork._dw_aug.head(qpmodel._dim)) < 1.E-11 || iter_m == 5) {
			qpresults._n_tot += iter + 1;
			break;
		}

		qpresults._x.noalias() += qpwork._alpha * qpwork._dw_aug.head(qpmodel._dim);

		qpwork._primal_residual_in_scaled_u.noalias() += qpwork._alpha * qpwork._Cdx;
		qpwork._primal_residual_in_scaled_l.noalias() += qpwork._alpha * qpwork._Cdx;
		qpwork._primal_residual_eq_scaled.noalias() += qpwork._alpha * qpwork._d_primal_residual_eq;
		qpresults._y.noalias() = qpwork._primal_residual_eq_scaled * qpresults._mu_eq; //mu stores mu

		qpwork._Hx.noalias() += qpwork._alpha * qpwork._d_dual_for_eq ; // stores Hx
		qpwork._ATy.noalias() = (qpwork._a_scaled).transpose() * qpresults._y ;

		qpresults._z =  (qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l)) *  qpresults._mu_in; //mu stores mu
		T rhs_c = max2(qpwork._correction_guess_rhs_g, infty_norm( qpwork._Hx));
		rhs_c = max2(rhs_c, infty_norm(qpwork._ATy));
		qpwork._dual_residual_scaled.noalias() = qpwork._c_scaled.transpose() * qpresults._z ; 
		rhs_c = max2(rhs_c, infty_norm(qpwork._dual_residual_scaled));
		qpwork._dual_residual_scaled.noalias() += qpwork._Hx + qpwork._g_scaled + qpwork._ATy + qpresults._rho * (qpresults._x - qpwork._xe);

		err_in = infty_norm(qpwork._dual_residual_scaled);
		if (qpsettings._VERBOSE){
			std::cout << "---it in " << iter << " projection norm " << err_in
							<< " alpha " << qpwork._alpha << std::endl;
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
	/*
	qpwork._ldl.factorize(qpwork._kkt);
	qpwork._rhs.head(qpmodel._dim) = -qpwork._g_scaled;
	qpwork._rhs.segment(qpmodel._dim,qpmodel._n_eq) = qpwork._b_scaled;
	//qpwork._ldl.solve_in_place(qpwork._rhs.head(qpmodel._dim+qpmodel._n_eq));
	detail::iterative_solve_with_permut_fact_new( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			T(1.e-9),
			qpmodel._dim+qpmodel._n_eq);

	qpresults._x = qpwork._dw_aug.head(qpmodel._dim);
	qpresults._y = qpwork._dw_aug.segment(qpmodel._dim,qpmodel._n_eq);
	//qpresults._x = qpwork._rhs.head(qpmodel._dim);
	//qpresults._y = qpwork._rhs.segment(qpmodel._dim,qpmodel._n_eq);
	qpwork._rhs.setZero();
	qpwork._dw_aug.setZero();

	qpwork._xe = qpresults._x;
	qpwork._ye = qpresults._y;
	qpwork._ze = qpresults._z;
	*/

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
				qpmodel
				);
		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				qpwork,
				qpresults
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
												 qpwork._primal_feasibility_rhs_1_eq,
												 qpwork._primal_feasibility_rhs_1_in_u),
										 qpwork._primal_feasibility_rhs_1_in_l)

										 ));

		const bool is_dual_feasible =
				dual_feasibility_lhs <=
				(qpsettings._eps_abs +
		     qpsettings._eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 qpwork._dual_feasibility_rhs_2)));
		
		if (is_primal_feasible) {
			/*
			if (dual_feasibility_lhs > qpsettings._refactor_dual_feasibility_threshold && //
			    qpresults._rho > qpsettings._refactor_rho_threshold) {
				T rho_new = max2( //
						(qpresults._rho * qpsettings._refactor_rho_update_factor),
						qpsettings._refactor_rho_threshold);
				qp::detail::refactorize(
						qpwork,
						qpresults,
						qpmodel,
						rho_new
						);

				qpresults._rho = rho_new;
			}
			*/
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
		
		const bool do_initial_guess_fact = (primal_feasibility_lhs < qpsettings._eps_IG || qpmodel._n_in == 0 ) ;

		T err_in(0.);

		if (do_initial_guess_fact) {

			err_in = qp::detail::initial_guess_fact(
					qpwork,
					qpresults,
					qpmodel,
					qpsettings,
					bcl_eta_in);
			qpresults._n_tot += 1;
		}

		bool do_correction_guess = (!do_initial_guess_fact && qpmodel._n_in != 0) ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in && qpmodel._n_in != 0) ;

		if (do_initial_guess_fact && err_in >= bcl_eta_in ) {

			/*
			* ATy contains : Hx_new + rho*(x_new-xe) + ATy_new
			* _primal_residual_eq_scaled contains : Ax_new - b -(y_new-ye)//mu_eq
			* Hence ATy becomes below as wanted : Hx_new + rho*(x_new-xe) + mu_eq * AT(Ax_new-b + ye/mu_eq)
			*/
			qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
			qpwork._Hx.noalias() += qpwork._alpha * qpwork._h_scaled * qpwork._dw_aug.head(qpmodel._dim);
			qpwork._ATy.noalias() +=  (qpwork._a_scaled.transpose() * qpwork._primal_residual_eq_scaled) * qpresults._mu_eq ; //mu stores mu
			qpwork._primal_residual_eq_scaled.noalias() += qpresults._y * qpresults._mu_eq_inv ; // contains now Ax_new - b + ye/mu_eq
			qpwork._active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
			qpwork._dual_residual_scaled.noalias() = qpwork._ATy;
			qpwork._dual_residual_scaled.noalias() +=  qpwork._c_scaled.transpose() * qpwork._active_part_z * qpresults._mu_in ; //mu stores mu  // used for newton step at first iteration

			qpwork._primal_residual_in_scaled_u.noalias() += qpresults._z * qpresults._mu_in_inv; //mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l.noalias() += qpresults._z * qpresults._mu_in_inv; //mu stores the inverse of mu
		}
		if (!do_initial_guess_fact ) {

			qpwork._ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains Cx unscaled from global primal residual
			qpwork._primal_residual_eq_scaled.noalias()  += qpwork._ye * qpresults._mu_eq_inv;//mu stores the inverse of mu
			qpwork._ATy.noalias() =  (qpwork._a_scaled.transpose() * qpwork._primal_residual_eq_scaled) * qpresults._mu_eq ; //mu stores mu
			qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
			
			qpwork._primal_residual_in_scaled_u.noalias()  += qpwork._ze * qpresults._mu_in_inv;//mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
			qpwork._primal_residual_in_scaled_u -= qpwork._u_scaled;
			qpwork._primal_residual_in_scaled_l -= qpwork._l_scaled;
			qpwork._dual_residual_scaled.noalias() = qpwork._Hx + qpwork._ATy + qpwork._g_scaled;
			qpwork._dual_residual_scaled.noalias() += qpresults._rho * (qpresults._x - qpwork._xe);
			qpwork._active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
			qpwork._dual_residual_scaled.noalias() +=  qpwork._c_scaled.transpose() * qpwork._active_part_z * qpresults._mu_in ; //mu stores mu  // used for newton step at first iteration

		}
		

		if (do_correction_guess) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "correction guess", T);

			err_in = qp::detail::correction_guess(
					qpwork,
					qpresults,
					qpmodel,
					qpsettings,
					bcl_eta_in);
			if (qpsettings._VERBOSE){
				std::cout << "primal_feasibility_lhs " << primal_feasibility_lhs
									<< " inner loop error : " << err_in << " bcl_eta_in "
									<< bcl_eta_in << std::endl;
			}
		}

		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		qp::detail::global_primal_residual(
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				qpwork,
				qpresults,
				qpmodel
				);
		
		T dual_feasibility_lhs_new(dual_feasibility_lhs);

		qp::detail::global_dual_residual(
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				qpwork,
				qpresults
				);
		const bool is_primal_feasible_new =
				primal_feasibility_lhs_new <=
				(qpsettings._eps_abs +
		     qpsettings._eps_rel *
		         max2(
								 max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 max2(
										 max2(
												 qpwork._primal_feasibility_rhs_1_eq,
												 qpwork._primal_feasibility_rhs_1_in_u),
										 qpwork._primal_feasibility_rhs_1_in_l)

										 ));

		const bool is_dual_feasible_new =
				dual_feasibility_lhs_new <=
				(qpsettings._eps_abs +
		     qpsettings._eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 qpwork._dual_feasibility_rhs_2)));

		if (is_primal_feasible_new) {
			if (is_dual_feasible_new) {

				qpwork._ruiz.unscale_primal_in_place({from_eigen,qpresults._x});
				qpwork._ruiz.unscale_dual_in_place_eq({from_eigen,qpresults._y});
				qpwork._ruiz.unscale_dual_in_place_in({from_eigen,qpresults._z});

				::Eigen::internal::set_is_malloc_allowed(true);
				break;
			}
		} 
			
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
		
		qpwork._xe = qpresults._x;
		qpwork._ye = qpresults._y;
		qpwork._ze = qpresults._z;
		
		::Eigen::internal::set_is_malloc_allowed(true);
	}
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS */
