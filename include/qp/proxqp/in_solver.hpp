#ifndef INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "qp/utils.hpp"
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
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		VectorViewMut<T> rhs,
		VectorViewMut<T> err,
		VectorView<T> sol,
		T mu_eq,
		T mu_in,
		T rho) {
	auto current_bijection_map = current_bijection_map_.to_eigen();
	auto err_ = err.to_eigen();
	auto sol_ = sol.to_eigen();

	err_ = -rhs.to_eigen();
	err_.topRows(dim) +=
			(qp_scaled.H).to_eigen() * sol_.topRows(dim) + rho * sol_.topRows(dim) +
			(qp_scaled.A).to_eigen().transpose() * sol_.middleRows(dim, n_eq);

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			err_.topRows(dim) += sol_(dim + n_eq + j) * qp_scaled.C.to_eigen().row(i);
			err_(dim + n_eq + j) +=
					(qp_scaled.C.to_eigen().row(i)).dot(sol_.topRows(dim)) -
					sol_(dim + n_eq + j) / mu_in;
		}
	}

	err_.middleRows(dim, n_eq) += (qp_scaled.A).to_eigen() * sol_.topRows(dim) -
	                              sol_.middleRows(dim, n_eq) / mu_eq;
}

template <typename T>
void iterative_solve_with_permut_fact_new( //
		VectorViewMut<T> rhs,
		VectorViewMut<T> sol,
		VectorViewMut<T> res,
		ldlt::Ldlt<T> const& ldl,
		T eps,
		isize max_it,
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		T mu_eq,
		T mu_in,
		T rho,
		bool& VERBOSE) {

	auto rhs_ = rhs.to_eigen();
	auto sol_ = sol.to_eigen();
	auto res_ = res.to_eigen();

	i32 it = 0;
	sol_ = rhs_;
	ldl.solve_in_place(sol_);

	auto compute_iterative_residual = [&] {
		qp::detail::iterative_residual<T>(
				qp_scaled,
				current_bijection_map_,
				dim,
				n_eq,
				n_c,
				n_in,
				{from_eigen, rhs_},
				{from_eigen, res_},
				{from_eigen, sol_},
				mu_eq,
				mu_in,
				rho);
	};

	compute_iterative_residual();
	++it;
	if (VERBOSE){
		std::cout << "infty_norm(res) " << qp::infty_norm(res_) << std::endl;
	}
	while (infty_norm(res_) >= eps) {
		if (it >= max_it) {
			break;
		}
		++it;
		res_ = -res_;
		ldl.solve_in_place(res_);
		sol_ += res_;

		res_.setZero();
		compute_iterative_residual();
		if (VERBOSE){
			std::cout << "infty_norm(res) " << qp::infty_norm(res_) << std::endl;
		}
	}
}

template <typename T>
void BCL_update_fact(
		T& primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		isize& n_mu_updates,
		T& bcl_mu_in,
		T& bcl_mu_eq,
		VectorViewMut<T> ye,
		VectorViewMut<T> ze,
		VectorViewMut<T> y,
		VectorViewMut<T> z,

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
		y.to_eigen() = ye.to_eigen();
		z.to_eigen() = ze.to_eigen();
		T new_bcl_mu_in(min2(bcl_mu_in * 10, cold_reset_bcl_mu_max));
		T new_bcl_mu_eq(min2(bcl_mu_eq * (10), cold_reset_bcl_mu_max * 100));
		if (bcl_mu_in != new_bcl_mu_in || bcl_mu_eq != new_bcl_mu_eq) {
			{ ++n_mu_updates; }
		}
		qp::detail::mu_update(
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
auto SaddlePointError(
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
		isize n_in) -> T {

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

	auto prim_in_u = C_ * x_ - u_ - (z_ - z_e) / mu_in;
	auto prim_in_l = C_ * x_ - l_ - (z_ - z_e) / mu_in;

	T prim_eq_e = infty_norm(A_ * x_ - b_ - (y_ - y_e) / mu_eq);
	T dual_e = infty_norm(
			H_ * x_ + rho * (x_ - x_e) + g_ + A_.transpose() * y_ +
			C_.transpose() * z_);
	T err = max2(prim_eq_e, dual_e);

	T prim_in_e(0);

	for (isize i = 0; i < n_in; i = i + 1) {
		using std::fabs;
		if (z_(i) > 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_u(i)));
		} else if (z_(i) < 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_l(i)));
		} else {
			prim_in_e = max2(prim_in_e, max2(prim_in_u(i), T(0)));
			prim_in_e = max2(prim_in_e, fabs(min2(prim_in_l(i), T(0))));
		}
	}
	err = max2(err, prim_in_e);
	return err;
}

template <typename T>
auto saddle_point(
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> x,
		VectorView<T> y,
		VectorView<T> z,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		T mu_in,
		isize n_in,
		VectorViewMut<T> prim_in_u,
		VectorViewMut<T> prim_in_l,
		VectorView<T> prim_eq,
		VectorViewMut<T> dual_eq) -> T {

	auto H_ = qp_scaled.H.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto x_e = xe.to_eigen();
	auto y_ = y.to_eigen();
	auto y_e = ye.to_eigen();
	auto z_ = z.to_eigen();
	auto z_e = ze.to_eigen();
	auto b_ = qp_scaled.b.to_eigen();
	auto l_ = qp_scaled.l.to_eigen();
	auto u_ = qp_scaled.u.to_eigen();

	prim_in_u.to_eigen() -= z_ / mu_in;
	prim_in_l.to_eigen() -= z_ / mu_in;
	T prim_eq_e = infty_norm(prim_eq.to_eigen());

	dual_eq.to_eigen().noalias() += C_.transpose() * z_;

	T dual_e = infty_norm(dual_eq.to_eigen());
	T err = max2(prim_eq_e, dual_e);

	T prim_in_e(0);

	for (isize i = 0; i < n_in; ++i) {
		using std::fabs;

		if (z_(i) > 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_u(i)));
		} else if (z_(i) < 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_l(i)));
		} else {
			prim_in_e = max2(prim_in_e, max2(prim_in_u(i), T(0)));
			prim_in_e = max2(prim_in_e, fabs(min2(prim_in_l(i), T(0))));
		}
	}
	err = max2(err, prim_in_e);
	return err;
}

template <typename T>
void newton_step_fact(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<T> dx,
		T mu_eq,
		T mu_in,
		T rho,
		T eps,
		isize dim,
		isize n_eq,
		isize n_in,
		VectorViewMut<T> z_pos,
		VectorViewMut<T> z_neg,
		VectorViewMut<T> dual_for_eq,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		isize& deletion,
		isize& adding,
		bool& VERBOSE) {

	auto z_pos_ = z_pos.to_eigen();
	auto z_neg_ = z_neg.to_eigen();
	auto dual_for_eq_ = dual_for_eq.to_eigen();
	auto l_active_set_n_u_ = l_active_set_n_u.to_eigen();
	auto l_active_set_n_l_ = l_active_set_n_l.to_eigen();
	auto active_inequalities_ = active_inequalities.to_eigen();

	auto H_ = qp_scaled.H.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();

	l_active_set_n_u_ = (z_pos_.array() > 0).matrix();
	l_active_set_n_l_ = (z_neg_.array() < 0).matrix();

	active_inequalities_ = l_active_set_n_u_ || l_active_set_n_l_;

	isize num_active_inequalities = active_inequalities_.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_rhs, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_err, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T));

	auto rhs = _rhs.to_eigen();
	auto dw = _dw.to_eigen();
	auto err = _err.to_eigen();
	{
		//LDLT_DECL_SCOPE_TIMER("in solver", "activeSetChange", T);
		qp::line_search::active_set_change_new(
				VectorView<bool>{from_eigen, active_inequalities_},
				current_bijection_map,
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

	rhs.topRows(dim) -= dual_for_eq_;
	for (isize j = 0; j < n_in; ++j) {
		rhs.topRows(dim) -=
				mu_in * (max2(z_pos_(j), T(0)) + min2(z_neg_(j), T(0))) * C_.row(j);
	}
	{
	//LDLT_DECL_SCOPE_TIMER("in solver", "SolveLS", T);
	detail::iterative_solve_with_permut_fact_new( //
			{from_eigen, rhs},
			{from_eigen, dw},
			{from_eigen, err},
			ldl,
			eps,
			3,
			qp_scaled,
			current_bijection_map,
			dim,
			n_eq,
			n_c,
			n_in,
			mu_eq,
			mu_in,
			rho,
			VERBOSE);
	}
	dx.to_eigen() = dw.topRows(dim);
}

template <typename T, typename Preconditioner>
auto initial_guess_fact(
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp_scaled,
		qp::QpViewBox<T> qp,
		T mu_in,
		T mu_eq,
		T rho,
		T eps_int,
		Preconditioner precond,
		isize dim,
		isize n_eq,
		isize n_in,

		VectorViewMut<T> primal_residual_eq,
		VectorViewMut<T> prim_in_u,
		VectorViewMut<T> prim_in_l,
		VectorViewMut<T> dual_for_eq,
		VectorViewMut<T> d_dual_for_eq,
		VectorViewMut<T> cdx,
		VectorViewMut<T> d_primal_residual_eq,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,
		VectorViewMut<T> dw_aug,

		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		T R,
		isize& deletion,
		isize& adding,
		bool& VERBOSE) -> T {

	auto primal_residual_eq_ = primal_residual_eq.to_eigen();
	auto prim_in_u_ = prim_in_u.to_eigen();
	auto prim_in_l_ = prim_in_l.to_eigen();
	auto dual_for_eq_ = dual_for_eq.to_eigen();
	auto d_dual_for_eq_ = d_dual_for_eq.to_eigen();
	auto cdx_ = cdx.to_eigen();
	auto d_primal_residual_eq_ = d_primal_residual_eq.to_eigen();
	auto l_active_set_n_u_ = l_active_set_n_u.to_eigen();
	auto l_active_set_n_l_ = l_active_set_n_l.to_eigen();
	auto active_inequalities_ = active_inequalities.to_eigen();
	auto dw_aug_ = dw_aug.to_eigen();

	auto H_ = qp_scaled.H.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_ = z.to_eigen();
	auto z_e = ze.to_eigen().eval();
	auto l_ = qp_scaled.l.to_eigen();
	auto u_ = qp_scaled.u.to_eigen();

	{
		//prim_in_u_ = C_ * x_; prim_in_u contains Cx unscaled from global primal residual
		prim_in_l_ = prim_in_u_;

		prim_in_u_ -= qp.u.to_eigen();
		prim_in_l_ -= qp.l.to_eigen();
	}

	//precond.unscale_primal_residual_in_place_in(
	//		VectorViewMut<T>{from_eigen, prim_in_u_});
	//precond.unscale_primal_residual_in_place_in(
	//		VectorViewMut<T>{from_eigen, prim_in_l_});
	precond.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, z_e});

	prim_in_u_ += z_e / mu_in;
	prim_in_l_ += z_e / mu_in;

	l_active_set_n_u_.array() = (prim_in_u_.array() >= 0);
	l_active_set_n_l_.array() = (prim_in_l_.array() <= 0);

	active_inequalities_ = l_active_set_n_u_ || l_active_set_n_l_;

	prim_in_u_ -= z_e / mu_in;
	prim_in_l_ -= z_e / mu_in;

	precond.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, prim_in_u_});
	precond.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, prim_in_l_});
	precond.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, z_e});

	// rescale value
	isize num_active_inequalities = active_inequalities_.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	{
	//LDLT_DECL_SCOPE_TIMER("in solver", "activeSetChange", T);
	qp::line_search::active_set_change_new(
			VectorView<bool>{from_eigen, active_inequalities_},
			current_bijection_map,
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
	LDLT_MULTI_WORKSPACE_MEMORY(
			(_rhs, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(err_it_, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T));
	auto rhs = _rhs.to_eigen().eval();
	auto dw = _dw.to_eigen().eval();
	auto err_it = err_it_.to_eigen().eval();

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			if (l_active_set_n_u_(i)) {
				rhs(j + dim + n_eq) = -prim_in_u_(i);
			} else if (l_active_set_n_l_(i)) {
				rhs(j + dim + n_eq) = -prim_in_l_(i);
			}
		} else {
			rhs.topRows(dim) += z_(i) * C_.row(i);
		}
	}

	rhs.topRows(dim) = -dual_for_eq_;
	rhs.middleRows(dim, n_eq) = -primal_residual_eq_;
	{
	//LDLT_DECL_SCOPE_TIMER("in solver", "SolveLS", T);
	detail::iterative_solve_with_permut_fact_new( //
			{from_eigen, rhs},
			{from_eigen, dw},
			{from_eigen, err_it},
			ldl,
			eps_int,
			3,
			qp_scaled,
			current_bijection_map,
			dim,
			n_eq,
			n_c,
			n_in,
			mu_eq,
			mu_in,
			rho,
			VERBOSE);
	}
	d_dual_for_eq_ = rhs.topRows(dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz
	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			if (l_active_set_n_u_(i)) {
				d_dual_for_eq_ -= dw(j + dim + n_eq) * C_.row(i);
			} else if (l_active_set_n_l_(i)) {
				d_dual_for_eq_ -= dw(j + dim + n_eq) * C_.row(i);
			}
		}
	}

	dw_aug_.setZero();
	dw_aug_.topRows(dim + n_eq) = dw.topRows(dim + n_eq);
	for (isize j = 0; j < n_in; ++j) {
		isize i = current_bijection_map(j);
		if (i < n_c) {
			dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
			cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;
		} else {
			dw_aug_(j + dim + n_eq) = -z_(j);
			cdx_(j) = C_.row(j).dot(dw_aug_.topRows(dim));
		}
	}

	prim_in_u_ += z_e / mu_in;
	prim_in_l_ += z_e / mu_in;

	// d_primal_residual_eq_ = (A_ * dw_aug_.topRows(dim) -
	// dw_aug_.middleRows(dim, n_eq) / mu_eq);
	d_primal_residual_eq_ =
			rhs.middleRows(dim, n_eq); // By definition of linear system solution
	// d_dual_for_eq_ = (H_ * dw_aug_.topRows(dim) +  A_.transpose() *
	// dw_aug_.middleRows(dim, n_eq) +  rho * dw_aug_.topRows(dim));

	// cdx_ = C_ * dw_aug_.topRows(dim);
	dual_for_eq_ -= C_.transpose() * z_e;
	T alpha_step = qp::line_search::initial_guess_LS(
			ze,
			{from_eigen, dw_aug_.tail(n_in)},
			{from_eigen, prim_in_l_},
			{from_eigen, prim_in_u_},
			{from_eigen, cdx_},
			{from_eigen, d_dual_for_eq_},
			{from_eigen, dual_for_eq_},
			{from_eigen, d_primal_residual_eq_},
			{from_eigen, primal_residual_eq_},
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
	prim_in_u_ += (alpha_step * cdx_);
	prim_in_l_ += (alpha_step * cdx_);
	l_active_set_n_u_ = (prim_in_u_.array() >= 0).matrix();
	l_active_set_n_l_ = (prim_in_l_.array() <= 0).matrix();
	active_inequalities_ = l_active_set_n_u_ || l_active_set_n_l_;

	x_ += alpha_step * dw_aug_.topRows(dim);
	y_ += alpha_step * dw_aug_.middleRows(dim, n_eq);

	for (isize i = 0; i < n_in; ++i) {
		if (l_active_set_n_u_(i)) {
			z(i) = max2(z(i) + alpha_step * dw_aug_(dim + n_eq + i), T(0.));
		} else if (l_active_set_n_l_(i)) {
			z(i) = min2(z(i) + alpha_step * dw_aug_(dim + n_eq + i), T(0.));
		} else {
			z(i) += alpha_step * dw_aug_(dim + n_eq + i);
		}
	}
	primal_residual_eq_ += alpha_step * d_primal_residual_eq_;
	dual_for_eq_ += alpha_step * d_dual_for_eq_;

	T err = detail::saddle_point(
			qp_scaled,
			x.as_const(),
			y.as_const(),
			z.as_const(),
			xe,
			ye,
			ze,
			mu_in,
			n_in,
			{from_eigen, prim_in_u_},
			{from_eigen, prim_in_l_},
			{from_eigen, primal_residual_eq_},
			{from_eigen, dual_for_eq_});
	return err;
}

template <typename T>
auto correction_guess(
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
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

		VectorViewMut<T> residual_in_y,
		VectorViewMut<T> z_pos,
		VectorViewMut<T> z_neg,
		VectorViewMut<T> dual_for_eq,
		VectorViewMut<T> Hdx,
		VectorViewMut<T> Adx,
		VectorViewMut<T> Cdx,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,

		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		VectorViewMut<T> dw_aug,
		T& correction_guess_rhs_g,
		isize& deletion,
		isize& adding,
		bool& VERBOSE) -> T {

	auto residual_in_y_ = residual_in_y.to_eigen();
	auto z_pos_ = z_pos.to_eigen();
	auto z_neg_ = z_neg.to_eigen();
	auto dual_for_eq_ = dual_for_eq.to_eigen();
	auto Hdx_ = Hdx.to_eigen();
	auto Adx_ = Adx.to_eigen();
	auto Cdx_ = Cdx.to_eigen();
	auto l_active_set_n_u_ = l_active_set_n_u.to_eigen();
	auto l_active_set_n_l_ = l_active_set_n_l.to_eigen();
	auto active_inequalities_ = active_inequalities.to_eigen();
	auto dw_aug_ = dw_aug.to_eigen();

	T err_in(0);

	for (i64 iter = 0; iter <= max_iter_in; ++iter) {

		if (iter == max_iter_in) {
			n_tot += max_iter_in;
			break;
		}

		dw_aug_.topRows(dim).setZero();

		qp::detail::newton_step_fact(
				qp_scaled,
				VectorViewMut<T>{from_eigen, dw_aug_.topRows(dim)},
				mu_eq,
				mu_in,
				rho,
				eps_int,
				dim,
				n_eq,
				n_in,
				{from_eigen, z_pos_},
				{from_eigen, z_neg_},
				{from_eigen, dual_for_eq_},
				{from_eigen, l_active_set_n_u_},
				{from_eigen, l_active_set_n_l_},
				{from_eigen, active_inequalities_},

				ldl,
				current_bijection_map,
				n_c,
				deletion,
				adding,
				VERBOSE);
		T alpha_step(1);
		Hdx_ = (qp_scaled.H).to_eigen() * dw_aug_.topRows(dim);
		Adx_ = (qp_scaled.A).to_eigen() * dw_aug_.topRows(dim);
		Cdx_ = (qp_scaled.C).to_eigen() * dw_aug_.topRows(dim);
		if (n_in > isize(0)) {
			alpha_step = qp::line_search::correction_guess_LS(
					{from_eigen, Hdx_},
					{from_eigen, Adx_},
					{from_eigen, Cdx_},
					{from_eigen, residual_in_y_},
					{from_eigen, z_pos_},
					{from_eigen, z_neg_},
					{from_eigen, dw_aug_.topRows(dim)},
					qp_scaled.g,
					x.as_const(),
					xe,
					ye,
					ze,
					mu_eq,
					mu_in,
					rho,
					n_in

			);
		}

		if (infty_norm(alpha_step * dw_aug_.topRows(dim)) < 1.E-11) {
			n_tot += iter + 1;
			break;
		}

		x.to_eigen().noalias() += alpha_step * dw_aug_.topRows(dim);
		z_pos_.noalias() += alpha_step * Cdx_;
		z_neg_.noalias() += alpha_step * Cdx_;
		residual_in_y_.noalias() += alpha_step * Adx_;
		y.to_eigen().noalias() = mu_eq * residual_in_y_;
		dual_for_eq_.noalias() 
		+= alpha_step * (mu_eq * (qp_scaled.A).to_eigen().transpose() * Adx_ +
		                  rho * dw_aug_.topRows(dim) + Hdx_) ;

		for (isize j = 0; j < n_in; ++j) {
			z(j) = mu_in * (max2(z_pos_(j), T(0)) + min2(z_neg_(j), T(0)));
		}

		Hdx_.noalias() = (qp_scaled.H).to_eigen() * x.to_eigen();
		T rhs_c = max2(correction_guess_rhs_g, infty_norm(Hdx_));

		dw_aug_.topRows(dim).noalias() =
				(qp_scaled.A.to_eigen().transpose()) * (y.to_eigen());
		rhs_c = max2(rhs_c, infty_norm(dw_aug_.topRows(dim)));
		Hdx_ += (dw_aug_.topRows(dim));

		dw_aug_.topRows(dim).noalias() =
				(qp_scaled.C.to_eigen().transpose()) * (z.to_eigen());
		rhs_c = max2(rhs_c, infty_norm(dw_aug_.topRows(dim)));
		Hdx_ += dw_aug_.topRows(dim);

		Hdx_ += (qp_scaled.g).to_eigen();

		err_in = infty_norm(Hdx_ + rho * (x.to_eigen() - xe.to_eigen()));
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
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		T err_IG,
		T beta,
		T R,
		Preconditioner precond = Preconditioner{},
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
	T bcl_eta_ext_init = 1 / pow(bcl_mu_in, exponent);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in = 1;

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_h_scaled, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_g_scaled, Uninit, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_a_scaled, Uninit, Mat(n_eq, dim), LDLT_CACHELINE_BYTES, T),
			(_c_scaled, Uninit, Mat(n_in, dim), LDLT_CACHELINE_BYTES, T),
			(_b_scaled, Uninit, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_u_scaled, Uninit, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_l_scaled, Uninit, Vec(n_in), LDLT_CACHELINE_BYTES, T),

			(_residual_scaled, Init, Vec(n_tot_max + n_in), LDLT_CACHELINE_BYTES, T),
			(_ye, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_ze, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_xe, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_eq, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_in, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),

			(_kkt,
				Uninit,
				Mat(dim + n_eq + n_c, dim + n_eq + n_c),
				LDLT_CACHELINE_BYTES,
				T),

			(_current_bijection_map, Init, Vec(n_in), LDLT_CACHELINE_BYTES, isize),

			(_dw_aug, Init, Vec(n_tot_max), LDLT_CACHELINE_BYTES, T),
			(_d_dual_for_eq, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_cdx, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_d_primal_residual_eq, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_l_active_set_n_u, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(_l_active_set_n_l, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(_active_inequalities, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool));

	auto d_dual_for_eq = _d_dual_for_eq.to_eigen();
	auto Cdx = _cdx.to_eigen();
	auto d_primal_residual_eq = _d_primal_residual_eq.to_eigen();
	auto l_active_set_n_u = _l_active_set_n_u.to_eigen();
	auto l_active_set_n_l = _l_active_set_n_l.to_eigen();
	auto active_inequalities = _active_inequalities.to_eigen();
	auto dw_aug = _dw_aug.to_eigen();

	auto current_bijection_map = _current_bijection_map.to_eigen();
	for (isize i = 0; i < n_in; i++) {
		current_bijection_map(i) = i;
	}

	auto H_copy = _h_scaled.to_eigen();
	auto kkt = _kkt.to_eigen();
	auto q_copy = _g_scaled.to_eigen();
	auto A_copy = _a_scaled.to_eigen();
	auto C_copy = _c_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();
	auto u_copy = _u_scaled.to_eigen();
	auto l_copy = _l_scaled.to_eigen();

	auto ye = _ye.to_eigen();
	auto ze = _ze.to_eigen();
	auto xe = _xe.to_eigen();
	auto diag_diff_in = _diag_diff_in.to_eigen();
	auto diag_diff_eq = _diag_diff_eq.to_eigen();

	auto residual_scaled = _residual_scaled.to_eigen();

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
	//{
	//LDLT_DECL_SCOPE_TIMER("in solver", "equilibration", T);
	//::Eigen::internal::set_is_malloc_allowed(false);
	precond.scale_qp_in_place(qp_scaled);

	//}
	ldlt::Ldlt<T> ldl{reserve_uninit,dim} ;
	//{
	//LDLT_DECL_SCOPE_TIMER("in solver", "setting kkt", T);
	
	kkt.topLeftCorner(dim, dim) = H_copy;
	kkt.topLeftCorner(dim, dim).diagonal().array() += rho;	
	kkt.block(0, dim, dim, n_eq) = qp_scaled.A.to_eigen().transpose();
	kkt.block(dim, 0, n_eq, dim) = qp_scaled.A.to_eigen();
	kkt.bottomRightCorner(n_eq + n_c, n_eq + n_c).setZero();
	kkt.diagonal().segment(dim, n_eq).setConstant(-T(1) / bcl_mu_eq);
	ldl.factorize(kkt);
	//}
	//ldlt::Ldlt<T> ldl{decompose, kkt};

	//{
	//LDLT_DECL_SCOPE_TIMER("in solver", "warm starting", T);

	// TODO(jcarpent): use a single decomposition
	ldlt::Ldlt<T> ldl_ws{decompose, kkt.topLeftCorner(dim, dim)};
	x.to_eigen() = -qp_scaled.g.to_eigen();
	ldl_ws.solve_in_place(x.to_eigen());
	//}

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	T correction_guess_rhs_g = infty_norm(qp_scaled.g.to_eigen());

	auto dual_residual_scaled = residual_scaled.topRows(dim);
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim, n_eq);
	auto primal_residual_in_scaled_u =
			residual_scaled.middleRows(dim + n_eq, n_in);
	auto primal_residual_in_scaled_l = residual_scaled.bottomRows(n_in);

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
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());
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
		//}
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
						VectorViewMut<isize>{from_eigen, current_bijection_map},
						MatrixViewMut<T, colmajor>{from_eigen, kkt},
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
				precond.unscale_primal_in_place(x);
				precond.unscale_dual_in_place_eq(y);
				precond.unscale_dual_in_place_in(z);
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
				//eturn {double(n_ext), double(n_mu_updates), double(n_tot),activeSetChange_tmp,SolveLS_tmp,double(deletion),double(adding)};
			}
		} 

		xe = x.to_eigen();
		ye = y.to_eigen();
		ze = z.to_eigen();

		const bool do_initial_guess_fact = primal_feasibility_lhs < err_IG;

		T err_in(0.);

		if (do_initial_guess_fact) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "initial guess", T);
			err_in = qp::detail::initial_guess_fact(
					VectorView<T>{from_eigen, xe},
					VectorView<T>{from_eigen, ye},
					VectorView<T>{from_eigen, ze},
					x,
					y,
					z,
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
					{from_eigen, primal_residual_eq_scaled},
					{from_eigen, primal_residual_in_scaled_u},
					{from_eigen, primal_residual_in_scaled_l},
					{from_eigen, dual_residual_scaled},
					{from_eigen, d_dual_for_eq},
					{from_eigen, Cdx},
					{from_eigen, d_primal_residual_eq},
					{from_eigen, l_active_set_n_u},
					{from_eigen, l_active_set_n_l},
					{from_eigen, active_inequalities},
					{from_eigen, dw_aug},

					ldl,
					VectorViewMut<isize>{from_eigen, current_bijection_map},
					n_c,
					R,
					deletion,
					adding,
					VERBOSE);
			n_tot += 1;
		}

		bool do_correction_guess = !do_initial_guess_fact ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in);

		if (do_correction_guess) {
			dual_residual_scaled.noalias() -= qp_scaled.C.to_eigen().transpose() * z.to_eigen();
			dual_residual_scaled.noalias() += bcl_mu_eq * (qp_scaled.A.trans().to_eigen() * primal_residual_eq_scaled);
		}

		if (do_initial_guess_fact && err_in >= bcl_eta_in) {
			primal_residual_eq_scaled.noalias() += y.to_eigen() / bcl_mu_eq;

			primal_residual_in_scaled_u.noalias() += z.to_eigen() / bcl_mu_in;
			primal_residual_in_scaled_l.noalias() += z.to_eigen() / bcl_mu_in;
		}
		if (!do_initial_guess_fact) {
			//auto Cx = qp_scaled.C.to_eigen() * x.to_eigen();

			precond.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen, primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains Cx unscaled from global primal residual

			primal_residual_eq_scaled.noalias()  += ye / bcl_mu_eq;
			primal_residual_in_scaled_u.noalias()  += ze / bcl_mu_in;

			//primal_residual_in_scaled_u += ze / bcl_mu_in;
			//primal_residual_in_scaled_u.noalias() += Cx;

			primal_residual_in_scaled_l = primal_residual_in_scaled_u;

			primal_residual_in_scaled_u -= qp_scaled.u.to_eigen();
			primal_residual_in_scaled_l -= qp_scaled.l.to_eigen();
		}

		if (do_correction_guess) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "correction guess", T);
			err_in = qp::detail::correction_guess(
					{from_eigen, xe},
					{from_eigen, ye},
					{from_eigen, ze},
					x,
					y,
					z,
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
					{from_eigen, primal_residual_eq_scaled},
					{from_eigen, primal_residual_in_scaled_u},
					{from_eigen, primal_residual_in_scaled_l},
					{from_eigen, dual_residual_scaled},
					{from_eigen, d_dual_for_eq},
					{from_eigen, d_primal_residual_eq},
					{from_eigen, Cdx},
					{from_eigen, l_active_set_n_u},
					{from_eigen, l_active_set_n_l},
					{from_eigen, active_inequalities},

					ldl,
					VectorViewMut<isize>{from_eigen, current_bijection_map},
					n_c,
					{from_eigen, dw_aug},
					correction_guess_rhs_g,
					deletion,
					adding,
					VERBOSE);
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
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());

		// LDLT_DECL_SCOPE_TIMER("in solver", "BCL", T);
		qp::detail::BCL_update_fact(
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				eps_abs,
				n_mu_updates,
				bcl_mu_in,
				bcl_mu_eq,
				VectorViewMut<T>{from_eigen, ye},
				VectorViewMut<T>{from_eigen, ze},
				y,
				z,
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
				{from_eigen, dual_residual_scaled},
				{from_eigen, dw_aug},
				qp_scaled.as_const(),
				precond,
				x.as_const(),
				y.as_const(),
				z.as_const());
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
	
	auto timing_map = LDLT_GET_MAP(T)["in solver"];
	//T eq_tmp = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) /
	//			duration_vec.size();
	double equilibration_tmp(0);
	double fact_tmp(0);
	double ws_tmp(0);
	double residuals_tmp(0);
	double IG_tmp(0);
	double CG_tmp(0);
	double BCL_tmp(0);
	double cold_restart_tmp(0);
	auto it = timing_map.find("equilibration");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		equilibration_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	it = timing_map.find("unscale solution");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		fact_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	
	it = timing_map.find("warm starting");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		ws_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	
	it = timing_map.find("residuals");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		residuals_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	
	it = timing_map.find("initial guess");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		IG_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	it = timing_map.find("correction guess");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		CG_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	it = timing_map.find("BCL");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		BCL_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	it = timing_map.find("cold restart");
	if (it != timing_map.end()) {
		auto& duration_vec = (*it).second.ref;
		auto duration = std::accumulate(duration_vec.begin(), duration_vec.end(), Duration{}) ;
		cold_restart_tmp = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() ;
	}
	return {double(max_iter), double(n_mu_updates), double(n_tot),equilibration_tmp,fact_tmp,ws_tmp,residuals_tmp,IG_tmp,CG_tmp,BCL_tmp,cold_restart_tmp};
	*/
	//return {double(max_iter), double(n_mu_updates), double(n_tot),activeSetChange_tmp,SolveLS_tmp,double(deletion),double(adding)};
	return {max_iter, n_mu_updates, n_tot};
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_IN_SOLVER_HPP_HDWGZKCLS */