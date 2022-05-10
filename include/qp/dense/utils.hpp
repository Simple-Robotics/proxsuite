#ifndef PROXSUITE_INCLUDE_QP_DENSE_UTILS_HPP
#define PROXSUITE_INCLUDE_QP_DENSE_UTILS_HPP

#include "qp/dense/views.hpp"
#include "qp/dense/Workspace.hpp"
#include <qp/dense/Data.hpp>
#include <qp/Results.hpp>
#include <qp/Settings.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <type_traits>

namespace proxsuite {
namespace qp {
namespace dense {

template <typename Derived>
void save_data(
		const std::string& filename, const ::Eigen::MatrixBase<Derived>& mat) {
	// https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
	const static Eigen::IOFormat CSVFormat(
			Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(filename);
	if (file.is_open()) {
		file << mat.format(CSVFormat);
		file.close();
	}
}

#define LDLT_DEDUCE_RET(...)                                                   \
	noexcept(noexcept(__VA_ARGS__))                                              \
			->typename std::remove_const<decltype(__VA_ARGS__)>::type {              \
		return __VA_ARGS__;                                                        \
	}                                                                            \
	static_assert(true, ".")
template <typename T>
auto positive_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));

// COMPUTES:
// primal_residual_eq_scaled = scaled(Ax - b)
//
// primal_feasibility_lhs = max(norm(unscaled(Ax - b)),
//                              norm(unscaled([Cx - u]+ + [Cx - l]-)))
// primal_feasibility_eq_rhs_0 = norm(unscaled(Ax))
// primal_feasibility_in_rhs_0 = norm(unscaled(Cx))
//
// MAY_ALIAS[primal_residual_in_scaled_u, primal_residual_in_scaled_l]
//
// INDETERMINATE:
// primal_residual_in_scaled_u = unscaled(Cx)
// primal_residual_in_scaled_l = unscaled([Cx - u]+ + [Cx - l]-)

template <typename T>
void global_primal_residual(
		const proxsuite::qp::dense::Data<T>& qpmodel,
		proxsuite::qp::Results<T>& qpresults,
		proxsuite::qp::dense::Workspace<T>& qpwork,
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		T& primal_feasibility_eq_lhs,
		T& primal_feasibility_in_lhs) {

	qpwork.primal_residual_eq_scaled.noalias() = qpwork.A_scaled * qpresults.x;
	qpwork.primal_residual_in_scaled_up.noalias() = qpwork.C_scaled * qpresults.x;

	qpwork.ruiz.unscale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
	primal_feasibility_eq_rhs_0 = infty_norm(qpwork.primal_residual_eq_scaled);
	qpwork.ruiz.unscale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_in_scaled_up});
	primal_feasibility_in_rhs_0 = infty_norm(qpwork.primal_residual_in_scaled_up);

	qpwork.primal_residual_in_scaled_low =
			positive_part(qpwork.primal_residual_in_scaled_up - qpmodel.u) +
			negative_part(qpwork.primal_residual_in_scaled_up - qpmodel.l);
	qpwork.primal_residual_eq_scaled -= qpmodel.b;

	primal_feasibility_in_lhs = infty_norm(qpwork.primal_residual_in_scaled_low);
	primal_feasibility_eq_lhs = infty_norm(qpwork.primal_residual_eq_scaled);
	primal_feasibility_lhs =
			std::max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	qpwork.ruiz.scale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
}

// The problem is primal infeasible if the following four conditions hold:
//
// ||unscaled(A^Tdy)|| <= eps_p_inf ||unscaled(dy)||
// b^T dy <= -eps_p_inf ||unscaled(dy)||
// ||unscaled(C^Tdz)|| <= eps_p_inf ||unscaled(dz)||
// u^T [dz]_+ - l^T[-dz]_+ <= -eps_p_inf ||unscaled(dz)||
//
// the variables in entry are changed in place

template <typename T>
bool global_primal_residual_infeasibility(
		::qp::VectorViewMut<T> ATdy,
		::qp::VectorViewMut<T> CTdz,
		::qp::VectorViewMut<T> dy,
		::qp::VectorViewMut<T> dz,
		Workspace<T>& qpwork,
		const Settings<T>& qpsettings) {

	qpwork.ruiz.unscale_dual_residual_in_place(ATdy);
	qpwork.ruiz.unscale_dual_residual_in_place(CTdz);
	T eq_inf = dy.to_eigen().dot(qpwork.b_scaled);
	T in_inf = positive_part(dz.to_eigen()).dot(qpwork.u_scaled) -
	           positive_part(-dz.to_eigen()).dot(qpwork.l_scaled);
	qpwork.ruiz.unscale_dual_in_place_eq(dy);
	qpwork.ruiz.unscale_dual_in_place_in(dz);

	T bound_y = qpsettings.eps_primal_inf * infty_norm(dy.to_eigen());
	T bound_z = qpsettings.eps_primal_inf * infty_norm(dz.to_eigen());

	bool res = infty_norm(ATdy.to_eigen()) <= bound_y && eq_inf <= -bound_y &&
	           infty_norm(CTdz.to_eigen()) <= bound_z && in_inf <= -bound_z;
	return res;
}

// The problem is dual infeasible if one of the conditions hold:
//
// FIRST
// ||unscaled(Adx)|| <= eps_d_inf ||unscaled(dx)||
// unscaled(Cdx)_i \in [-eps_d_inf,eps_d_inf] ||unscaled(dx)|| if u_i and l_i
// are finite 					or >= -eps_d_inf||unscaled(dx)|| if u_i = +inf 					or <=
// eps_d_inf||unscaled(dx)|| if l_i = -inf
//
// SECOND
//
// ||unscaled(Hdx)|| <= c eps_d_inf * ||unscaled(dx)||  and  q^Tdx <= -c
// eps_d_inf  ||unscaled(dx)|| or dx^THdx <= -c eps_d_inf^2 dx the variables in
// entry are changed in place

template <typename T>
bool global_dual_residual_infeasibility(
		::qp::VectorViewMut<T> Adx,
		::qp::VectorViewMut<T> Cdx,
		::qp::VectorViewMut<T> Hdx,
		::qp::VectorViewMut<T> dx,
		Workspace<T>& qpwork,
		const Settings<T>& qpsettings,
		const Data<T>& qpmodel) {

	T dxHdx = (dx.to_eigen()).dot(Hdx.to_eigen());
	qpwork.ruiz.unscale_dual_residual_in_place(Hdx);
	qpwork.ruiz.unscale_primal_residual_in_place_eq(Adx);
	qpwork.ruiz.unscale_primal_residual_in_place_in(Cdx);
	T gdx = (dx.to_eigen()).dot(qpwork.g_scaled);
	qpwork.ruiz.unscale_primal_in_place(dx);

	T bound = infty_norm(dx.to_eigen()) * qpsettings.eps_dual_inf;
	T bound_neg = -bound;

	bool first_cond = infty_norm(Adx.to_eigen()) <= bound;

	for (i64 iter = 0; iter < qpmodel.n_in; ++iter) {
		T Cdx_i = Cdx.to_eigen()[iter];
		if (qpwork.u_scaled[iter] <= 1.E20 && qpwork.l_scaled[iter] >= -1.E20) {
			first_cond = first_cond && Cdx_i <= bound && Cdx_i >= bound_neg;
		} else if (qpwork.u_scaled[iter] > 1.E20) {
			first_cond = first_cond && Cdx_i >= bound_neg;
		} else if (qpwork.l_scaled[iter] < -1.E20) {
			first_cond = first_cond && Cdx_i <= bound;
		}
	}

	bound *= qpwork.ruiz.c;
	bound_neg *= qpwork.ruiz.c;
	bool second_cond_alt1 =
			infty_norm(Hdx.to_eigen()) <= bound && gdx <= bound_neg;
	bound_neg *= qpsettings.eps_dual_inf;
	bool second_cond_alt2 = dxHdx <= bound_neg;

	bool res = first_cond && (second_cond_alt1 || second_cond_alt2);
	return res;
}

// dual_feasibility_lhs = norm(dual_residual_scaled)
// dual_feasibility_rhs_0 = norm(unscaled(Hx))
// dual_feasibility_rhs_1 = norm(unscaled(ATy))
// dual_feasibility_rhs_3 = norm(unscaled(CTz))
//
// dual_residual_scaled = scaled(Hx + g + ATy + CTz)
template <typename T>
void global_dual_residual(
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3) {

	qpwork.dual_residual_scaled = qpwork.g_scaled;
	qpwork.CTz.noalias() =
			qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * qpresults.x;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_0 = infty_norm(qpwork.CTz);
	qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * qpresults.y;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_1 = infty_norm(qpwork.CTz);

	qpwork.CTz.noalias() = qpwork.C_scaled.transpose() * qpresults.z;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_3 = infty_norm(qpwork.CTz);

	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.dual_residual_scaled});

	dual_feasibility_lhs = infty_norm(qpwork.dual_residual_scaled);

	qpwork.ruiz.scale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.dual_residual_scaled});
}

} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_INCLUDE_QP_DENSE_UTILS_HPP */
