#ifndef INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS

#include "qp/views.hpp"
#include <qp/QPWorkspace.hpp>
#include <qp/QPData.hpp>
#include <qp/QPResults.hpp>
#include <cmath>
#include <type_traits>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

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

template <
		typename T>
void global_primal_residual(
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& qpresults,
		qp::QPData<T>& qpmodel
		) {

	//LDLT_DECL_SCOPE_TIMER("in solver", "primal residual", T);

	qpwork._primal_residual_eq_scaled.noalias() = qpwork._a_scaled * qpresults._x;
	qpwork._primal_residual_in_scaled_u.noalias() = qpwork._c_scaled * qpresults._x;

	qpwork._ruiz.unscale_primal_residual_in_place_eq(VectorViewMut<T>{from_eigen,qpwork._primal_residual_eq_scaled});
	primal_feasibility_eq_rhs_0 = infty_norm( qpwork._primal_residual_eq_scaled);
	qpwork._ruiz.unscale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen,qpwork._primal_residual_in_scaled_u});
	primal_feasibility_in_rhs_0 = infty_norm( qpwork._primal_residual_in_scaled_u);


	qpwork._primal_residual_eq_scaled -= qpmodel._b;
	qpwork._primal_residual_in_scaled_l =
			detail::positive_part(qpwork._primal_residual_in_scaled_u - qpmodel._u) +
			detail::negative_part(qpwork._primal_residual_in_scaled_u - qpmodel._l);
	primal_feasibility_lhs = max2(
			infty_norm(qpwork._primal_residual_in_scaled_l),
			infty_norm(qpwork._primal_residual_eq_scaled));
	qpwork._ruiz.scale_primal_residual_in_place_eq(VectorViewMut<T>{from_eigen,qpwork._primal_residual_eq_scaled});
}

// dual_feasibility_lhs = norm(dual_residual_scaled)
// dual_feasibility_rhs_0 = norm(unscaled(H×x))
// dual_feasibility_rhs_1 = norm(unscaled(AT×y))
// dual_feasibility_rhs_3 = norm(unscaled(CT×z))
//
//
// dual_residual_scaled = scaled(H×x + g + AT×y + CT×z)
// dw_aug = indeterminate

template <typename T>
void global_dual_residual(
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& qpresults
		) {

	/*
	* dual_residual_scaled = scaled(Hx+g+ATy+CTz)
	* dual_feasibility_lhs =  ||unscaled(Hx+g+ATy+CTz)||
	* dual_feasibility_rhs_0 = ||unscaled(Hx)||
	* dual_feasibility_rhs_1 = ||unscaled(ATy)||
	* dual_feasibility_rhs_3 =  ||unscaled(CTz)||
	*/

	qpwork._dual_residual_scaled = qpwork._g_scaled;
	qpwork._Hx.noalias() = qpwork._h_scaled * qpresults._x;

	qpwork._dual_residual_scaled += qpwork._Hx;
	qpwork._ruiz.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
	dual_feasibility_rhs_0 = infty_norm(qpwork._Hx);

	qpwork._ATy.noalias() = qpwork._a_scaled.transpose() * qpresults._y;
	qpwork._dual_residual_scaled += qpwork._ATy;
	qpwork._ruiz.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._ATy});
	dual_feasibility_rhs_1 = infty_norm(qpwork._ATy);

	qpwork._CTz.noalias() = qpwork._c_scaled.transpose() * qpresults._z;
	qpwork._dual_residual_scaled += qpwork._CTz;
	qpwork._ruiz.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._CTz});
	dual_feasibility_rhs_3 = infty_norm(qpwork._CTz);

	qpwork._ruiz.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._dual_residual_scaled});

	dual_feasibility_lhs = infty_norm(qpwork._dual_residual_scaled);

}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS */
