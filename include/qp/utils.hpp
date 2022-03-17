#ifndef INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS

//#include "ldlt/views.hpp"
//#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
//#include "ldlt/factorize.hpp"
//#include "ldlt/detail/meta.hpp"
//#include "ldlt/solve.hpp"
//#include "ldlt/update.hpp"
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

/*
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
*/
template <typename T>
auto square(T const& expr)
	LDLT_DEDUCE_RET(expr*expr);


template <typename T>
void refactorize(
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& qpresults,
		qp::QPData<T>& qpmodel,
		T rho_new
		) {

	qpwork._kkt.diagonal().array() += rho_new - qpresults._rho;
	qpwork._ldl.factorize(qpwork._kkt);

	if (qpresults._n_c == 0) {
		return;
	}

	for (isize j = 0; j < qpresults._n_c; ++j) {
		for (isize i = 0; i < qpmodel._n_in; ++i) {
			if (j == qpwork._current_bijection_map(i)) {
				qpwork._dw_aug.head(qpmodel._dim) = qpwork._c_scaled.row(i);
				qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + j) = - qpresults._mu_in; // mu_in stores the inverse of mu_in
				qpwork._ldl.insert_at(qpmodel._n_eq + qpmodel._dim + j, qpwork._dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c));
				qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + j) = T(0);
			}
		}
	}
}


template <typename T>
void mu_update(
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& qpresults,
		qp::QPData<T>& qpmodel,
		T mu_eq_new_inv,
		T mu_in_new_inv) {
	T diff = T(0);

	qpwork._dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c).setZero();
	if (qpmodel._n_eq > 0) {
		diff = qpresults._mu_eq_inv -  mu_eq_new_inv; // mu stores the inverse of mu

		for (isize i = 0; i < qpmodel._n_eq; i++) {
			qpwork._dw_aug(qpmodel._dim + i) = T(1);
			qpwork._ldl.rank_one_update(qpwork._dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c), diff);
			qpwork._dw_aug(qpmodel._dim + i) = T(0);
		}
	}
	if (qpresults._n_c > 0) {
		diff = qpresults._mu_in_inv - mu_in_new_inv; // mu stores the inverse of mu
		for (isize i = 0; i < qpresults._n_c; i++) {
			qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + i) = T(1);
			qpwork._ldl.rank_one_update(qpwork._dw_aug.head(qpmodel._dim+qpmodel._n_eq+qpresults._n_c), diff);
			qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + i) = T(0);
		}
	}
}


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
