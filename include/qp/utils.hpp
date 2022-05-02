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

template <typename Derived>
void save_data(
		const std::string& filename, const Eigen::MatrixBase<Derived>& mat) {
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
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
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
			detail::positive_part(qpwork.primal_residual_in_scaled_up - qpmodel.u) +
			detail::negative_part(qpwork.primal_residual_in_scaled_up - qpmodel.l);
	qpwork.primal_residual_eq_scaled -= qpmodel.b;

	primal_feasibility_in_lhs = infty_norm(qpwork.primal_residual_in_scaled_low);
	primal_feasibility_eq_lhs = infty_norm(qpwork.primal_residual_eq_scaled);
	primal_feasibility_lhs =
			max2(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	qpwork.ruiz.scale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
}

// dual_feasibility_lhs = norm(dual_residual_scaled)
// dual_feasibility_rhs_0 = norm(unscaled(Hx))
// dual_feasibility_rhs_1 = norm(unscaled(ATy))
// dual_feasibility_rhs_3 = norm(unscaled(CTz))
//
// dual_residual_scaled = scaled(Hx + g + ATy + CTz)
template <typename T>
void global_dual_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
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
};

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS */
