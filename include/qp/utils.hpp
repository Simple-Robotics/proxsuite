#ifndef INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include <qp/QPData.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
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
auto positive_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto square(T const& expr)
	LDLT_DEDUCE_RET(expr*expr);


template <typename T, Layout L>
void refactorize(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		MatrixViewMut<T, L> kkt,
		isize n_c,
		T mu_in,
		T rho_old,
		T rho_new,
		ldlt::Ldlt<T>& ldl) {
	auto current_bijection_map = current_bijection_map_.to_eigen();

	isize const dim = qp_scaled.H.rows;
	isize const n_eq = qp_scaled.A.rows;
	isize const n_in = qp_scaled.C.rows;

	auto Htot = kkt.to_eigen();

	Htot.diagonal().array() += rho_new - rho_old; 
	ldl.factorize(Htot);

	if (n_c == 0) {
		return;
	}

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_new_col, Init, Vec(n_eq + dim + n_c), LDLT_CACHELINE_BYTES, T));

	for (isize j = 0; j < n_c; ++j) {
		for (isize i = 0; i < n_in; ++i) {
			if (j == current_bijection_map(i)) {
				auto new_col = _new_col.segment(0, dim + n_eq + j + 1).to_eigen();
				new_col.head(dim) = qp_scaled.C.to_eigen().row(i);
				new_col(dim + n_eq + j) = -T(1) / mu_in;
				ldl.insert_at(n_eq + dim + j, new_col);
				new_col(dim + n_eq + j) = T(0);
			}
		}
	}
}


template <typename T>
void mu_update(
		qp::Qpdata<T>& qpdata,
		T mu_eq_old,
		T mu_eq_new,
		T mu_in_old,
		T mu_in_new,
		isize dim,
		isize n_eq,
		isize n_c,
		ldlt::Ldlt<T>& ldl) {

	qpdata._dw_aug.head(dim+n_eq+n_c).setZero();
	if (n_eq > 0) {
		T diff = T(1) / mu_eq_old - T(1) / mu_eq_new;

		for (isize i = 0; i < n_eq; i++) {
			qpdata._dw_aug(dim + i) = T(1);
			ldl.rank_one_update(qpdata._dw_aug.head(dim+n_eq+n_c), diff);
			qpdata._dw_aug(dim + i) = T(0);
		}
	}
	if (n_c > 0) {
		T diff = T(1) / mu_in_old - T(1) / mu_in_new;
		for (isize i = 0; i < n_c; i++) {
			qpdata._dw_aug(dim + n_eq + i) = T(1);
			ldl.rank_one_update(qpdata._dw_aug.head(dim+n_eq+n_c), diff);
			qpdata._dw_aug(dim + n_eq + i) = T(0);
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
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
void global_primal_residual(
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		qp::Qpdata<T>& qpdata,
		qp::QpViewBox<T> qp,
		qp::QpViewBox<T> qp_scaled,
		Preconditioner& precond
		) {

	//LDLT_DECL_SCOPE_TIMER("in solver", "primal residual", T);
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto b_ = qp_scaled.b.to_eigen();

	qpdata._primal_residual_eq_scaled.noalias() = A_ * qpdata._x;
	qpdata._primal_residual_in_scaled_u.noalias() = C_ * qpdata._x;

	precond.unscale_primal_residual_in_place_eq(VectorViewMut<T>{from_eigen,qpdata._primal_residual_eq_scaled});
	primal_feasibility_eq_rhs_0 = infty_norm( qpdata._primal_residual_eq_scaled);
	precond.unscale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen,qpdata._primal_residual_in_scaled_u});
	primal_feasibility_in_rhs_0 = infty_norm( qpdata._primal_residual_in_scaled_u);

	
	qpdata._primal_residual_eq_scaled -= qp.b.to_eigen();
	qpdata._primal_residual_in_scaled_l =
			detail::positive_part(qpdata._primal_residual_in_scaled_u - qp.u.to_eigen()) +
			detail::negative_part(qpdata._primal_residual_in_scaled_u - qp.l.to_eigen());

	primal_feasibility_lhs = max2(
			infty_norm(qpdata._primal_residual_in_scaled_l),
			infty_norm(qpdata._primal_residual_eq_scaled));
	precond.scale_primal_residual_in_place_eq(VectorViewMut<T>{from_eigen,qpdata._primal_residual_eq_scaled});
}

// dual_feasibility_lhs = norm(dual_residual_scaled)
// dual_feasibility_rhs_0 = norm(unscaled(H×x))
// dual_feasibility_rhs_1 = norm(unscaled(AT×y))
// dual_feasibility_rhs_3 = norm(unscaled(CT×z))
//
//
// dual_residual_scaled = scaled(H×x + g + AT×y + CT×z)
// dw_aug = indeterminate

template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
void global_dual_residual(
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3,
		qp::Qpdata<T>& qpdata,
		qp::QpViewBox<T> qp_scaled,
		Preconditioner& precond,
		VectorView<T> x,
		VectorView<T> y,
		VectorView<T> z) {

	/*
	* dual_residual_scaled = scaled(Hx+g+ATy+CTz)
	* dw_aug = tmp variable
	* dual_feasibility_lhs =  ||unscaled(Hx+g+ATy+CTz)||
	* dual_feasibility_rhs_0 = ||unscled(Hx)||
	* dual_feasibility_rhs_1 = ||unscaled(ATy)||
	* dual_feasibility_rhs_3 =  ||unscaled(CTz)||
	*/
	//LDLT_DECL_SCOPE_TIMER("in solver", "dual residual", T);
	auto H_ = qp_scaled.H.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_ = z.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();

	isize const dim = qp_scaled.H.rows;

	qpdata._dual_residual_scaled = g_;
	qpdata._Hx.noalias() = H_ * qpdata._x;
	qpdata._dual_residual_scaled += qpdata._Hx;
	precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._Hx});
	dual_feasibility_rhs_0 = infty_norm(qpdata._Hx);

	qpdata._ATy.noalias() = A_.transpose() * y_;
	qpdata._dual_residual_scaled += qpdata._ATy;
	precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._ATy});
	dual_feasibility_rhs_1 = infty_norm(qpdata._ATy);

	qpdata._CTz.noalias() = C_.transpose() * z_;
	qpdata._dual_residual_scaled += qpdata._CTz;
	precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._CTz});
	dual_feasibility_rhs_3 = infty_norm(qpdata._CTz);

	precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._dual_residual_scaled});

	dual_feasibility_lhs = infty_norm(qpdata._dual_residual_scaled);

	precond.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._dual_residual_scaled});
	precond.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._Hx});
	precond.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._ATy});
	precond.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpdata._CTz});
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_UTILS_SOLVER_HPP_HDWGZKCLS */