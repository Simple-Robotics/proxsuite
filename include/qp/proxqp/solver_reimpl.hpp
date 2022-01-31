#ifndef INRIA_LDLT_SOLVER_REIMPL_HPP_JUE9ZFA0S
#define INRIA_LDLT_SOLVER_REIMPL_HPP_JUE9ZFA0S

#include <ldlt/ldlt.hpp>
#include <veg/vec.hpp>
#include <veg/memory/dynamic_stack.hpp>
#include <qp/views.hpp>

namespace qp {
using ldlt::from_eigen;

namespace detail {

template <typename T>
auto positive_part(T const& expr)
		VEG_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		VEG_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));

} // namespace detail

using veg::Ref;
using veg::ref;
using veg::RefMut;
using veg::mut;

template <typename T>
struct SolverParams {
	T rho;

	T mu_eq_inv;
	T mu_in_inv;
	T mu_f;

	T bcl_alpha;
	T bcl_beta;

	T eps_int;
	T eps_ext;
	T eps_abs;
	T eps_rel;

	T dual_feasibility_fast_dual_threshold;
	T rho_fast_dual;

	i64 k_max;
	i64 max_iter;
};

template <typename T>
struct SolverResult {
	i64 n_ext;
};

#define LDLT_TO_EIGEN(X) auto X /* NOLINT */ = (X##_.to_eigen())

namespace detail {
template <typename T>
struct PrimalFeasibility {
	T lhs;
	T eq_rhs_0;
	T in_rhs_0;
	T eq_lhs;
	T in_lhs;
};

template <typename T>
struct DualFeasibility {
	T lhs;
	T rhs_0;
	T rhs_1;
	T rhs_3;
};

template <typename T, typename P>
auto global_primal_residual(
		ldlt::VectorViewMut<T> primal_residual_eq_scaled_,
		ldlt::VectorViewMut<T> primal_residual_in_scaled_hi_,
		ldlt::VectorViewMut<T> primal_residual_in_scaled_lo_,

		ldlt::VectorView<T> x_,
		QpViewBox<T> qp,
		QpViewBox<T> qp_scaled,
		Ref<P> precond) -> PrimalFeasibility<T> {

	LDLT_TO_EIGEN(x);
	LDLT_TO_EIGEN(primal_residual_eq_scaled);
	LDLT_TO_EIGEN(primal_residual_in_scaled_hi);
	LDLT_TO_EIGEN(primal_residual_in_scaled_lo);

	{
		primal_residual_eq_scaled.noalias() = qp_scaled.A.to_eigen() * x;
		primal_residual_in_scaled_hi.noalias() = qp_scaled.C.to_eigen() * x;

		precond->unscale_primal_residual_in_place_eq(VectorViewMut<T>{
				from_eigen,
				primal_residual_eq_scaled,
		});
		T eq_rhs_0 = infty_norm(primal_residual_eq_scaled);
		precond->unscale_primal_residual_in_place_in(VectorViewMut<T>{
				from_eigen,
				primal_residual_in_scaled_hi,
		});
		T in_rhs_0 = infty_norm(primal_residual_in_scaled_hi);
		primal_residual_in_scaled_lo =
				detail::positive_part(primal_residual_in_scaled_hi - qp.u) +
				detail::negative_part(primal_residual_in_scaled_hi - qp.l);

		T in_lhs = infty_norm(primal_residual_in_scaled_lo);
		T eq_lhs = infty_norm(primal_residual_eq_scaled);
		T lhs = max2( //
				eq_lhs,
				in_lhs);
		precond->scale_primal_residual_in_place_eq(VectorViewMut<T>{
				from_eigen,
				primal_residual_eq_scaled,
		});

		return {lhs, eq_rhs_0, in_rhs_0, eq_lhs, in_lhs};
	}
}

template <typename T, typename P>
auto global_dual_residual(
		ldlt::VectorViewMut<T> dual_residual_scaled_,

		ldlt::VectorView<T> x_,
		ldlt::VectorView<T> y_,
		ldlt::VectorView<T> z_,

		QpViewBox<T> qp_scaled,
		Ref<P> precond,
		veg::dynstack::DynStackMut stack) -> DualFeasibility<T> {
	LDLT_TO_EIGEN(dual_residual_scaled);
	LDLT_TO_EIGEN(x);
	LDLT_TO_EIGEN(y);
	LDLT_TO_EIGEN(z);

	dual_residual_scaled = qp_scaled.g.to_eigen();
	auto rhs_0 = [&]() -> T {
		LDLT_TEMP_VEC(T, Hx, x.rows(), stack);
		Hx.noalias() += qp_scaled.H.to_eigen() * x;
		dual_residual_scaled += Hx;
		precond->unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, Hx});
		return infty_norm(Hx);
	}();

	auto rhs_1 = [&]() -> T {
		LDLT_TEMP_VEC(T, ATy, x.rows(), stack);
		ATy.noalias() += qp_scaled.A.trans().to_eigen() * y;
		dual_residual_scaled += ATy;
		precond->unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, ATy});
		return infty_norm(ATy);
	}();

	auto rhs_3 = [&]() -> T {
		LDLT_TEMP_VEC(T, CTz, x.rows(), stack);
		CTz.noalias() += qp_scaled.C.trans().to_eigen() * z;
		dual_residual_scaled += CTz;
		precond->unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, CTz});
		return infty_norm(CTz);
	}();

	precond->unscale_dual_residual_in_place(VectorViewMut<T>{
			from_eigen,
			dual_residual_scaled,
	});
	T lhs = infty_norm(dual_residual_scaled);
	precond->scale_dual_residual_in_place(VectorViewMut<T>{
			from_eigen,
			dual_residual_scaled,
	});
	return {lhs, rhs_0, rhs_1, rhs_3};
}

template <typename T>
void refactor_ldlt( //
		ldlt::MatrixViewMut<T, colmajor> kkt_,
		RefMut<ldlt::Ldlt<T>> ldl,
		QpViewBox<T> qp,
		SolverParams<T> params,
		T rho_new,
		veg::Slice<isize> ineq_mapping,
		veg::dynstack::DynStackMut stack) {

	LDLT_TO_EIGEN(kkt);

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_c = ldl->dim();

	kkt.diagonal().head(dim).array() += rho_new - params.rho;
	kkt.diagonal().segment(dim, n_eq).array() -= params.mu_eq_inv;

	ldl->factor(kkt, LDLT_FWD(stack));

	for (isize j = 0; j < n_c; ++j) {
		for (isize i = 0; i < n_in; ++i) {
			if (j == ineq_mapping[i]) {
			}
		}
	}
}
} // namespace detail

template <typename T, typename P>
auto solve_qp(
		ldlt::VectorViewMut<T> x_,
		ldlt::VectorViewMut<T> y_,
		ldlt::VectorViewMut<T> z_,
		QpViewBox<T> qp,
		QpViewBox<T> qp_scaled,
		SolverParams<T> params,
		Ref<P> precond,
		veg::dynstack::DynStackMut stack) -> SolverResult<T> {
	using detail::max2;

	SolverResult<T> result;

	isize n = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;

	LDLT_TEMP_VEC_UNINIT(T, primal_residual_eq_scaled, n_eq, stack);
	LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled_hi, n_in, stack);
	LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled_lo, n_in, stack);

	LDLT_TEMP_VEC_UNINIT(T, dual_residual_scaled, n, stack);

	auto x = x_.to_eigen();
	auto y = y_.to_eigen();
	auto z = z_.to_eigen();

	T primal_feasibility_rhs_1_eq =
			(params.eps_rel == 0) ? T(0) : infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u =
			(params.eps_rel == 0) ? T(0) : infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l =
			(params.eps_rel == 0) ? T(0) : infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 =
			(params.eps_rel == 0) ? T(0) : infty_norm(qp.g.to_eigen());

	using namespace detail;

	for (i64 iter = 0; iter < params.max_iter; ++iter) {
		++result.n_ext;
		PrimalFeasibility<T> primal_feasibility = detail::global_primal_residual(
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_hi},
				{from_eigen, primal_residual_in_scaled_lo},
				x_.as_const(),
				qp,
				qp_scaled,
				precond);
		DualFeasibility<T> dual_feasibility = detail::global_dual_residual(
				{from_eigen, dual_residual_scaled},
				x_.as_const(),
				y_.as_const(),
				z_.as_const(),
				qp_scaled,
				precond,
				LDLT_FWD(stack));

		T rhs_primal(params.eps_abs);
		T rhs_dual(params.eps_abs);

		if (params.eps_rel != 0) {
			rhs_primal += params.eps_rel * max_list({
																				 primal_feasibility.eq_rhs_0,
																				 primal_feasibility.in_rhs_0,
																				 primal_feasibility_rhs_1_eq,
																				 primal_feasibility_rhs_1_in_u,
																				 primal_feasibility_rhs_1_in_l,
																		 });

			rhs_dual += params.eps_rel * max_list({
																			 dual_feasibility.rhs_3,
																			 dual_feasibility.rhs_0,
																			 dual_feasibility.rhs_1,
																			 dual_feasibility_rhs_2,
																	 });
		}

		bool is_primal_feasible = primal_feasibility.lhs <= rhs_primal;
		bool is_dual_feasible = dual_feasibility.lhs <= rhs_dual;

		if (is_primal_feasible) {
			if (dual_feasibility.lhs >= params.dual_feasibility_fast_dual_threshold &&
			    params.rho != params.rho_fast_dual) {
				T rho_new = params.rho_fast_dual;

				params.rho = rho_new;
			}

			if (is_dual_feasible) {
			}
		}
	}
}

} // namespace qp

#endif /* end of include guard INRIA_LDLT_SOLVER_REIMPL_HPP_JUE9ZFA0S */
