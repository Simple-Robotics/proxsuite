#ifndef INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include <cmath>

namespace qp {
namespace preconditioner {
struct IdentityPrecond {
	template <typename T>
	void scale_qp_in_place(QpViewMut<T> /*qp*/) const noexcept {}

	template <typename T>
	void scale_primal_in_place(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void scale_dual_in_place(VectorViewMut<T> /*y*/) const noexcept {}

	template <typename T>
	void scale_primal_residual_in_place(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void scale_dual_residual_in_place(VectorViewMut<T> /*y*/) const noexcept {}

	template <typename T>
	void unscale_primal_in_place(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void unscale_dual_in_place(VectorViewMut<T> /*y*/) const noexcept {}

	template <typename T>
	void unscale_primal_residual_in_place(VectorViewMut<T> /*x*/) const noexcept {
	}
	template <typename T>
	void unscale_dual_residual_in_place(VectorViewMut<T> /*y*/) const noexcept {}
};
} // namespace preconditioner

namespace detail {

template <typename T>
struct TypeIdentityImpl {
	using Type = T;
};

template <typename T>
using DoNotDeduce = typename TypeIdentityImpl<T>::Type;

template <typename Dst, typename Lhs, typename Rhs>
void mul_add_no_alias(Dst& dst, Lhs const& lhs, Rhs const& rhs) {
	dst.noalias().operator+=(lhs.operator*(rhs));
}
template <typename Dst, typename Lhs, typename Rhs>
void mul_no_alias(Dst& dst, Lhs const& lhs, Rhs const& rhs) {
	dst.setZero();
	mul_add_no_alias(dst, lhs, rhs);
}

struct EqSolverTimer {};
struct QpSolveStats {
	i64 n_iters;
	i64 n_mu_updates;
};

template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
auto solve_qp( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		qp::QpView<T> qp,
		i64 max_iter,
		DoNotDeduce<T> eps_abs,
		DoNotDeduce<T> eps_rel,
		Preconditioner precond = Preconditioner{}) -> QpSolveStats {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	i64 n_mu_updates = 0;

	auto rho = T(1e-10);
	auto bcl_mu = T(1e3);
	T bcl_eta = 1 / pow(bcl_mu, T(0.1));

	LDLT_MULTI_WORKSPACE_MEMORY(
			((_h_scaled, Mat(dim, dim)),
	     (_g_scaled, Vec(dim)),
	     (_a_scaled, Mat(n_eq, dim)),
	     (_b_scaled, Vec(n_eq)),
	     (_htot, Mat((dim + n_eq), (dim + n_eq))),
	     (_d, Vec(dim + n_eq)),
	     (_residual_scaled, Vec(dim + n_eq)),
	     (_residual_scaled_tmp, Vec(dim + n_eq)),
	     (_residual_unscaled, Vec(dim + n_eq)),
	     (_next_dual, Vec(n_eq)),
	     (_diag_diff, Vec(n_eq))),
			T);

	auto H_copy = _h_scaled.to_eigen();
	auto q_copy = _g_scaled.to_eigen();
	auto A_copy = _a_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();

	H_copy = qp.H.to_eigen();
	q_copy = qp.g.to_eigen();
	A_copy = qp.A.to_eigen();
	b_copy = qp.b.to_eigen();

	auto qp_scaled = qp::QpViewMut<T>{
			{from_eigen, H_copy},
			{from_eigen, q_copy},
			{from_eigen, A_copy},
			{from_eigen, b_copy},

			// no inequalities
			{from_ptr_rows_cols_stride, nullptr, 0, dim, 0},
			{from_ptr_size, nullptr, 0},
	};
	{
		LDLT_DECL_SCOPE_TIMER("eq solver", "scale qp", T);
		precond.scale_qp_in_place(qp_scaled);
	}
	{
		LDLT_DECL_SCOPE_TIMER("eq solver", "scale solution", T);
		precond.scale_primal_in_place(x);
		precond.scale_dual_in_place(y);
	}

	auto Htot = _htot.to_eigen();
	auto d = _d.to_eigen();

	{
		LDLT_DECL_SCOPE_TIMER("eq solver", "set H", T);
		Htot.topLeftCorner(dim, dim) = qp_scaled.H.as_const().to_eigen();
		for (isize i = 0; i < dim; ++i) {
			Htot(i, i) += rho;
		}

		Htot.topRightCorner(dim, n_eq) = qp_scaled.A.as_const().trans().to_eigen();

		// TODO: unneeded: see "ldlt/factorize.hpp"
		Htot.bottomLeftCorner(n_eq, dim) = qp_scaled.A.as_const().to_eigen();
		Htot.bottomRightCorner(n_eq, n_eq).setZero();
		{
			T tmp = -T(1) / bcl_mu;
			for (isize i = 0; i < n_eq; ++i) {
				Htot(dim + i, dim + i) = tmp;
			}
		}
	}

	auto ldlt_mut = LdltViewMut<T>{
			{from_eigen, Htot},
			{from_eigen, d},
	};

	// initial LDLT factorization
	{
		LDLT_DECL_SCOPE_TIMER("eq solver", "factorization", T);
		ldlt::factorize(
				ldlt_mut,
				MatrixView<T, colmajor>{from_eigen, Htot},
				ldlt::factorization_strategy::standard);
	}

	auto residual_scaled = _residual_scaled.to_eigen();
	auto residual_scaled_tmp = _residual_scaled_tmp.to_eigen();
	auto residual_unscaled = _residual_unscaled.to_eigen();

	auto next_dual = _next_dual.to_eigen();
	auto diag_diff = _diag_diff.to_eigen();

	T primal_feasibility_rhs_1 = infty_norm(qp.b.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	for (i64 iter = 0; iter <= max_iter; ++iter) {

		auto dual_residual_scaled = residual_scaled.topRows(dim);
		auto primal_residual_scaled = residual_scaled.bottomRows(n_eq);
		auto dual_residual_unscaled = residual_unscaled.topRows(dim);
		auto primal_residual_unscaled = residual_unscaled.bottomRows(n_eq);

		// compute primal residual
		T primal_feasibility_rhs_0(0);
		T dual_feasibility_rhs_0(0);
		T dual_feasibility_rhs_1(0);

		T primal_feasibility_lhs(0);
		T dual_feasibility_lhs(0);
		{
			{
				LDLT_DECL_SCOPE_TIMER("eq solver", "primal residual", T);
				auto A_ = qp_scaled.A.as_const().to_eigen();
				auto x_ = x.as_const().to_eigen();
				auto b_ = qp_scaled.b.as_const().to_eigen();

				// A×x - b
				primal_residual_scaled.setZero();
				primal_residual_scaled.noalias() += A_ * x_;

				{
					auto w = residual_scaled_tmp.bottomRows(n_eq);
					w = primal_residual_scaled;
					precond.unscale_primal_residual_in_place(
							VectorViewMut<T>{from_eigen, w});
					primal_feasibility_rhs_0 = infty_norm(w);
				}
				primal_residual_scaled -= b_;

				primal_residual_unscaled = primal_residual_scaled;
				precond.unscale_primal_residual_in_place(
						VectorViewMut<T>{from_eigen, primal_residual_unscaled});

				primal_feasibility_lhs = infty_norm(primal_residual_unscaled);
			}

			if (iter > 0) {
				if (primal_feasibility_lhs <= bcl_eta) {
					y.to_eigen() = next_dual;
					bcl_eta = bcl_eta / pow(bcl_mu, T(0.9));
				} else {
					T new_bcl_mu = max2(bcl_mu * T(10), T(1e12));
					if (bcl_mu != new_bcl_mu) {
						{
							LDLT_DECL_SCOPE_TIMER("eq solver", "mu update", T);
							diag_diff.setConstant(T(1) / bcl_mu - T(1) / new_bcl_mu);
							ldlt::diagonal_update(
									ldlt_mut,
									ldlt_mut.as_const(),
									{from_eigen, diag_diff},
									dim,
									ldlt::diagonal_update_strategies::multi_pass);
							++n_mu_updates;
						}
					}
					bcl_mu = new_bcl_mu;
					bcl_eta = T(1) / pow(bcl_mu, T(0.1));
				}
			}
			if (iter == max_iter) {
				break;
			}
		}

		bool is_primal_feasible =
				primal_feasibility_lhs <=
				(eps_abs + eps_rel * max2( //
																 primal_feasibility_rhs_0,
																 primal_feasibility_rhs_1));

		// compute dual residual
		{
			LDLT_DECL_SCOPE_TIMER("eq solver", "dual residual", T);
			auto H_ = qp_scaled.H.as_const().to_eigen();
			auto A_ = qp_scaled.A.as_const().to_eigen();
			auto x_ = x.as_const().to_eigen();
			auto y_ = y.as_const().to_eigen();
			auto g_ = qp_scaled.g.as_const().to_eigen();

			// H×x + g + A.T×y

			// TODO(2): if TODO(1) is applied, update dual_residual_scaled before
			// newton step
			dual_residual_scaled = g_;
			{
				auto w = residual_scaled_tmp.topRows(dim);

				w.setZero();
				w.noalias() += H_ * x_;
				{ dual_residual_scaled += w; }
				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, w});
				dual_feasibility_rhs_0 = infty_norm(w);

				w.setZero();
				w.noalias() += A_.transpose() * y_;
				{ dual_residual_scaled += w; }

				precond.unscale_dual_residual_in_place(VectorViewMut<T>{from_eigen, w});
				dual_feasibility_rhs_1 = infty_norm(w);
			}

			dual_residual_unscaled = dual_residual_scaled;
			precond.unscale_dual_residual_in_place(
					VectorViewMut<T>{from_eigen, dual_residual_unscaled});

			dual_feasibility_lhs = infty_norm(dual_residual_unscaled);
		}

		// TODO(1): always true for QP?
		bool is_dual_feasible =
				dual_feasibility_lhs <=
				(eps_abs + eps_rel * max2(
																 dual_feasibility_rhs_0,
																 max2( //
																		 dual_feasibility_rhs_1,
																		 dual_feasibility_rhs_2)));

		if (is_primal_feasible && is_dual_feasible) {
			{
				LDLT_DECL_SCOPE_TIMER("eq solver", "unscale solution", T);
				precond.unscale_primal_in_place(x);
				precond.unscale_dual_in_place(y);
			}
			return {iter, n_mu_updates};
		}

		// newton step
		{
			auto rhs = residual_scaled;

			rhs = -rhs;
			{
				LDLT_DECL_SCOPE_TIMER("eq solver", "newton step", T);
				ldlt::solve({from_eigen, rhs}, ldlt_mut.as_const(), {from_eigen, rhs});
			}

			x.to_eigen() += rhs.topRows(dim);
			next_dual = y.as_const().to_eigen() + rhs.bottomRows(n_eq);
		}
	}
	return {max_iter, n_mu_updates};
}
} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS */
