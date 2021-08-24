#ifndef INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include "ldlt/qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include <cmath>

namespace qp {
namespace preconditioner {
struct IdentityPrecond {
	template <typename Scalar, Layout LH, Layout LC>
	void scale_qp_in_place(QpViewMut<Scalar, LH, LC> /*qp*/) const noexcept {}

	template <typename Scalar>
	void scale_primal_in_place(VectorViewMut<Scalar> /*x*/) const noexcept {}
	template <typename Scalar>
	void scale_dual_in_place(VectorViewMut<Scalar> /*y*/) const noexcept {}

	template <typename Scalar>
	void
	scale_primal_residue_in_place(VectorViewMut<Scalar> /*x*/) const noexcept {}
	template <typename Scalar>
	void scale_dual_residue_in_place(VectorViewMut<Scalar> /*y*/) const noexcept {
	}

	template <typename Scalar>
	void unscale_primal_in_place(VectorViewMut<Scalar> /*x*/) const noexcept {}
	template <typename Scalar>
	void unscale_dual_in_place(VectorViewMut<Scalar> /*y*/) const noexcept {}

	template <typename Scalar>
	void
	unscale_primal_residue_in_place(VectorViewMut<Scalar> /*x*/) const noexcept {}
	template <typename Scalar>
	void
	unscale_dual_residue_in_place(VectorViewMut<Scalar> /*y*/) const noexcept {}
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

template <
		typename Scalar,
		Layout LH,
		Layout LC,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
auto solve_qp( //
		VectorViewMut<Scalar> x,
		VectorViewMut<Scalar> y,
		qp::QpView<Scalar, LH, LC> qp,
		i32 max_iter,
		DoNotDeduce<Scalar> eps_abs,
		DoNotDeduce<Scalar> eps_rel,
		Preconditioner precond = Preconditioner{}) -> i32 {

	i32 dim = qp.H.rows;
	i32 n_eq = qp.A.rows;

	auto rho = Scalar(1e-10);
	auto bcl_mu = Scalar(1e2);
	Scalar bcl_eta = 1 / pow(bcl_mu, Scalar(0.1));

	LDLT_MULTI_WORKSPACE_MEMORY(
			((_h_scaled, dim * dim),
	     (_g_scaled, dim),
	     (_a_scaled, n_eq * dim),
	     (_b_scaled, n_eq),
	     (_htot, (dim + n_eq) * (dim + n_eq)),
	     (_d, dim + n_eq),
	     (_residue_scaled, dim + n_eq),
	     (_residue_scaled_tmp, dim + n_eq),
	     (_residue_unscaled, dim + n_eq),
	     (_next_dual, n_eq),
	     (_diag_diff, n_eq)),
			Scalar);

	auto H_copy = to_eigen_matrix_mut(
			MatrixViewMut<Scalar, colmajor>{_h_scaled, dim, dim, dim});
	auto q_copy = to_eigen_vector_mut(VectorViewMut<Scalar>{_g_scaled, dim});
	auto A_copy = to_eigen_matrix_mut(
			MatrixViewMut<Scalar, colmajor>{_a_scaled, n_eq, dim, n_eq});
	auto b_copy = to_eigen_vector_mut(VectorViewMut<Scalar>{_b_scaled, n_eq});

	H_copy = to_eigen_matrix(qp.H);
	q_copy = to_eigen_vector(qp.g);
	A_copy = to_eigen_matrix(qp.A);
	b_copy = to_eigen_vector(qp.b);

	auto qp_scaled = qp::QpViewMut<Scalar, LH, LC>{
			from_eigen_matrix_mut(H_copy),
			from_eigen_vector_mut(q_copy),
			from_eigen_matrix_mut(A_copy),
			from_eigen_vector_mut(b_copy),

			// no inequalities
			{nullptr, 0, dim, 0},
			{nullptr, 0},
	};
	precond.scale_qp_in_place(qp_scaled);
	precond.scale_primal_in_place(x);
	precond.scale_dual_in_place(y);

	auto Htot = to_eigen_matrix_mut(MatrixViewMut<Scalar, colmajor>{
			_htot,
			dim + n_eq,
			dim + n_eq,
			dim + n_eq,
	});
	auto d = to_eigen_vector_mut(VectorViewMut<Scalar>{_d, dim + n_eq});

	Htot.setZero();

	Htot.topLeftCorner(dim, dim) = to_eigen_matrix(qp_scaled.H.as_const());
	for (i32 i = 0; i < dim; ++i) {
		Htot(i, i) += rho;
	}

	// TODO: unneeded
	Htot.topRightCorner(dim, n_eq) =
			to_eigen_matrix(qp_scaled.A.as_const()).transpose();

	Htot.bottomLeftCorner(n_eq, dim) = to_eigen_matrix(qp_scaled.A.as_const());
	{
		Scalar tmp = -Scalar(1) / bcl_mu;
		for (i32 i = 0; i < n_eq; ++i) {
			Htot(dim + i, dim + i) = tmp;
		}
	}

	auto ldlt_mut = LdltViewMut<Scalar, colmajor>{
			from_eigen_matrix_mut(Htot),
			from_eigen_vector_mut(d),
	};

	// initial LDLT factorization
	ldlt::factorize(
			ldlt_mut,
			from_eigen_matrix(Htot),
			ldlt::factorization_strategy::standard);

	auto residue_scaled =
			to_eigen_vector_mut(VectorViewMut<Scalar>{_residue_scaled, dim + n_eq});
	auto residue_scaled_tmp = to_eigen_vector_mut(
			VectorViewMut<Scalar>{_residue_scaled_tmp, dim + n_eq});

	auto residue_unscaled =
			to_eigen_vector_mut(VectorViewMut<Scalar>{_residue_unscaled, dim + n_eq});

	auto next_dual = to_eigen_vector_mut(VectorViewMut<Scalar>{_next_dual, n_eq});
	auto diag_diff = to_eigen_vector_mut(VectorViewMut<Scalar>{_diag_diff, n_eq});

	Scalar primal_feasibility_rhs_1 = infty_norm(to_eigen_vector(qp.b));
	Scalar dual_feasibility_rhs_2 = infty_norm(to_eigen_vector(qp.g));

	for (i32 iter = 0; iter <= max_iter; ++iter) {

		auto dual_residue_scaled = residue_scaled.topRows(dim);
		auto primal_residue_scaled = residue_scaled.bottomRows(n_eq);
		auto dual_residue_unscaled = residue_unscaled.topRows(dim);
		auto primal_residue_unscaled = residue_unscaled.bottomRows(n_eq);

		// compute primal residue
		Scalar primal_feasibility_rhs_0(0);
		Scalar dual_feasibility_rhs_0(0);
		Scalar dual_feasibility_rhs_1(0);

		Scalar primal_feasibility_lhs(0);
		Scalar dual_feasibility_lhs(0);
		{
			auto A_ = to_eigen_matrix(qp_scaled.A.as_const());
			auto x_ = to_eigen_vector(x.as_const());
			auto b_ = to_eigen_vector(qp_scaled.b.as_const());

			// A×x - b
			primal_residue_scaled.setZero();
			primal_residue_scaled.noalias() += A_ * x_;

			{
				auto w = residue_scaled_tmp.bottomRows(n_eq);
				w = primal_residue_scaled;
				precond.unscale_primal_residue_in_place(from_eigen_vector_mut(w));
				primal_feasibility_rhs_0 = infty_norm(w);
			}
			primal_residue_scaled -= b_;

			primal_residue_unscaled = primal_residue_scaled;
			precond.unscale_primal_residue_in_place(
					from_eigen_vector_mut(primal_residue_unscaled));

			primal_feasibility_lhs = infty_norm(primal_residue_unscaled);

			if (iter > 0) {
				if (primal_feasibility_lhs <= bcl_eta) {
					to_eigen_vector_mut(y) = next_dual;
					bcl_eta = bcl_eta / pow(bcl_mu, Scalar(0.9));
				} else {
					Scalar new_bcl_mu = max2(bcl_mu * Scalar(10), Scalar(1e12));
					if (bcl_mu != new_bcl_mu) {
						diag_diff.setConstant(Scalar(1) / bcl_mu - Scalar(1) / new_bcl_mu);
						ldlt::diagonal_update(
								ldlt_mut,
								ldlt_mut.as_const(),
								from_eigen_vector(diag_diff),
								dim,
								ldlt::diagonal_update_strategies::single_pass);
					}
					bcl_mu = new_bcl_mu;
					bcl_eta = Scalar(1) / pow(bcl_mu, Scalar(0.1));
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

		// compute dual residue
		{
			auto H_ = to_eigen_matrix(qp_scaled.H.as_const());
			auto A_ = to_eigen_matrix(qp_scaled.A.as_const());
			auto x_ = to_eigen_vector(x.as_const());
			auto y_ = to_eigen_vector(y.as_const());
			auto g_ = to_eigen_vector(qp_scaled.g.as_const());

			// H×x + g + A.T×y

			// TODO(2): if TODO(1) is applied, update dual_residue_scaled before
			// newton step
			dual_residue_scaled = g_;
			{
				auto w = residue_scaled_tmp.topRows(dim);

				w.setZero();
				w.noalias() += H_ * x_;
				{ dual_residue_scaled += w; }
				precond.unscale_dual_residue_in_place(from_eigen_vector_mut(w));
				dual_feasibility_rhs_0 = infty_norm(w);

				w.setZero();
				w.noalias() += A_.transpose() * y_;
				{ dual_residue_scaled += w; }

				precond.unscale_dual_residue_in_place(from_eigen_vector_mut(w));
				dual_feasibility_rhs_1 = infty_norm(w);
			}

			dual_residue_unscaled = dual_residue_scaled;
			precond.unscale_dual_residue_in_place(
					from_eigen_vector_mut(dual_residue_unscaled));

			dual_feasibility_lhs = infty_norm(dual_residue_unscaled);
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
			precond.unscale_primal_in_place(x);
			precond.unscale_dual_in_place(y);
			return iter;
		}

		// newton step
		{
			auto rhs = residue_scaled;

			rhs = -rhs;
			ldlt::solve(
					from_eigen_vector_mut(rhs),
					ldlt_mut.as_const(),
					from_eigen_vector(rhs));

			to_eigen_vector_mut(x) += rhs.topRows(dim);
			next_dual = to_eigen_vector(y.as_const()) + rhs.bottomRows(n_eq);
		}
	}
	return max_iter;
}
} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS */
