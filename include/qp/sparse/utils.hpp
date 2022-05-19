/** \file */

#ifndef PROXSUITE_QP_SPARSE_UTILS_HPP
#define PROXSUITE_QP_SPARSE_UTILS_HPP

#include <linearsolver/dense/core.hpp>
#include <linearsolver/sparse/core.hpp>
#include <linearsolver/sparse/factorize.hpp>
#include <linearsolver/sparse/update.hpp>
#include <linearsolver/sparse/rowmod.hpp>
#include <qp/dense/views.hpp>
#include <qp/settings.hpp>
#include <veg/vec.hpp>
#include "qp/results.hpp"
#include "qp/sparse/views.hpp"
#include "qp/sparse/model.hpp"
#include "qp/sparse/preconditioner/ruiz.hpp"
#include "qp/sparse/preconditioner/identity.hpp"

#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace qp {
namespace sparse {

namespace detail {
template <typename T>
auto positive_part(T const& expr)
		VEG_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		VEG_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));

template <typename T, typename I>
VEG_NO_INLINE void noalias_gevmmv_add_impl( //
		VectorViewMut<T> out_l,
		VectorViewMut<T> out_r,
		linearsolver::sparse::MatRef<T, I> a,
		VectorView<T> in_l,
		VectorView<T> in_r) {
	VEG_ASSERT_ALL_OF /* NOLINT */ (
			a.nrows() == out_r.dim,
			a.ncols() == in_r.dim,
			a.ncols() == out_l.dim,
			a.nrows() == in_l.dim);
	// equivalent to
	// out_r.to_eigen().noalias() += a.to_eigen() * in_r.to_eigen();
	// out_l.to_eigen().noalias() += a.to_eigen().transpose() * in_l.to_eigen();

	auto* ai = a.row_indices();
	auto* ax = a.values();
	auto n = a.ncols();

	for (usize j = 0; j < usize(n); ++j) {
		usize col_start = a.col_start(j);
		usize col_end = a.col_end(j);

		T acc0 = 0;
		T acc1 = 0;
		T acc2 = 0;
		T acc3 = 0;

		T in_rj = in_r(isize(j));

		usize pcount = col_end - col_start;

		usize p = col_start;

		auto zx = linearsolver::sparse::util::zero_extend;

		for (; p < col_start + pcount / 4 * 4; p += 4) {
			auto i0 = isize(zx(ai[p + 0]));
			auto i1 = isize(zx(ai[p + 1]));
			auto i2 = isize(zx(ai[p + 2]));
			auto i3 = isize(zx(ai[p + 3]));

			T ai0j = ax[p + 0];
			T ai1j = ax[p + 1];
			T ai2j = ax[p + 2];
			T ai3j = ax[p + 3];

			out_r(i0) += ai0j * in_rj;
			out_r(i1) += ai1j * in_rj;
			out_r(i2) += ai2j * in_rj;
			out_r(i3) += ai3j * in_rj;

			acc0 += ai0j * in_l(i0);
			acc1 += ai1j * in_l(i1);
			acc2 += ai2j * in_l(i2);
			acc3 += ai3j * in_l(i3);
		}

		for (; p < col_end; ++p) {
			auto i = isize(zx(ai[p]));

			T aij = ax[p];
			out_r(i) += aij * in_rj;
			acc0 += aij * in_l(i);
		}

		acc0 = ((acc0 + acc1) + (acc2 + acc3));
		out_l(isize(j)) += acc0;
	}
}

template <typename T, typename I>
VEG_NO_INLINE void noalias_symhiv_add_impl( //
		VectorViewMut<T> out,
		linearsolver::sparse::MatRef<T, I> a,
		VectorView<T> in) {
	VEG_ASSERT_ALL_OF /* NOLINT */ ( //
			a.nrows() == a.ncols(),
			a.nrows() == out.dim,
			a.ncols() == in.dim);
	// equivalent to
	// out.to_eigen().noalias() +=
	// 		a.to_eigen().template selfadjointView<Eigen::Upper>() * in.to_eigen();

	auto* ai = a.row_indices();
	auto* ax = a.values();
	auto n = a.ncols();

	for (usize j = 0; j < usize(n); ++j) {
		usize col_start = a.col_start(j);
		usize col_end = a.col_end(j);

		if (col_start == col_end) {
			continue;
		}

		T acc0 = 0;
		T acc1 = 0;
		T acc2 = 0;
		T acc3 = 0;

		T in_j = in(isize(j));

		usize pcount = col_end - col_start;

		auto zx = linearsolver::sparse::util::zero_extend;

		if (zx(ai[col_end - 1]) == j) {
			T ajj = ax[col_end - 1];
			out(isize(j)) += ajj * in_j;
			pcount -= 1;
		}

		usize p = col_start;

		for (; p < col_start + pcount / 4 * 4; p += 4) {
			auto i0 = isize(zx(ai[p + 0]));
			auto i1 = isize(zx(ai[p + 1]));
			auto i2 = isize(zx(ai[p + 2]));
			auto i3 = isize(zx(ai[p + 3]));

			T ai0j = ax[p + 0];
			T ai1j = ax[p + 1];
			T ai2j = ax[p + 2];
			T ai3j = ax[p + 3];

			out(i0) += ai0j * in_j;
			out(i1) += ai1j * in_j;
			out(i2) += ai2j * in_j;
			out(i3) += ai3j * in_j;

			acc0 += ai0j * in(i0);
			acc1 += ai1j * in(i1);
			acc2 += ai2j * in(i2);
			acc3 += ai3j * in(i3);
		}
		for (; p < col_start + pcount; ++p) {
			auto i = isize(zx(ai[p]));

			T aij = ax[p];
			out(i) += aij * in_j;
			acc0 += aij * in(i);
		}
		acc0 = ((acc0 + acc1) + (acc2 + acc3));
		out(isize(j)) += acc0;
	}
}

template <typename OutL, typename OutR, typename A, typename InL, typename InR>
void noalias_gevmmv_add(
		OutL&& out_l, OutR&& out_r, A const& a, InL const& in_l, InR const& in_r) {
	// noalias general vector matrix matrix vector add
	noalias_gevmmv_add_impl<typename A::Scalar, typename A::StorageIndex>(
			{qp::from_eigen, out_l},
			{qp::from_eigen, out_r},
			{linearsolver::sparse::from_eigen, a},
			{qp::from_eigen, in_l},
			{qp::from_eigen, in_r});
}

template <typename Out, typename A, typename In>
void noalias_symhiv_add(Out&& out, A const& a, In const& in) {
	// noalias symmetric (hi) matrix vector add
	noalias_symhiv_add_impl<typename A::Scalar, typename A::StorageIndex>(
			{qp::from_eigen, out},
			{linearsolver::sparse::from_eigen, a},
			{qp::from_eigen, in});
}

template <typename T, typename I>
struct AugmentedKkt : Eigen::EigenBase<AugmentedKkt<T, I>> {
	struct Raw /* NOLINT */ {
		linearsolver::sparse::MatRef<T, I> kkt_active;
		veg::Slice<bool> active_constraints;
		isize n;
		isize n_eq;
		isize n_in;
		T rho;
		T mu_eq;
		T mu_in;
	} _;

	AugmentedKkt /* NOLINT */ (Raw raw) noexcept : _{raw} {}

	using Scalar = T;
	using RealScalar = T;
	using StorageIndex = I;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false,
	};

	auto rows() const noexcept -> isize { return _.n + _.n_eq + _.n_in; }
	auto cols() const noexcept -> isize { return rows(); }
	template <typename Rhs>
	auto operator*(Eigen::MatrixBase<Rhs> const& x) const
			-> Eigen::Product<AugmentedKkt, Rhs, Eigen::AliasFreeProduct> {
		return Eigen::Product< //
				AugmentedKkt,
				Rhs,
				Eigen::AliasFreeProduct>{
				*this,
				x.derived(),
		};
	}
};

template <typename T>
using VecMapMut = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1>,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;
template <typename T>
using VecMap = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1> const,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;

template <typename V>
auto vec(V const& v) -> VecMap<typename V::Scalar> {
	static_assert(V::InnerStrideAtCompileTime == 1, ".");
	return {
			v.data(),
			v.rows(),
			v.cols(),
			Eigen::Stride<Eigen::Dynamic, 1>{
					v.outerStride(),
					v.innerStride(),
			},
	};
}

template <typename V>
auto vec_mut(V&& v) -> VecMapMut<typename veg::uncvref_t<V>::Scalar> {
	static_assert(veg::uncvref_t<V>::InnerStrideAtCompileTime == 1, ".");
	return {
			v.data(),
			v.rows(),
			v.cols(),
			Eigen::Stride<Eigen::Dynamic, 1>{
					v.outerStride(),
					v.innerStride(),
			},
	};
}

template <typename T, typename I>
auto middle_cols(
		linearsolver::sparse::MatRef<T, I> mat, isize start, isize ncols, isize nnz)
		-> linearsolver::sparse::MatRef<T, I> {
	VEG_ASSERT(start <= mat.ncols());
	VEG_ASSERT(ncols <= mat.ncols() - start);

	return {
			linearsolver::sparse::from_raw_parts,
			mat.nrows(),
			ncols,
			nnz,
			mat.col_ptrs() + start,
			mat.is_compressed() ? nullptr : (mat.nnz_per_col() + start),
			mat.row_indices(),
			mat.values(),
	};
}

template <typename T, typename I>
auto middle_cols_mut(
		linearsolver::sparse::MatMut<T, I> mat, isize start, isize ncols, isize nnz)
		-> linearsolver::sparse::MatMut<T, I> {
	VEG_ASSERT(start <= mat.ncols());
	VEG_ASSERT(ncols <= mat.ncols() - start);
	return {
			linearsolver::sparse::from_raw_parts,
			mat.nrows(),
			ncols,
			nnz,
			mat.col_ptrs_mut() + start,
			mat.is_compressed() ? nullptr : (mat.nnz_per_col_mut() + start),
			mat.row_indices_mut(),
			mat.values_mut(),
	};
}

template <typename T, typename I>
auto top_rows_unchecked(
		veg::Unsafe /*unsafe*/, linearsolver::sparse::MatRef<T, I> mat, isize nrows)
		-> linearsolver::sparse::MatRef<T, I> {
	VEG_ASSERT(nrows <= mat.nrows());
	return {
			linearsolver::sparse::from_raw_parts,
			nrows,
			mat.ncols(),
			mat.nnz(),
			mat.col_ptrs(),
			mat.nnz_per_col(),
			mat.row_indices(),
			mat.values(),
	};
}

template <typename T, typename I>
auto top_rows_mut_unchecked(
		veg::Unsafe /*unsafe*/, linearsolver::sparse::MatMut<T, I> mat, isize nrows)
		-> linearsolver::sparse::MatMut<T, I> {
	VEG_ASSERT(nrows <= mat.nrows());
	return {
			linearsolver::sparse::from_raw_parts,
			nrows,
			mat.ncols(),
			mat.nnz(),
			mat.col_ptrs_mut(),
			mat.nnz_per_col_mut(),
			mat.row_indices_mut(),
			mat.values_mut(),
	};
}

template <typename T, typename I, typename P>
auto unscaled_primal_dual_residual(
		VecMapMut<T> primal_residual_eq_scaled,
		VecMapMut<T> primal_residual_in_scaled_lo,
		VecMapMut<T> primal_residual_in_scaled_up,
		VecMapMut<T> dual_residual_scaled,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		T dual_feasibility_rhs_0,
		T dual_feasibility_rhs_1,
		T dual_feasibility_rhs_3,
		P& precond,
		Model<T, I> const& data,
		QpView<T, I> qp_scaled,
		VecMap<T> x_e,
		VecMap<T> y_e,
		VecMap<T> z_e,
		veg::dynstack::DynStackMut stack) -> veg::Tuple<T, T> {
	isize n = x_e.rows();

	LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
	dual_residual_scaled = qp_scaled.g.to_eigen();

	{
		tmp.setZero();
		noalias_symhiv_add(tmp, qp_scaled.H.to_eigen(), x_e);
		dual_residual_scaled += tmp;

		precond.unscale_dual_residual_in_place({qp::from_eigen, tmp});
		dual_feasibility_rhs_0 = infty_norm(tmp);
	}

	{
		auto ATy = tmp;
		ATy.setZero();
		primal_residual_eq_scaled.setZero();

		detail::noalias_gevmmv_add(
				primal_residual_eq_scaled, ATy, qp_scaled.AT.to_eigen(), x_e, y_e);

		dual_residual_scaled += ATy;

		precond.unscale_dual_residual_in_place({qp::from_eigen, ATy});
		dual_feasibility_rhs_1 = infty_norm(ATy);
	}

	{
		auto CTz = tmp;
		CTz.setZero();
		primal_residual_in_scaled_up.setZero();

		detail::noalias_gevmmv_add(
				primal_residual_in_scaled_up, CTz, qp_scaled.CT.to_eigen(), x_e, z_e);

		dual_residual_scaled += CTz;

		precond.unscale_dual_residual_in_place({qp::from_eigen, CTz});
		dual_feasibility_rhs_3 = infty_norm(CTz);
	}

	precond.unscale_primal_residual_in_place_eq(
			{qp::from_eigen, primal_residual_eq_scaled});

	primal_feasibility_eq_rhs_0 = infty_norm(primal_residual_eq_scaled);

	precond.unscale_primal_residual_in_place_in(
			{qp::from_eigen, primal_residual_in_scaled_up});
	primal_feasibility_in_rhs_0 = infty_norm(primal_residual_in_scaled_up);

	auto b = data.b;
	auto l = data.l;
	auto u = data.u;
	primal_residual_in_scaled_lo =
			positive_part(primal_residual_in_scaled_up - u) +
			negative_part(primal_residual_in_scaled_up - l);

	primal_residual_eq_scaled -= b;
	T primal_feasibility_eq_lhs = infty_norm(primal_residual_eq_scaled);
	T primal_feasibility_in_lhs = infty_norm(primal_residual_in_scaled_lo);
	T primal_feasibility_lhs =
			std::max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	// scaled Ax - b
	precond.scale_primal_residual_in_place_eq(
			{qp::from_eigen, primal_residual_eq_scaled});
	// scaled Cx
	precond.scale_primal_residual_in_place_in(
			{qp::from_eigen, primal_residual_in_scaled_up});

	precond.unscale_dual_residual_in_place(
			{qp::from_eigen, dual_residual_scaled});
	T dual_feasibility_lhs = infty_norm(dual_residual_scaled);
	precond.scale_dual_residual_in_place({qp::from_eigen, dual_residual_scaled});

	return veg::tuplify(primal_feasibility_lhs, dual_feasibility_lhs);
}

} // namespace detail
} // namespace sparse
} // namespace qp
} // namespace proxsuite

namespace Eigen {
namespace internal {
template <typename T, typename I>
struct traits<proxsuite::qp::sparse::detail::AugmentedKkt<T, I>>
		: Eigen::internal::traits<Eigen::SparseMatrix<T, Eigen::ColMajor, I>> {};

template <typename Rhs, typename T, typename I>
struct generic_product_impl<
		proxsuite::qp::sparse::detail::AugmentedKkt<T, I>,
		Rhs,
		SparseShape,
		DenseShape,
		GemvProduct>
		: generic_product_impl_base<
					proxsuite::qp::sparse::detail::AugmentedKkt<T, I>,
					Rhs,
					generic_product_impl<
							proxsuite::qp::sparse::detail::AugmentedKkt<T, I>,
							Rhs>> {
	using Mat_ = proxsuite::qp::sparse::detail::AugmentedKkt<T, I>;

	using Scalar = typename Product<Mat_, Rhs>::Scalar;

	template <typename Dst>
	static void scaleAndAddTo(
			Dst& dst, Mat_ const& lhs, Rhs const& rhs, Scalar const& alpha) {
		using veg::isize;

		VEG_ASSERT(alpha == Scalar(1));
		proxsuite::qp::sparse::detail::noalias_symhiv_add(
				dst, lhs._.kkt_active.to_eigen(), rhs);

		{
			isize n = lhs._.n;
			isize n_eq = lhs._.n_eq;
			isize n_in = lhs._.n_in;

			auto dst_x = dst.head(n);
			auto dst_y = dst.segment(n, n_eq);
			auto dst_z = dst.tail(n_in);

			auto rhs_x = rhs.head(n);
			auto rhs_y = rhs.segment(n, n_eq);
			auto rhs_z = rhs.tail(n_in);

			dst_x += lhs._.rho * rhs_x;
			dst_y += (-1 / lhs._.mu_eq) * rhs_y;
			for (isize i = 0; i < n_in; ++i) {
				dst_z[i] +=
						(lhs._.active_constraints[i] ? -1 / lhs._.mu_in : T(1)) * rhs_z[i];
			}
		}
	}
};
} // namespace internal
} // namespace Eigen

#endif /* end of include guard PROXSUITE_QP_SPARSE_UTILS_HPP */
