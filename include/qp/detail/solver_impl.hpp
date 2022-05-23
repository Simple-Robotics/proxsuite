#ifndef PROXSUITE_QP_DETAIL_SOLVER_IMPL_HPP
#define PROXSUITE_QP_DETAIL_SOLVER_IMPL_HPP

#include "linearsolver/sparse/core.hpp"
#include "linearsolver/sparse/factorize.hpp"
#include "qp/dense/views.hpp"
#include "qp/sparse/views.hpp"
#include "qp/status.hpp"

#include <Eigen/Core>
#include <linearsolver/dense/core.hpp>
#include <veg/box.hpp>
#include <veg/memory/dynamic_stack.hpp>
#include <veg/vec.hpp>
#include <veg/fn_dyn.hpp>
#include <ostream>
#include <memory>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
using veg::isize;
using veg::usize;

namespace qp {
namespace sparse {
namespace detail {
template <typename T, typename I>
struct AugmentedKkt : Eigen::EigenBase<AugmentedKkt<T, I>> {
	struct Internal {
		linearsolver::sparse::MatRef<T, I> kkt_active;
		bool const* active_constraints;
		isize n;
		isize n_eq;
		isize n_in;
		T rho;
		T mu_eq;
		T mu_in;
	} internal;

	explicit AugmentedKkt(Internal i) noexcept : internal{i} {}

	using Scalar = T;
	using RealScalar = T;
	using StorageIndex = I;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false,
	};

	auto rows() const noexcept -> isize {
		return internal.n + internal.n_eq + internal.n_in;
	}
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

template <typename Out, typename A, typename In>
void noalias_symhiv_add(Out&& out, A const& a, In const& in) {
	// noalias symmetric (hi) matrix vector add
	detail::noalias_symhiv_add_impl<typename A::Scalar, typename A::StorageIndex>(
			{qp::from_eigen, out},
			{linearsolver::sparse::from_eigen, a},
			{qp::from_eigen, in});
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
				dst, lhs.internal.kkt_active.to_eigen(), rhs);

		{
			isize n = lhs.internal.n;
			isize n_eq = lhs.internal.n_eq;
			isize n_in = lhs.internal.n_in;

			auto dst_x = dst.head(n);
			auto dst_y = dst.segment(n, n_eq);
			auto dst_z = dst.tail(n_in);

			auto rhs_x = rhs.head(n);
			auto rhs_y = rhs.segment(n, n_eq);
			auto rhs_z = rhs.tail(n_in);

			dst_x += lhs.internal.rho * rhs_x;
			dst_y += (-lhs.internal.mu_eq) * rhs_y;
			for (isize i = 0; i < n_in; ++i) {
				dst_z[i] +=
						(lhs.internal.active_constraints[i] ? -lhs.internal.mu_in : T(1)) *
						rhs_z[i];
			}
		}
	}
};
} // namespace internal
} // namespace Eigen

namespace proxsuite {
namespace qp {

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using Stride = Eigen::Stride<Eigen::Dynamic, 1>;

template <typename T>
struct SolverState {
	T mu_eq = T(1e-3);
	T mu_in = T(1e-1);
	T rho = T(1e-6);
	T nu = T(1e0);

	SolverState() = default;
};

template <typename T>
struct Results {
	struct Internal {
		SolverState<T> state;
		isize n = 0;
		isize n_eq = 0;
		isize n_in = 0;
		veg::Vec<T> solution;
		veg::Vec<bool> active_constraints;
	} internal;

	Results() = default;

	auto active_constraints() const -> bool const* {
		return internal.active_constraints.ptr();
	}
	auto active_constraints_mut() -> bool* {
		return internal.active_constraints.ptr_mut();
	}

	void reserve_exact(isize n, isize n_eq, isize n_in) {
		internal.solution.reserve_exact(n + n_eq + n_in);
		internal.active_constraints.reserve_exact(n_in);
	}

	void reserve(isize n, isize n_eq, isize n_in) {
		internal.solution.reserve(n + n_eq + n_in);
		internal.active_constraints.reserve(n_in);
	}

	void resize(isize n, isize n_eq, isize n_in) {
		internal.solution.resize(n + n_eq + n_in);
		internal.active_constraints.resize(n_in);
		internal.n = n;
		internal.n_eq = n_eq;
		internal.n_in = n_in;
	}

	auto sol() const -> Eigen::Map<Vector<T> const, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		auto n_eq = internal.n_eq;
		auto n_in = internal.n_in;
		return {
				internal.solution.ptr(),
				internal.solution.len(),
				1,
				Stride{internal.solution.len(), 1},
		};
	}
	auto sol_mut() -> Eigen::Map<Vector<T>, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		auto n_eq = internal.n_eq;
		auto n_in = internal.n_in;
		return {
				internal.solution.ptr_mut(),
				internal.solution.len(),
				1,
				Stride{internal.solution.len(), 1},
		};
	}

	auto x() const -> Eigen::Map<Vector<T> const, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		return linearsolver::dense::util::subrows(sol(), 0, n);
	}
	auto x_mut() -> Eigen::Map<Vector<T>, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		return linearsolver::dense::util::subrows(sol_mut(), 0, n);
	}
	auto y() const -> Eigen::Map<Vector<T> const, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		auto n_eq = internal.n_eq;
		return linearsolver::dense::util::subrows(sol(), n, n_eq);
	}
	auto y_mut() -> Eigen::Map<Vector<T>, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		auto n_eq = internal.n_eq;
		return linearsolver::dense::util::subrows(sol_mut(), n, n_eq);
	}
	auto z() const -> Eigen::Map<Vector<T> const, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		auto n_eq = internal.n_eq;
		auto n_in = internal.n_in;
		return linearsolver::dense::util::subrows(sol(), n + n_eq, n_in);
	}
	auto z_mut() -> Eigen::Map<Vector<T>, Eigen::Unaligned, Stride> {
		auto n = internal.n;
		auto n_eq = internal.n_eq;
		auto n_in = internal.n_in;
		return linearsolver::dense::util::subrows(sol_mut(), n + n_eq, n_in);
	}
};

enum struct SolutionInit {
	WARM,
	COLD_EQ_CONSTRAINED,
};

template <typename T>
struct Settings {
	T bcl_alpha = T(0.1);
	T bcl_beta = T(0.9);

	T refactor_dual_feasibility_threshold = T(1e-2);
	T refactor_rho_threshold = T(1e-7);

	T mu_max_eq = T(1e-9);
	T mu_max_in = T(1e-8);

	T mu_update_factor = T(0.1);

	T cold_reset_mu_eq = T(1.0 / 1.1);
	T cold_reset_mu_in = T(1.0 / 1.1);

	isize max_iter_outer = 10000;
	isize max_iter_inner = 1500;
	T eps_abs = T(1e-9);
	T eps_rel = T(0.0);
	T eps_refact = T(1e-6);
	isize n_iterative_refinement = 10;

	//SolutionInit warm_start = SolutionInit::COLD_EQ_CONSTRAINED;
	InitialGuessStatus initial_guess = InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
	std::ostream* logger = nullptr;

	T eps_primal_inf = T(1e-14);
	T eps_dual_inf = T(1e-14);

	Settings() = default;
};

namespace sparse {
template <typename T, typename I>
struct Ldlt {
	veg::Vec<I> etree;
	veg::Vec<I> perm;
	veg::Vec<I> perm_inv;
	veg::Vec<I> col_ptrs;
	veg::Vec<I> nnz_counts;
	veg::Vec<I> row_indices;
	veg::Vec<T> values;
};

template <typename T, typename I>
struct Kkt {
	veg::Vec<I> col_ptrs;
	veg::Vec<I> row_indices;
	veg::Vec<T> values;

	auto as_ref() const -> linearsolver::sparse::MatRef<T, I> {
		auto n_tot = col_ptrs.len() - 1;
		auto nnz = isize(linearsolver::sparse::util::zero_extend(col_ptrs[n_tot]));
		return {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				col_ptrs.ptr(),
				nullptr,
				row_indices.ptr(),
				values.ptr(),
		};
	}
	auto as_mut() -> linearsolver::sparse::MatMut<T, I> {
		auto n_tot = col_ptrs.len() - 1;
		auto nnz = isize(linearsolver::sparse::util::zero_extend(col_ptrs[n_tot]));
		return {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				col_ptrs.ptr_mut(),
				nullptr,
				row_indices.ptr_mut(),
				values.ptr_mut(),
		};
	}
};

template <typename T, typename I>
struct Model {
	Kkt<T, I> kkt;
	veg::Vec<T> g;
	veg::Vec<T> b;
	veg::Vec<T> l;
	veg::Vec<T> u;
};

template <typename T, typename I>
struct Workspace {
	using MatrixFreeSolver = Eigen::MINRES<
			detail::AugmentedKkt<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>;

	struct Internal {
		veg::Vec<veg::mem::byte> storage;

		bool do_ldlt = false;
		Ldlt<T, I> ldlt;

		// stored in unique_ptr because we need a stable address
		std::unique_ptr<detail::AugmentedKkt<T, I>> matrix_free_kkt;
		std::unique_ptr<MatrixFreeSolver> matrix_free_solver;
	} internal;

	Workspace() = default;

	// must not be used while the storage is currently in use
	auto stack_mut() -> veg::dynstack::DynStackMut {
		return {
				veg::from_slice_mut,
				internal.storage.as_mut(),
		};
	}
};

namespace detail {
template <typename T>
void copy_vector_to_workspace(
		veg::Vec<T>& dst, linearsolver::sparse::DenseVecRef<T> src) {
	isize n = src.nrows();
	dst.resize_for_overwrite(src.nrows());
	T* dstp = dst.ptr_mut();
	T const* srcp = src.as_slice().ptr();
	std::copy(srcp, srcp + n, dstp);
}

template <typename T, typename I>
void insert_submatrix(
		usize& col,
		usize& pos,
		I* kktp,
		I* kkti,
		T* kktx,
		linearsolver::sparse::MatRef<T, I> const& m,
		bool assert_sym_hi = false) {
	I const* mi = m.row_indices();
	T const* mx = m.values();
	isize ncols = m.ncols();

	for (usize j = 0; j < usize(ncols); ++j) {
		usize col_start = m.col_start(j);
		usize col_end = m.col_end(j);

		kktp[col + 1] = linearsolver::sparse::util::checked_non_negative_plus(
				kktp[col], I(col_end - col_start));
		++col;

		for (usize p = col_start; p < col_end; ++p) {
			usize i = linearsolver::sparse::util::zero_extend(mi[p]);
			if (assert_sym_hi) {
				VEG_ASSERT(i <= j);
			}

			kkti[pos] = I(i);
			kktx[pos] = mx[p];

			++pos;
		}
	}
}

template <typename T, typename I>
auto refactorize_req(
		veg::Tag<T> xtag, veg::Tag<I> itag, isize n_tot, isize kkt_nnz)
		-> veg::dynstack::StackReq {
	auto symbolic_req = linearsolver::sparse::factorize_symbolic_req(
			itag, n_tot, kkt_nnz, linearsolver::sparse::Ordering::user_provided);
	auto diag_req = linearsolver::dense::temp_vec_req(xtag, n_tot);
	auto numeric_req = linearsolver::sparse::factorize_numeric_req(
			xtag,
			itag,
			n_tot,
			kkt_nnz,
			linearsolver::sparse::Ordering::user_provided);
	return symbolic_req | (diag_req & numeric_req);
}

template <typename T, typename I>
void refactorize(
		SolverState<T> const& state,
		linearsolver::sparse::MatRef<T, I> const& kkt_active,
		bool const* active_constraints,
		Workspace<T, I>& work,
		veg::dynstack::DynStackMut stack) {
	auto& aug_kkt = *work.internal.matrix_free_kkt.get();
	auto& iterative_solver = *work.internal.matrix_free_solver.get();

	isize n = aug_kkt.internal.n;
	isize n_eq = aug_kkt.internal.n_eq;
	isize n_in = aug_kkt.internal.n_in;
	isize n_tot = n + n_eq + n_in;

	if (work.internal.do_ldlt) {
		Ldlt<T, I>& ldlt = work.internal.ldlt;
		linearsolver::sparse::factorize_symbolic_non_zeros(
				ldlt.nnz_counts.ptr_mut(),
				ldlt.etree.ptr_mut(),
				ldlt.perm_inv.ptr_mut(),
				ldlt.perm.ptr(),
				kkt_active.symbolic(),
				stack);

		LDLT_TEMP_VEC_UNINIT(T, diag, n_tot, stack);
		for (isize i = 0; i < n; ++i) {
			diag[i] = state.rho;
		}
		for (isize i = 0; i < n_eq; ++i) {
			diag[n + i] = -state.mu_eq;
		}
		for (isize i = 0; i < n_in; ++i) {
			diag[n + n_eq + i] = active_constraints[i] ? -state.mu_in : T(1);
		}

		linearsolver::sparse::factorize_numeric(
				ldlt.values.ptr_mut(),
				ldlt.row_indices.ptr_mut(),
				diag.data(),
				ldlt.perm.ptr(),
				ldlt.col_ptrs.ptr(),
				ldlt.etree.ptr(),
				ldlt.perm_inv.ptr(),
				kkt_active,
				stack);

	} else {
		aug_kkt = AugmentedKkt<T, I>{{
				kkt_active,
				active_constraints,
				n,
				n_eq,
				n_in,
				state.rho,
				state.mu_eq,
				state.mu_in,
		}};
		iterative_solver.compute(aug_kkt);
	}
}
} // namespace detail

template <typename T, typename I>
void setup(
		Workspace<T, I>& work,
		Model<T, I>& model,
		Results<T>& results,
		Settings<T> const& settings,
		QpView<T, I> const& qp) {
	using namespace veg::dynstack;

	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();
	isize n_tot = n + n_eq + n_in;

	if (settings.warm_start == SolutionInit::WARM) {
		VEG_ASSERT_ALL_OF(
				results.internal.n == n,
				results.internal.n_eq == n_eq,
				results.internal.n_in == n_in,
				results.internal.solution.len() == n_tot,
				results.internal.active_constraints.len() == n_in);
	} else if (settings.warm_start == SolutionInit::COLD_EQ_CONSTRAINED) {
		results.resize(n, n_eq, n_in);
		for (isize i = 0; i < n_in; ++i) {
			results.active_constraints_mut()[i] = false;
		}
	} else {
		VEG_UNIMPLEMENTED();
	}

	isize H_nnz = qp.H.nnz();
	isize A_nnz = qp.AT.nnz();
	isize C_nnz = qp.CT.nnz();
	isize nnz_tot = H_nnz + A_nnz + C_nnz;

	Ldlt<T, I>& ldlt = work.internal.ldlt;
	Kkt<T, I>& kkt_storage = model.kkt;

	// copy qp vectors
	{
		detail::copy_vector_to_workspace(model.g, qp.g);
		detail::copy_vector_to_workspace(model.b, qp.b);
		detail::copy_vector_to_workspace(model.l, qp.l);
		detail::copy_vector_to_workspace(model.u, qp.u);
	}

	// copy qp matrices
	{
		kkt_storage.col_ptrs.resize_for_overwrite(n_tot + 1);
		kkt_storage.row_indices.resize_for_overwrite(nnz_tot);
		kkt_storage.values.resize_for_overwrite(nnz_tot);

		I* kktp = kkt_storage.col_ptrs.ptr_mut();
		I* kkti = kkt_storage.row_indices.ptr_mut();
		T* kktx = kkt_storage.values.ptr_mut();

		kktp[0] = 0;
		usize col = 0;
		usize pos = 0;

		detail::insert_submatrix(col, pos, kktp, kkti, kktx, qp.H, true);
		detail::insert_submatrix(col, pos, kktp, kkti, kktx, qp.AT);
		detail::insert_submatrix(col, pos, kktp, kkti, kktx, qp.CT);
	}

	// allocate storage for symbolic factorization
	{
		auto req = linearsolver::sparse::factorize_symbolic_req(
				veg::Tag<I>{}, n_tot, nnz_tot, linearsolver::sparse::Ordering::amd);
		work.internal.storage.resize_for_overwrite(req.alloc_req());
	}

	// perform symbolic factorization, checking for overflow
	bool overflow = false;
	auto zx = linearsolver::sparse::util::zero_extend;
	{
		ldlt.etree.resize_for_overwrite(n_tot);
		ldlt.perm.resize_for_overwrite(n_tot);
		ldlt.perm_inv.resize_for_overwrite(n_tot);
		ldlt.col_ptrs.resize_for_overwrite(n_tot + 1);

		DynStackMut stack = work.stack_mut();
		auto kkt_symbolic = linearsolver::sparse::SymbolicMatRef<I>{
				veg::from_raw_parts,
				n_tot,
				n_tot,
				nnz_tot,
				kkt_storage.col_ptrs.ptr(),
				nullptr,
				kkt_storage.row_indices.ptr(),
		};
		linearsolver::sparse::factorize_symbolic_non_zeros(
				ldlt.col_ptrs.ptr_mut() + 1,
				ldlt.etree.ptr_mut(),
				ldlt.perm_inv.ptr_mut(),
				static_cast<I const*>(nullptr),
				kkt_symbolic,
				stack);

		// compute permutation from inverse
		for (isize i = 0; i < n_tot; ++i) {
			ldlt.perm[isize(zx(ldlt.perm_inv[i]))] = I(i);
		}

		auto pcol_ptrs = ldlt.col_ptrs.ptr_mut();
		pcol_ptrs[0] = I(0);

		using veg::u64;
		u64 acc = 0;
		for (usize i = 0; i < usize(n_tot); ++i) {
			acc += u64(zx(pcol_ptrs[i + 1]));
			if (acc != u64(I(acc))) {
				overflow = true;
			}
			pcol_ptrs[(i + 1)] = I(acc);
		}
	}
	auto max_lnnz = isize(zx(ldlt.col_ptrs[n_tot]));

	// if ldlt is sparse, use ldlt
	work.internal.do_ldlt = !overflow && max_lnnz < 10000000;

	StackReq req =
			linearsolver::dense::temp_vec_req(veg::Tag<I>{}, n_tot) &
			detail::refactorize_req(
					veg::Tag<T>{}, veg::Tag<I>{}, n_tot, H_nnz + A_nnz + C_nnz);

	work.internal.storage.resize_for_overwrite(req.alloc_req());

	// initial factorization
	{
		linearsolver::sparse::MatRef<T, I> kkt = kkt_storage.as_ref();
		DynStackMut stack = work.stack_mut();
		LDLT_TEMP_VEC_UNINIT(I, kkt_nnz_counts, n_tot, stack);

		// H and A are always active
		for (usize j = 0; j < usize(n + n_eq); ++j) {
			kkt_nnz_counts[isize(j)] = I(kkt.col_end(j) - kkt.col_start(j));
		}

		isize C_active_nnz = 0;
		for (isize j = 0; j < n_in; ++j) {
			auto is_active = results.active_constraints()[j];
			kkt_nnz_counts[n + n_eq + j] = is_active
			                                   ? I(kkt.col_end(usize(n + n_eq + j)) -
			                                       kkt.col_start(usize(n + n_eq + j)))
			                                   : I(0);
			C_active_nnz += kkt_nnz_counts[n + n_eq + j];
		}

		linearsolver::sparse::MatRef<T, I> kkt_active = {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				H_nnz + A_nnz + C_active_nnz,
				kkt.col_ptrs(),
				kkt_nnz_counts.data(),
				kkt.row_indices(),
				kkt.values(),
		};

		work.internal.matrix_free_solver.reset(
				new typename Workspace<T, I>::MatrixFreeSolver);
		work.internal.matrix_free_kkt.reset(new detail::AugmentedKkt<T, I>{
				{kkt_active, results.active_constraints(), n, n_eq, n_in, {}, {}, {}}});

		isize ldlt_ntot = work.internal.do_ldlt ? n_tot : 0;
		isize ldlt_lnnz = work.internal.do_ldlt ? max_lnnz : 0;

		ldlt.nnz_counts.resize_for_overwrite(ldlt_ntot);
		ldlt.row_indices.resize_for_overwrite(ldlt_lnnz);
		ldlt.values.resize_for_overwrite(ldlt_lnnz);

		linearsolver::sparse::MatMut<T, I> ldlt_mut = {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				0,
				ldlt.col_ptrs.ptr_mut(),
				work.internal.do_ldlt ? ldlt.nnz_counts.ptr_mut() : nullptr,
				ldlt.row_indices.ptr_mut(),
				ldlt.values.ptr_mut(),
		};

		detail::refactorize(
				results.internal.state,
				kkt_active,
				results.active_constraints(),
				work,
				stack);
	}
}
} // namespace sparse
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DETAIL_SOLVER_IMPL_HPP */
