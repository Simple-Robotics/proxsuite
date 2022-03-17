#ifndef INRIA_LDLT_LDLT_HPP_VCVSK3EOS
#define INRIA_LDLT_LDLT_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include <veg/vec.hpp>
#include <veg/memory/dynamic_stack.hpp>
#include <fmt/ostream.h>

namespace ldlt {
namespace detail {
struct SimdAlignedSystemAlloc {
	friend auto operator==(
			SimdAlignedSystemAlloc /*unused*/,
			SimdAlignedSystemAlloc /*unused*/) noexcept -> bool {
		return true;
	}
};
} // namespace detail
} // namespace ldlt

template <>
struct veg::mem::Alloc<ldlt::detail::SimdAlignedSystemAlloc> {
	static constexpr usize min_align = SIMDE_NATURAL_VECTOR_SIZE / 8;

	using RefMut = veg::RefMut<ldlt::detail::SimdAlignedSystemAlloc>;

	VEG_INLINE static auto adjusted_layout(Layout l) noexcept -> Layout {
		if (l.align < min_align) {
			l.align = min_align;
		}
		return l;
	}

	VEG_INLINE static void
	dealloc(RefMut /*alloc*/, void* ptr, Layout l) noexcept {
		return Alloc<SystemAlloc>::dealloc(
				mut(SystemAlloc{}), ptr, adjusted_layout(l));
	}

	VEG_NODISCARD VEG_INLINE static auto
	alloc(RefMut /*alloc*/, Layout l) noexcept -> mem::AllocBlock {
		return Alloc<SystemAlloc>::alloc(mut(SystemAlloc{}), adjusted_layout(l));
	}

	VEG_NODISCARD VEG_INLINE static auto grow(
			RefMut /*alloc*/,
			void* ptr,
			Layout l,
			usize new_size,
			RelocFn reloc) noexcept -> mem::AllocBlock {
		return Alloc<SystemAlloc>::grow(
				mut(SystemAlloc{}), ptr, adjusted_layout(l), new_size, reloc);
	}
	VEG_NODISCARD VEG_INLINE static auto shrink(
			RefMut /*alloc*/,
			void* ptr,
			Layout l,
			usize new_size,
			RelocFn reloc) noexcept -> mem::AllocBlock {
		return Alloc<SystemAlloc>::shrink(
				mut(SystemAlloc{}), ptr, adjusted_layout(l), new_size, reloc);
	}
};

namespace ldlt {

LDLT_DEFINE_TAG(decompose, Decompose);
LDLT_DEFINE_TAG(reserve_uninit, ReserveUninit);

template <typename T>
struct Ldlt {
private:
	static constexpr auto DYN = Eigen::Dynamic;
	using ColMat = Eigen::Matrix<T, DYN, DYN, Eigen::ColMajor>;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	using LView = Eigen::TriangularView<
			Eigen::Map< //
					ColMat const,
					Eigen::Unaligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitLower>;
	using LViewMut = Eigen::TriangularView<
			Eigen::Map< //
					ColMat,
					Eigen::Unaligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitLower>;

	using LTView = Eigen::TriangularView<
			Eigen::Map< //
					RowMat const,
					Eigen::Unaligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitUpper>;
	using LTViewMut = Eigen::TriangularView<
			Eigen::Map< //
					RowMat,
					Eigen::Unaligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitUpper>;

	using DView =
			Eigen::Map<Vec const, Eigen::Unaligned, Eigen::InnerStride<DYN>>;
	using DViewMut = Eigen::Map<Vec, Eigen::Unaligned, Eigen::InnerStride<DYN>>;

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
	using Perm = Eigen::PermutationWrapper<VecMapISize>;

	using StorageSimdVec = veg::Vec<
			T,
			veg::meta::if_t<
					detail::should_vectorize<T>::value,
					detail::SimdAlignedSystemAlloc,
					veg::mem::SystemAlloc>>;

	StorageSimdVec ld_storage;
	isize stride{};
	veg::Vec<isize> perm;
	veg::Vec<isize> perm_inv;
	VEG_REFLECT(Ldlt, ld_storage, stride, perm, perm_inv);

	static auto _adjusted_stride(isize n) noexcept -> isize {
		isize simd_stride = (SIMDE_NATURAL_VECTOR_SIZE / 8) / isize{alignof(T)};
		return detail::should_vectorize<T>::value
		           ? (n + simd_stride - 1) / simd_stride * simd_stride
		           : n;
	}

	// soft invariants:
	// - perm.len() == perm_inv.len() == dim
	// - dim < stride
	// - ld_storage.len() >= dim * stride
public:
	Ldlt() = default;

	void reserve_uninit(isize cap) noexcept {
		static_assert(VEG_CONCEPT(nothrow_constructible<T>), ".");

		auto new_stride = _adjusted_stride(cap);
		if (cap <= stride && cap * new_stride <= ld_storage.len()) {
			return;
		}

		ld_storage.reserve_exact(cap * new_stride);
		perm.reserve_exact(cap);
		perm_inv.reserve_exact(cap);

		ld_storage.resize_for_overwrite(cap * new_stride);
		stride = new_stride;
	}

	void reserve(isize cap) noexcept {
		auto new_stride = _adjusted_stride(cap);
		if (cap <= stride && cap * new_stride <= ld_storage.len()) {
			return;
		}
		auto n = dim();

		ld_storage.reserve_exact(cap * new_stride);
		perm.reserve_exact(cap);
		perm_inv.reserve_exact(cap);

		ld_storage.resize_for_overwrite(cap * new_stride);

		for (isize i = 0; i < n; ++i) {
			auto col = n - i - 1;
			T* ptr = ld_col_mut().data();
			std::move_backward( //
					ptr + col * stride,
					ptr + col * stride + n,
					ptr + col * new_stride + n);
		}
		stride = new_stride;
	}

	static auto rank_one_update_req(isize n) noexcept -> veg::dynstack::StackReq {
		return {n * isize{sizeof(T)}, detail::_align<T>()};
	}

	void rank_one_update( //
			Eigen::Ref<Vec const> z,
			T alpha,
			veg::dynstack::DynStackMut stack) {
		auto n = dim();
		VEG_ASSERT(z.rows() == n);
		LDLT_TEMP_VEC_UNINIT(T, work, n, stack);

		for (isize i = 0; i < n; ++i) {
			work[i] = z[perm[i]];
		}
		LdltViewMut<T> ld{{from_eigen, ld_col_mut()}};
		detail::rank1_update_clobber_z(ld, {from_eigen, work}, alpha);
	}

	auto dim() const noexcept -> isize { return perm.len(); }

	auto ld_col() const noexcept -> Eigen::Map< //
			ColMat const,
			Eigen::Unaligned,
			Eigen::OuterStride<DYN>> {
		return {ld_storage.ptr(), dim(), dim(), stride};
	}
	auto ld_col_mut() noexcept -> Eigen::Map< //
			ColMat,
			Eigen::Unaligned,
			Eigen::OuterStride<DYN>> {
		return {ld_storage.ptr_mut(), dim(), dim(), stride};
	}
	auto ld_row() const noexcept -> Eigen::Map< //
			RowMat const,
			Eigen::Unaligned,
			Eigen::OuterStride<DYN>> {
		return {
				ld_storage.ptr(),
				dim(),
				dim(),
				Eigen::OuterStride<DYN>{stride},
		};
	}
	auto ld_row_mut() noexcept -> Eigen::Map< //
			RowMat,
			Eigen::Unaligned,
			Eigen::OuterStride<DYN>> {
		return {
				ld_storage.ptr_mut(),
				dim(),
				dim(),
				Eigen::OuterStride<DYN>{stride},
		};
	}

	auto l() const noexcept -> LView {
		return ld_col().template triangularView<Eigen::UnitLower>();
	}
	auto l_mut() noexcept -> LViewMut {
		return ld_col_mut().template triangularView<Eigen::UnitLower>();
	}
	auto lt() const noexcept -> LTView {
		return ld_row().template triangularView<Eigen::UnitUpper>();
	}
	auto lt_mut() noexcept -> LTViewMut {
		return ld_row_mut().template triangularView<Eigen::UnitUpper>();
	}

	auto d() const noexcept -> DView {
		return {
				ld_storage.ptr(),
				dim(),
				1,
				Eigen::InnerStride<DYN>{stride + 1},
		};
	}
	auto d_mut() noexcept -> DView {
		return {
				ld_storage.ptr_mut(),
				dim(),
				1,
				Eigen::InnerStride<DYN>{stride + 1},
		};
	}
	auto p() -> Perm { return {VecMapISize(perm.ptr(), dim())}; }
	auto pt() -> Perm { return {VecMapISize(perm_inv.ptr(), dim())}; }

	static auto factor_req(isize n) -> veg::dynstack::StackReq {
		return {
				n * _adjusted_stride(n) * isize{sizeof(T)},
				detail::_align<T>() * isize{alignof(T)}};
	}
	void factor(
			Eigen::Ref<ColMat const> mat /* NOLINT */,
			veg::dynstack::DynStackMut stack) {
		VEG_ASSERT(mat.rows() == mat.cols());
		isize n = mat.rows();
		reserve_uninit(n);

		perm.resize_for_overwrite(n);
		perm_inv.resize_for_overwrite(n);

		ldlt::detail::compute_permutation<T>( //
				perm.ptr_mut(),
				perm_inv.ptr_mut(),
				{from_eigen, mat.diagonal()});

		LDLT_TEMP_MAT_UNINIT(T, work, n, n, stack);
		ld_col_mut() = mat;
		ldlt::detail::apply_permutation_sym_work<T>( //
				{from_eigen, ld_col_mut()},
				perm.ptr(),
				{from_eigen, work},
				-1);

		ldlt::factorize(LdltViewMut<T>{{from_eigen, ld_col_mut()}});
	}

	static auto solve_in_place_req(isize n) -> veg::dynstack::StackReq {
		return {n * isize{sizeof(T)}, detail::_align<T>() * isize(alignof(T))};
	}
	void
	solve_in_place(Eigen::Ref<Vec> rhs, veg::dynstack::DynStackMut stack) const {
		isize n = rhs.rows();
		LDLT_TEMP_VEC_UNINIT(T, work, n, stack);

		ldlt::detail::apply_perm_rows<T>::fn(
				work.data(), 0, rhs.data(), 0, n, 1, perm.ptr(), 0);
		ldlt::solve(
				{from_eigen, work},
				LdltView<T>{{from_eigen, ld_col()}},
				{from_eigen, work});

		ldlt::detail::apply_perm_rows<T>::fn(
				rhs.data(), 0, work.data(), 0, n, 1, perm_inv.ptr(), 0);
	}

	auto dbg_reconstructed_matrix() const -> ColMat {
		isize n = dim();
		auto tmp = ColMat(n, n);
		tmp = l();
		tmp = tmp * d().asDiagonal();
		auto A = ColMat(tmp * lt());

		for (isize i = 0; i < n; i++) {
			tmp.row(i) = A.row(perm_inv[i]);
		}
		for (isize i = 0; i < n; i++) {
			A.col(i) = tmp.col(perm_inv[i]);
		}
		return A;
	}

	void delete_at(isize i) noexcept {
		// delete corresponding row/col after permutation
		// modify permutation

		isize n = dim();
		isize i_actual = perm_inv[i];

		{
			row_delete(LdltViewMut<T>{{from_eigen, ld_col_mut()}}, i_actual);

			perm.pop_mid(i_actual);
			perm_inv.pop_mid(i);

			for (isize k = 0; k < n - 1; ++k) {
				auto& p_k = perm[k];
				auto& pinv_k = perm_inv[k];

				if (p_k > i) {
					--p_k;
				}
				if (pinv_k > i_actual) {
					--pinv_k;
				}
			}
		}
	}

	static auto insert_at_req(isize a_dim) -> veg::dynstack::StackReq {
		return {a_dim * isize{sizeof(T)}, detail::_align<T>()};
	}

	void insert_at(
			isize i, Eigen::Ref<Vec const> a, veg::dynstack::DynStackMut stack) {
		// insert row/col at end of matrix
		// modify permutation

		// TODO: choose better insertion slot
		isize n = dim();
		reserve(n + 1);

		isize i_actual = n;

		{

			for (isize k = 0; k < n; ++k) {
				auto& p_k = perm[k];
				auto& pinv_k = perm_inv[k];

				if (p_k >= i) {
					++p_k;
				}
				if (pinv_k >= i_actual) {
					++pinv_k;
				}
			}

			auto old_view = LdltView<T>{{from_eigen, ld_col_mut()}};

			perm.push_mid(i, i_actual);
			perm_inv.push_mid(i_actual, i);

			auto new_view = LdltViewMut<T>{{from_eigen, ld_col_mut()}};

			LDLT_TEMP_VEC_UNINIT(T, permuted_a, n + 1, stack);
			for (isize k = 0; k < n + 1; ++k) {
				permuted_a[k] = a[perm[k]];
			}

			row_append(new_view, old_view, {from_eigen, permuted_a});
		}
	}
};
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_LDLT_HPP_VCVSK3EOS */
