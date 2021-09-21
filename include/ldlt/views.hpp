#ifndef INRIA_LDLT_VIEWS_HPP_UGNXAQSBS
#define INRIA_LDLT_VIEWS_HPP_UGNXAQSBS

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include "ldlt/detail/meta.hpp"
#include <Eigen/Core>
#include <cstdint>
#include <new>

namespace ldlt {
namespace detail {

struct NoCopy {
	NoCopy() = default;
	~NoCopy() = default;

	NoCopy(NoCopy const&) = delete;
	NoCopy(NoCopy&&) = delete;
	auto operator=(NoCopy const&) -> NoCopy& = delete;
	auto operator=(NoCopy&&) -> NoCopy& = delete;
};

template <typename Fn>
struct Defer /* NOLINT */ {
	Fn fn;
	NoCopy _;

	LDLT_INLINE ~Defer() noexcept(noexcept(LDLT_FWD(fn)())) { LDLT_FWD(fn)(); }
};

namespace nb {
struct defer {
	template <typename Fn>
	LDLT_INLINE constexpr auto operator()(Fn fn) const -> Defer<Fn> {
		return {LDLT_FWD(fn), {}};
	}
};
struct max2 {
	template <typename T>
	LDLT_INLINE constexpr auto operator()(T const& a, T const& b) const
			-> T const& {
		return a > b ? a : b;
	}
};
struct min2 {
	template <typename T>
	LDLT_INLINE constexpr auto operator()(T a, T b) const -> T {
		return (a < b) ? a : b;
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(defer);
LDLT_DEFINE_NIEBLOID(max2);
LDLT_DEFINE_NIEBLOID(min2);

template <typename T, bool = std::is_floating_point<T>::value>
struct SetZeroImpl {
	static void fn(T* dest, usize n) {
		for (usize i = 0; i < n; ++i) {
			*dest = 0;
		}
	}
};

template <typename T>
struct SetZeroImpl<T, true> {
	static void fn(T* dest, usize n) {
		// TODO: assert bit representation is zero
		std::memset(dest, 0, n * sizeof(T));
	}
};

template <typename T>
void set_zero(T* dest, usize n) {
	SetZeroImpl<T>::fn(dest, n);
}

constexpr auto round_up(isize n, isize k) noexcept -> isize {
	return (n + k - 1) / k * k;
}

inline auto next_aligned(void* ptr, usize align) noexcept -> void* {
	using BytePtr = unsigned char*;
	using VoidPtr = void*;
	using UPtr = std::uintptr_t;

	UPtr mask = align - 1;
	UPtr iptr = UPtr(ptr);
	UPtr aligned_ptr = (iptr + mask) & ~mask;
	return VoidPtr(BytePtr(ptr) + aligned_ptr - iptr);
}

constexpr usize align_simd = (SIMDE_NATURAL_VECTOR_SIZE / 8U);
constexpr usize align_cacheline = 64;
constexpr usize align_simd_and_cacheline = max2(align_simd, align_cacheline);

template <typename T>
struct SimdCachelineAlignStep {
	static constexpr isize value =
			isize(max2(usize(1), align_simd_and_cacheline / alignof(T)));
};

template <typename T>
struct UniqueMalloca {
	void* malloca_ptr;
	T* data;
	usize n;

	static constexpr usize align = max2(alignof(T), align_simd_and_cacheline);

	static constexpr usize max_stack_count =
			usize(LDLT_MAX_STACK_ALLOC_SIZE) / sizeof(T);

	static constexpr auto can_alloca(usize count) -> bool {
		return count < max_stack_count;
	}

	LDLT_INLINE UniqueMalloca(void* ptr, usize count)
			: malloca_ptr(ptr),
				data(static_cast<T*>(detail::next_aligned(ptr, align))),
				n(count) {
		new (data) T[n];
	}

	UniqueMalloca(UniqueMalloca const&) = delete;
	UniqueMalloca(UniqueMalloca&&) = delete;
	auto operator=(UniqueMalloca const&) -> UniqueMalloca& = delete;
	auto operator=(UniqueMalloca&&) -> UniqueMalloca& = delete;
	LDLT_INLINE ~UniqueMalloca() {
		for (usize i = 0; i < n; ++i) {
			data[i].~T();
		}
		if (n < max_stack_count) {
#if LDLT_HAS_ALLOCA
			LDLT_FREEA(malloca_ptr);
#endif
		} else {
			std::free(malloca_ptr);
		}
	}
};
} // namespace detail

enum struct Layout : unsigned char {
	colmajor = 0,
	rowmajor = 1,
};

constexpr Layout colmajor = Layout::colmajor;
constexpr Layout rowmajor = Layout::rowmajor;

constexpr auto flip_layout(Layout l) noexcept -> Layout {
	return Layout(1 - u32(l));
}
constexpr auto to_eigen_layout(Layout l) -> int {
	return l == colmajor ? Eigen::ColMajor : Eigen::RowMajor;
}
constexpr auto from_eigen_layout(int l) -> Layout {
	return (unsigned(l) & Eigen::RowMajorBit) == Eigen::RowMajor ? rowmajor
	                                                             : colmajor;
}

static_assert(
		to_eigen_layout(from_eigen_layout(Eigen::ColMajor)) == Eigen::ColMajor,
		".");
static_assert(
		to_eigen_layout(from_eigen_layout(Eigen::RowMajor)) == Eigen::RowMajor,
		".");

namespace detail {
template <Layout L>
struct ElementAccess;

template <>
struct ElementAccess<Layout::colmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, isize row, isize col, isize outer_stride) noexcept -> T* {
		return ptr + (usize(row) + usize(col) * usize(outer_stride));
	}

	template <usize N, typename T>
	LDLT_INLINE static auto
	load_col_pack(T const* ptr, isize /*outer_stride*/) noexcept -> Pack<T, N> {
		return Pack<T, N>::load_unaligned(ptr);
	}
	template <usize N, typename T>
	LDLT_INLINE static void
	store_col_pack(T* ptr, Pack<T, N> pack, isize /*outer_stride*/) noexcept {
		return pack.store_unaligned(ptr);
	}

	using NextRowStride = Eigen::Stride<0, 0>;
	using NextColStride = Eigen::InnerStride<Eigen::Dynamic>;
	LDLT_INLINE static auto next_row_stride(isize outer_stride) noexcept
			-> NextRowStride {
		(void)outer_stride;
		return NextRowStride{};
	}
	LDLT_INLINE static auto next_col_stride(isize outer_stride) noexcept
			-> NextColStride {
		return NextColStride /* NOLINT(modernize-return-braced-init-list) */ (
				outer_stride);
	}

	template <typename T>
	LDLT_INLINE static void
	transpose_if_rowmajor(T* ptr, isize dim, isize outer_stride) {
		(void)ptr, (void)dim, (void)outer_stride;
	}
};

template <>
struct ElementAccess<Layout::rowmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, isize row, isize col, isize outer_stride) noexcept -> T* {
		return ptr + (usize(col) + usize(row) * usize(outer_stride));
	}

	using NextColStride = Eigen::Stride<0, 0>;
	using NextRowStride = Eigen::InnerStride<Eigen::Dynamic>;
	LDLT_INLINE static auto next_col_stride(isize outer_stride) noexcept
			-> NextColStride {
		(void)outer_stride;
		return NextColStride{};
	}
	LDLT_INLINE static auto next_row_stride(isize outer_stride) noexcept
			-> NextRowStride {
		return NextRowStride /* NOLINT(modernize-return-braced-init-list) */ (
				outer_stride);
	}

	template <typename T>
	LDLT_INLINE static void
	transpose_if_rowmajor(T* ptr, isize dim, isize outer_stride) {
		Eigen::Map<                            //
				Eigen::Matrix<                     //
						T,                             //
						Eigen::Dynamic,                //
						Eigen::Dynamic                 //
						>,                             //
				Eigen::Unaligned,                  //
				Eigen::OuterStride<Eigen::Dynamic> //
				>{
				ptr,
				dim,
				dim,
				Eigen::OuterStride<Eigen::Dynamic>(outer_stride),
		}
				.transposeInPlace();
	}
};
} // namespace detail

namespace detail {
template <typename T>
struct unlref {
	using Type = T;
};
template <typename T>
struct unlref<T&> {
	using Type = T;
};

template <typename T>
auto is_eigen_matrix_base_impl(Eigen::MatrixBase<T> const volatile*) -> True;
auto is_eigen_matrix_base_impl(void const volatile*) -> False;

template <typename T>
auto is_eigen_owning_matrix_base_impl(Eigen::PlainObjectBase<T> const volatile*)
		-> True;
auto is_eigen_owning_matrix_base_impl(void const volatile*) -> False;

#define LDLT_DECLVAL(...) (static_cast<auto (*)()->__VA_ARGS__>(nullptr)())

template <typename... Ts>
using Void = void;

template <typename Mat, typename T>
using DataExpr = decltype(static_cast<T*>(LDLT_DECLVAL(Mat&).data()));

template <
		typename Dummy,
		typename Fallback,
		template <typename...>
		class F,
		typename... Ts>
struct DetectedImpl : False {
	using Type = Fallback;
};

template <typename Fallback, template <typename...> class F, typename... Ts>
struct DetectedImpl<Void<F<Ts...>>, Fallback, F, Ts...> : True {
	using Type = F<Ts...>;
};

template <typename Fallback, template <typename...> class F, typename... Ts>
using Detected = typename DetectedImpl<void, Fallback, F, Ts...>::Type;

template <typename T>
using CompTimeColsImpl = Constant<isize, isize(T::ColsAtCompileTime)>;
template <typename T>
using CompTimeRowsImpl = Constant<isize, isize(T::RowsAtCompileTime)>;
template <typename T>
using CompTimeInnerStrideImpl =
		Constant<isize, isize(T::InnerStrideAtCompileTime)>;
template <typename T>
using LayoutImpl =
		Constant<Layout, (bool(T::IsRowMajor) ? rowmajor : colmajor)>;

template <typename T, Layout L>
using EigenMatMap = Eigen::Map<        //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				Eigen::Dynamic,                //
				(L == colmajor)                //
						? Eigen::ColMajor          //
						: Eigen::RowMajor          //
				> const,                       //
		Eigen::Unaligned,                  //
		Eigen::OuterStride<Eigen::Dynamic> //
		>;
template <typename T, Layout L>
using EigenMatMapMut = Eigen::Map<     //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				Eigen::Dynamic,                //
				(L == colmajor)                //
						? Eigen::ColMajor          //
						: Eigen::RowMajor          //
				>,                             //
		Eigen::Unaligned,                  //
		Eigen::OuterStride<Eigen::Dynamic> //
		>;

template <typename T, typename Stride>
using EigenVecMap = Eigen::Map< //
		Eigen::Matrix<              //
				T,                      //
				Eigen::Dynamic,         //
				1                       //
				> const,                //
		Eigen::Unaligned,           //
		Stride                      //
		>;
template <typename T, typename Stride>
using EigenVecMapMut = Eigen::Map< //
		Eigen::Matrix<                 //
				T,                         //
				Eigen::Dynamic,            //
				1                          //
				>,                         //
		Eigen::Unaligned,              //
		Stride                         //
		>;

template <typename T, Layout L>
using ColToVec = EigenVecMap<T, typename ElementAccess<L>::NextRowStride>;
template <typename T, Layout L>
using RowToVec = EigenVecMap<T, typename ElementAccess<L>::NextColStride>;
template <typename T, Layout L>
using ColToVecMut = EigenVecMapMut<T, typename ElementAccess<L>::NextRowStride>;
template <typename T, Layout L>
using RowToVecMut = EigenVecMapMut<T, typename ElementAccess<L>::NextColStride>;

template <typename T>
using VecMap = EigenVecMap<T, Eigen::Stride<0, 0>>;
template <typename T>
using VecMapMut = EigenVecMapMut<T, Eigen::Stride<0, 0>>;

} // namespace detail

template <typename T>
using unref = typename detail::unlref<T&>::Type;

namespace eigen {
template <typename T>
using CompTimeCols =
		detail::Detected<detail::Constant<isize, 0>, detail::CompTimeColsImpl, T>;
template <typename T>
using CompTimeRows =
		detail::Detected<detail::Constant<isize, 0>, detail::CompTimeRowsImpl, T>;
template <typename T>
using CompTimeInnerStride = detail::
		Detected<detail::Constant<isize, 0>, detail::CompTimeInnerStrideImpl, T>;
template <typename T>
using GetLayout = detail::Detected<
		detail::Constant<Layout, Layout(static_cast<unsigned char>(-1))>,
		detail::LayoutImpl,
		T>;
} // namespace eigen

namespace concepts {
LDLT_DEF_CONCEPT(typename T, rvalue_ref, std::is_rvalue_reference<T>::value);
LDLT_DEF_CONCEPT(typename T, lvalue_ref, std::is_lvalue_reference<T>::value);
LDLT_DEF_CONCEPT(
		(template <typename...> class F, typename... Ts),
		detected,
		detail::DetectedImpl<void, void, F, Ts...>::value);

namespace aux {
LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		has_data_expr,
		LDLT_CONCEPT(detected<detail::DataExpr, Mat, T>));

LDLT_DEF_CONCEPT(
		(typename Mat),
		matrix_base,
		decltype(detail::is_eigen_matrix_base_impl(
				static_cast<Mat*>(nullptr)))::value);

LDLT_DEF_CONCEPT(
		(typename Mat),
		is_plain_object_base,
		decltype(detail::is_eigen_owning_matrix_base_impl(
				static_cast<Mat*>(nullptr)))::value);

LDLT_DEF_CONCEPT(
		(typename Mat),
		tmp_matrix,
		(LDLT_CONCEPT(aux::is_plain_object_base<unref<Mat>>) &&
     !LDLT_CONCEPT(lvalue_ref<Mat>)));
} // namespace aux

LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		eigen_view,
		(LDLT_CONCEPT(aux::matrix_base<unref<Mat>>) &&
     LDLT_CONCEPT(aux::has_data_expr<Mat, T const>)));

LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		eigen_view_mut,
		(LDLT_CONCEPT(aux::matrix_base<unref<Mat>>) &&
     LDLT_CONCEPT(aux::has_data_expr<Mat, T>) &&
     !LDLT_CONCEPT(aux::tmp_matrix<Mat>)));

LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		eigen_strided_vector_view,
		(LDLT_CONCEPT(eigen_view<Mat, T>) &&
     (eigen::CompTimeCols<unref<Mat>>::value == 1)));

LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		eigen_strided_vector_view_mut,
		(LDLT_CONCEPT(eigen_view_mut<Mat, T>) &&
     (eigen::CompTimeCols<unref<Mat>>::value == 1)));

LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		eigen_vector_view,
		(LDLT_CONCEPT(eigen_strided_vector_view<Mat, T>) &&
     (eigen::CompTimeInnerStride<unref<Mat>>::value == 1)));

LDLT_DEF_CONCEPT(
		(typename Mat, typename T),
		eigen_vector_view_mut,
		(LDLT_CONCEPT(eigen_strided_vector_view_mut<Mat, T>) &&
     (eigen::CompTimeInnerStride<unref<Mat>>::value == 1)));
} // namespace concepts

LDLT_DEFINE_TAG(from_ptr_size, FromPtrSize);
LDLT_DEFINE_TAG(from_ptr_size_stride, FromPtrSizeStride);
LDLT_DEFINE_TAG(from_ptr_rows_cols_stride, FromPtrRowsColsStride);
LDLT_DEFINE_TAG(from_eigen, FromEigen);

template <typename T>
struct VectorView {
	T const* data;
	isize dim;

	LDLT_INLINE
	VectorView(FromPtrSize /*tag*/, T const* _data, isize _dim) noexcept
			: data(_data), dim(_dim) {}

	LDLT_TEMPLATE(
			typename Vec,
			requires(LDLT_CONCEPT(eigen_vector_view<Vec, T>)),
			LDLT_INLINE VectorView,
			(/*tag*/, FromEigen),
			(vec, Vec const&))
	noexcept
			: data(vec.data()), dim(vec.rows()) {}

	LDLT_INLINE auto ptr(isize index) const noexcept -> T const* {
		return data + index;
	}
	LDLT_INLINE auto operator()(isize index) const noexcept -> T const& {
		return *ptr(index);
	}
	LDLT_INLINE auto segment(isize i, isize size) const noexcept -> VectorView {
		return {
				from_ptr_size,
				data + i,
				size,
		};
	}
	LDLT_INLINE auto to_eigen() const -> detail::VecMap<T> {
		return detail::VecMap<T>(data, Eigen::Index(dim));
	}
};

template <typename T>
struct VectorViewMut {
	T* data;
	isize dim;

	LDLT_INLINE
	VectorViewMut(FromPtrSize /*tag*/, T* _data, isize _dim) noexcept
			: data(_data), dim(_dim) {}

	LDLT_TEMPLATE(
			typename Vec,
			requires(LDLT_CONCEPT(eigen_vector_view_mut<Vec, T>)),
			LDLT_INLINE VectorViewMut,
			(/*tag*/, FromEigen),
			(vec, Vec&&))
	noexcept
			: data(vec.data()), dim(vec.rows()) {}

	LDLT_INLINE auto as_const() const noexcept -> VectorView<T> {
		return {
				from_ptr_size,
				data,
				dim,
		};
	}
	LDLT_INLINE auto ptr(isize index) const noexcept -> T* {
		return data + index;
	}
	LDLT_INLINE auto operator()(isize index) const noexcept -> T& {
		return *ptr(index);
	}
	LDLT_INLINE auto segment(isize i, isize size) const noexcept
			-> VectorViewMut {
		return {
				from_ptr_size,
				data + i,
				size,
		};
	}
	LDLT_INLINE auto to_eigen() const -> detail::VecMapMut<T> {
		return detail::VecMapMut<T>(data, Eigen::Index(dim));
	}
};

template <typename T>
struct StridedVectorView {
	T const* data;
	isize dim;
	isize stride;

	LDLT_INLINE
	StridedVectorView(
			FromPtrSizeStride /*tag*/,
			T const* _data,
			isize _dim,
			isize _stride) noexcept
			: data(_data), dim(_dim), stride(_stride) {}

	LDLT_TEMPLATE(
			typename Vec,
			requires(LDLT_CONCEPT(eigen_strided_vector_view<Vec, T>)),
			LDLT_INLINE StridedVectorView,
			(/*tag*/, FromEigen),
			(vec, Vec const&))
	noexcept
			: data(vec.data()), dim(vec.rows()), stride(vec.innerStride()) {}

	LDLT_INLINE auto ptr(isize index) const noexcept -> T const* {
		return data + stride * index;
	}
	LDLT_INLINE auto operator()(isize index) const noexcept -> T const& {
		return *ptr(index);
	}
	LDLT_INLINE auto segment(isize i, isize size) const noexcept
			-> StridedVectorView {
		return {
				from_ptr_size_stride,
				data + stride * i,
				size,
				stride,
		};
	}
	LDLT_INLINE auto to_eigen() const
			-> detail::EigenVecMap<T, Eigen::InnerStride<Eigen::Dynamic>> {
		return detail::EigenVecMap<T, Eigen::InnerStride<Eigen::Dynamic>>(
				data,
				Eigen::Index(dim),
				Eigen::Index(1),
				Eigen::InnerStride<Eigen::Dynamic>(Eigen::Index(stride)));
	}
};

template <typename T>
struct StridedVectorViewMut {
	T* data;
	isize dim;
	isize stride;

	LDLT_INLINE
	StridedVectorViewMut(
			FromPtrSizeStride /*tag*/, T* _data, isize _dim, isize _stride) noexcept
			: data(_data), dim(_dim), stride(_stride) {}

	LDLT_TEMPLATE(
			typename Vec,
			requires(LDLT_CONCEPT(eigen_strided_vector_view_mut<Vec, T>)),
			LDLT_INLINE StridedVectorViewMut,
			(/*tag*/, FromEigen),
			(vec, Vec&&))
	noexcept
			: data(vec.data()), dim(vec.rows()), stride(vec.innerStride()) {}

	LDLT_INLINE auto as_const() const noexcept -> StridedVectorView<T> {
		return {
				from_ptr_size_stride,
				data,
				dim,
				stride,
		};
	}
	LDLT_INLINE auto ptr(isize index) const noexcept -> T* {
		return data + stride * index;
	}
	LDLT_INLINE auto operator()(isize index) const noexcept -> T& {
		return *ptr(index);
	}
	LDLT_INLINE auto segment(isize i, isize size) const noexcept
			-> StridedVectorViewMut {
		return {
				from_ptr_size_stride,
				data + stride * i,
				size,
				stride,
		};
	}
	LDLT_INLINE auto to_eigen() const
			-> detail::EigenVecMapMut<T, Eigen::InnerStride<Eigen::Dynamic>> {
		return detail::EigenVecMapMut<T, Eigen::InnerStride<Eigen::Dynamic>>(
				data,
				Eigen::Index(dim),
				Eigen::Index(1),
				Eigen::InnerStride<Eigen::Dynamic>(Eigen::Index(stride)));
	}
};

template <typename T, Layout L>
struct MatrixView {
	T const* data;
	isize rows;
	isize cols;
	isize outer_stride;

	LDLT_INLINE MatrixView(
			FromPtrRowsColsStride /*tag*/,
			T const* _data,
			isize _rows,
			isize _cols,
			isize _outer_stride) noexcept
			: data(_data), rows(_rows), cols(_cols), outer_stride(_outer_stride) {}

	LDLT_TEMPLATE(
			typename Mat,
			requires(
					LDLT_CONCEPT(eigen_view<Mat, T>) &&
					eigen::GetLayout<unref<Mat>>::value == L),
			LDLT_INLINE MatrixView,
			(/*tag*/, FromEigen),
			(mat, Mat const&))
	noexcept
			: data(mat.data()),
				rows(mat.rows()),
				cols(mat.cols()),
				outer_stride(mat.outerStride()) {}

	LDLT_INLINE auto ptr(isize row, isize col) const noexcept -> T const* {
		return detail::ElementAccess<L>::offset(data, row, col, outer_stride);
	}
	LDLT_INLINE auto operator()(isize row, isize col) const noexcept -> T const& {
		return *ptr(row, col);
	}
	LDLT_INLINE auto
	block(isize row, isize col, isize nrows, isize ncols) const noexcept
			-> MatrixView {
		return {
				from_ptr_rows_cols_stride,
				detail::ElementAccess<L>::offset(data, row, col, outer_stride),
				nrows,
				ncols,
				outer_stride,
		};
	}

private:
	LDLT_INLINE auto
	col_impl(detail::Constant<Layout, colmajor> /*tag*/, isize c) const noexcept
			-> VectorView<T> {
		return {
				from_ptr_size,
				data + c * outer_stride,
				rows,
		};
	}
	LDLT_INLINE auto
	col_impl(detail::Constant<Layout, rowmajor> /*tag*/, isize c) const noexcept
			-> StridedVectorView<T> {
		return {
				from_ptr_size_stride,
				data + c,
				rows,
				outer_stride,
		};
	}

public:
	LDLT_INLINE auto col(isize c) const noexcept -> detail::
			Conditional<(L == colmajor), VectorView<T>, StridedVectorView<T>> {
		return col_impl(detail::Constant<Layout, L>{}, c);
	}
	LDLT_INLINE auto row(isize r) const noexcept -> detail::
			Conditional<(L == rowmajor), VectorView<T>, StridedVectorView<T>> {
		return trans().col(r);
	}
	LDLT_INLINE auto trans() const noexcept
			-> MatrixView<T, ldlt::flip_layout(L)> {
		return {
				from_ptr_rows_cols_stride,
				data,
				cols,
				rows,
				outer_stride,
		};
	}
	LDLT_INLINE auto to_eigen() const noexcept -> detail::EigenMatMap<T, L> {
		return detail::EigenMatMap<T, L>(
				data,
				Eigen::Index(rows),
				Eigen::Index(cols),
				Eigen::OuterStride<Eigen::Dynamic>(Eigen::Index(outer_stride)));
	}
};

template <typename T, Layout L>
struct MatrixViewMut {
	T* data;
	isize rows;
	isize cols;
	isize outer_stride;

	LDLT_INLINE MatrixViewMut(
			FromPtrRowsColsStride /*tag*/,
			T* _data,
			isize _rows,
			isize _cols,
			isize _outer_stride) noexcept
			: data(_data), rows(_rows), cols(_cols), outer_stride(_outer_stride) {}

	LDLT_TEMPLATE(
			typename Mat,
			requires(
					LDLT_CONCEPT(eigen_view<Mat, T>) &&
					eigen::GetLayout<unref<Mat>>::value == L),
			LDLT_INLINE MatrixViewMut,
			(/*tag*/, FromEigen),
			(mat, Mat&&))
	noexcept
			: data(mat.data()),
				rows(mat.rows()),
				cols(mat.cols()),
				outer_stride(mat.outerStride()) {}

	LDLT_INLINE auto ptr(isize row, isize col) const noexcept -> T* {
		return detail::ElementAccess<L>::offset(data, row, col, outer_stride);
	}
	LDLT_INLINE auto operator()(isize row, isize col) const noexcept -> T& {
		return *ptr(row, col);
	}
	LDLT_INLINE auto
	block(isize row, isize col, isize nrows, isize ncols) const noexcept
			-> MatrixViewMut {
		return {
				from_ptr_rows_cols_stride,
				detail::ElementAccess<L>::offset(data, row, col, outer_stride),
				nrows,
				ncols,
				outer_stride,
		};
	}

private:
	LDLT_INLINE auto
	col_impl(detail::Constant<Layout, colmajor> /*tag*/, isize c) const noexcept
			-> VectorViewMut<T> {
		return {
				from_ptr_size,
				data + c * outer_stride,
				rows,
		};
	}
	LDLT_INLINE auto
	col_impl(detail::Constant<Layout, rowmajor> /*tag*/, isize c) const noexcept
			-> StridedVectorViewMut<T> {
		return {
				from_ptr_size_stride,
				data + c,
				rows,
				outer_stride,
		};
	}

public:
	LDLT_INLINE auto col(isize c) const noexcept -> detail::
			Conditional<(L == colmajor), VectorViewMut<T>, StridedVectorViewMut<T>> {
		return col_impl(detail::Constant<Layout, L>{}, c);
	}
	LDLT_INLINE auto row(isize r) const noexcept -> detail::
			Conditional<(L == rowmajor), VectorViewMut<T>, StridedVectorViewMut<T>> {
		return trans().col(r);
	}
	LDLT_INLINE auto trans() const noexcept
			-> MatrixViewMut<T, ldlt::flip_layout(L)> {
		return {
				from_ptr_rows_cols_stride,
				data,
				cols,
				rows,
				outer_stride,
		};
	}
	LDLT_INLINE auto to_eigen() const noexcept -> detail::EigenMatMapMut<T, L> {
		return detail::EigenMatMapMut<T, L>(
				data,
				Eigen::Index(rows),
				Eigen::Index(cols),
				Eigen::OuterStride<Eigen::Dynamic>(Eigen::Index(outer_stride)));
	}
	LDLT_INLINE auto as_const() const noexcept -> MatrixView<T, L> {
		return {
				from_ptr_rows_cols_stride,
				data,
				rows,
				cols,
				outer_stride,
		};
	}
};

template <typename T>
struct LdltView {
	MatrixView<T, colmajor> l;
	VectorView<T> d;
};
template <typename T>
struct LdltViewMut {
	MatrixViewMut<T, colmajor> l;
	VectorViewMut<T> d;

	LDLT_INLINE constexpr auto as_const() const noexcept -> LdltView<T> {
		return {l.as_const(), d.as_const()};
	}
};

namespace detail {
template <typename T>
void noalias_mul_add(
		MatrixViewMut<T, colmajor> dst,
		MatrixView<T, colmajor> lhs,
		MatrixView<T, colmajor> rhs,
		T factor) {

	if ((dst.cols == 0) || (dst.rows == 0) || (lhs.cols == 0)) {
		return;
	}

	if (dst.cols == 1 && dst.rows == 1) {
		// dot
		auto rhs_col = rhs.col(0);
		auto lhs_row = lhs.row(0);
		auto lhs_as_col = lhs.col(0);
		lhs_as_col.dim = lhs_row.dim;
		if (lhs_row.stride == 1) {
			dst(0, 0) += factor * lhs_as_col.to_eigen().dot(rhs_col.to_eigen());
		} else {
			dst(0, 0) += factor * lhs_row.to_eigen().dot(rhs_col.to_eigen());
		}
	} else if (dst.cols == 1) {
		// gemv
		auto rhs_col = rhs.col(0);
		auto dst_col = dst.col(0);
		dst_col.to_eigen().noalias().operator+=(
				factor*(lhs.to_eigen().operator*(rhs_col.to_eigen())));
	}

#if !EIGEN_VERSION_AT_LEAST(3, 3, 8)
	else if ((dst.rows < 20) && (dst.cols < 20) && (rhs.rows < 20)) {
		// gemm
		// workaround for eigen 3.3.7 bug:
		// https://gitlab.com/libeigen/eigen/-/issues/1562
		using Stride = Eigen::OuterStride<Eigen::Dynamic>;
		using Mat = Eigen::
				Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 20, 20>;
		using MapMut = Eigen::Map<Mat, Eigen::Unaligned, Stride>;
		using Map = Eigen::Map<Mat const, Eigen::Unaligned, Stride>;

		MapMut(dst.data, dst.rows, dst.cols, Stride(dst.outer_stride))
				.noalias()
				.
				operator+=(
						Map(lhs.data, lhs.rows, lhs.cols, Stride(lhs.outer_stride))
								.
								operator*(
										Map(rhs.data, rhs.rows, rhs.cols, Stride(rhs.outer_stride))
												.
												operator*(factor)));
	}
#endif

	else {
		// gemm
		dst.to_eigen().noalias().operator+=(
				lhs.to_eigen().operator*(rhs.to_eigen().operator*(factor)));
	}
}

template <typename T>
void noalias_mul_add_vec(
		VectorViewMut<T> dst,
		MatrixView<T, colmajor> lhs,
		VectorView<T> rhs,
		T factor) {
	detail::noalias_mul_add<T>(
			{
					from_ptr_rows_cols_stride,
					dst.data,
					dst.dim,
					1,
					0,
			},
			lhs,
			{
					from_ptr_rows_cols_stride,
					rhs.data,
					rhs.dim,
					1,
					0,
			},
			LDLT_FWD(factor));
}

template <typename T>
auto dot(StridedVectorView<T> lhs, VectorView<T> rhs) -> T {
	auto out = T(0);
	detail::noalias_mul_add<T>(
			{
					from_ptr_rows_cols_stride,
					std::addressof(out),
					1,
					1,
					0,
			},
			{
					from_ptr_rows_cols_stride,
					lhs.data,
					1,
					lhs.dim,
					lhs.stride,
			},
			{
					from_ptr_rows_cols_stride,
					rhs.data,
					rhs.dim,
					1,
					0,
			},
			1);
	return out;
}
template <typename T>
void assign_cwise_prod(
		VectorViewMut<T> out, StridedVectorView<T> lhs, VectorView<T> rhs) {
	out.to_eigen() = lhs.to_eigen().cwiseProduct(rhs.to_eigen());
}
template <typename T>
void assign_scalar_prod(VectorViewMut<T> out, T factor, VectorView<T> in) {
	out.to_eigen() = in.to_eigen().operator*(factor);
}

template <typename T>
void trans_tr_unit_up_solve_in_place_on_right(
		MatrixView<T, colmajor> tr, MatrixViewMut<T, colmajor> rhs) {
	if (rhs.cols == 1) {
		tr.to_eigen()
				.transpose()
				.template triangularView<Eigen::UnitUpper>()
				.template solveInPlace<Eigen::OnTheRight>(rhs.col(0).to_eigen());
	} else {
		tr.to_eigen()
				.transpose()
				.template triangularView<Eigen::UnitUpper>()
				.template solveInPlace<Eigen::OnTheRight>(rhs.to_eigen());
	}
}

template <typename T>
void apply_diag_inv_on_right(
		MatrixViewMut<T, colmajor> out,
		VectorView<T> d,
		MatrixView<T, colmajor> in) {
	if (out.cols == 1) {
		out.col(0).to_eigen() =
				in.col(0).to_eigen().operator*(d.to_eigen().asDiagonal().inverse());
	} else {
		out.to_eigen() =
				in.to_eigen().operator*(d.to_eigen().asDiagonal().inverse());
	}
}
template <typename T>
void apply_diag_on_right(
		MatrixViewMut<T, colmajor> out,
		VectorView<T> d,
		MatrixView<T, colmajor> in) {
	if (out.cols == 1) {
		out.col(0).to_eigen() =
				in.col(0).to_eigen().operator*(d.to_eigen().asDiagonal());
	} else {
		out.to_eigen() = in.to_eigen().operator*(d.to_eigen().asDiagonal());
	}
}

template <typename T>
void noalias_mul_sub_tr_lo(
		MatrixViewMut<T, colmajor> out,
		MatrixView<T, colmajor> lhs,
		MatrixView<T, rowmajor> rhs) {
	if (lhs.cols == 1) {
		out.to_eigen().template triangularView<Eigen::Lower>().operator-=(
				lhs.col(0).to_eigen().operator*(rhs.row(0).to_eigen()));
	} else {
		out.to_eigen().template triangularView<Eigen::Lower>().operator-=(
				lhs.to_eigen().operator*(rhs.to_eigen()));
	}
}

LDLT_EXPLICIT_TPL_DECL(4, noalias_mul_add<f64>);
LDLT_EXPLICIT_TPL_DECL(3, assign_cwise_prod<f64>);
LDLT_EXPLICIT_TPL_DECL(3, assign_scalar_prod<f64>);
LDLT_EXPLICIT_TPL_DECL(2, trans_tr_unit_up_solve_in_place_on_right<f64>);
LDLT_EXPLICIT_TPL_DECL(3, apply_diag_inv_on_right<f64>);
LDLT_EXPLICIT_TPL_DECL(3, apply_diag_on_right<f64>);
LDLT_EXPLICIT_TPL_DECL(3, noalias_mul_sub_tr_lo<f64>);

LDLT_EXPLICIT_TPL_DECL(4, noalias_mul_add<f32>);
LDLT_EXPLICIT_TPL_DECL(3, assign_cwise_prod<f32>);
LDLT_EXPLICIT_TPL_DECL(3, assign_scalar_prod<f32>);
LDLT_EXPLICIT_TPL_DECL(2, trans_tr_unit_up_solve_in_place_on_right<f32>);
LDLT_EXPLICIT_TPL_DECL(3, apply_diag_inv_on_right<f32>);
LDLT_EXPLICIT_TPL_DECL(3, apply_diag_on_right<f32>);
LDLT_EXPLICIT_TPL_DECL(3, noalias_mul_sub_tr_lo<f32>);
} // namespace detail
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_VIEWS_HPP_UGNXAQSBS */
