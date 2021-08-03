#ifndef LDLT_LDLT_HPP_FDFNWYGES
#define LDLT_LDLT_HPP_FDFNWYGES

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include <vector>
#include <fmt/ostream.h>

namespace ldlt {
using usize = decltype(sizeof(0));
using f32 = float;
using f64 = double;

LDLT_DEFINE_TAG(with_dim, WithDim);
LDLT_DEFINE_TAG(with_dim_uninit, WithDimUninit);

enum struct Layout {
	colmajor,
	rowmajor,
};

constexpr Layout colmajor = Layout::colmajor;
constexpr Layout rowmajor = Layout::rowmajor;

namespace detail {
template <Layout L>
struct MemoryOffset;

template <>
struct MemoryOffset<Layout::colmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, i32 row, i32 col, i32 outer_stride) noexcept -> T* {
		return ptr + (usize(row) + usize(col) * usize(outer_stride));
	}
};

template <>
struct MemoryOffset<Layout::rowmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, i32 row, i32 col, i32 outer_stride) noexcept -> T* {
		return ptr + (usize(col) + usize(row) * usize(outer_stride));
	}
};
} // namespace detail

template <typename Scalar, Layout L>
struct MatrixView {
	Scalar const* data;
	i32 dim;

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept
			-> Scalar const& {
		return *detail::MemoryOffset<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct LowerTriangularMatrixView {
	Scalar const* data;
	i32 dim;

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept
			-> Scalar const& {
		return *detail::MemoryOffset<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct LowerTriangularMatrixViewMut {
	Scalar* data;
	i32 dim;

	LDLT_INLINE auto as_const() const noexcept
			-> LowerTriangularMatrixView<Scalar, L> {
		return {data, dim};
	}
	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept -> Scalar& {
		return *detail::MemoryOffset<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct LowerTriangularMatrix {
private:
	std::vector<Scalar> _data = {};
	i32 _dim = 0;

public:
	LowerTriangularMatrix() noexcept = default;
	LowerTriangularMatrix(WithDim /*tag*/, i32 dim)
			: _data(dim * dim), _dim(dim) {}
	LowerTriangularMatrix(WithDimUninit /*tag*/, i32 dim)
			: LowerTriangularMatrix(with_dim, dim) {}

	LDLT_INLINE auto as_view() const noexcept
			-> LowerTriangularMatrixView<Scalar, L> {
		return {_data.data(), _dim};
	}
	LDLT_INLINE auto as_mut() noexcept
			-> LowerTriangularMatrixViewMut<Scalar, L> {
		return {_data.data(), _dim};
	}
};

template <typename Scalar>
struct DiagonalMatrixView {
	Scalar const* data;
	i32 dim;

	HEDLEY_ALWAYS_INLINE auto operator()(i32 index) const noexcept
			-> Scalar const& {
		return *(data + index);
	}
};

template <typename Scalar>
struct DiagonalMatrixViewMut {
	Scalar* data;
	i32 dim;
	LDLT_INLINE auto as_const() const noexcept -> DiagonalMatrixView<Scalar> {
		return {data, dim};
	}
	HEDLEY_ALWAYS_INLINE auto operator()(i32 index) const noexcept -> Scalar& {
		return *(data + index);
	}
};

template <typename Scalar>
struct DiagonalMatrix {
private:
	std::vector<Scalar> _data{};
	i32 _dim = 0;

public:
	DiagonalMatrix() noexcept = default;
	DiagonalMatrix(WithDim /*tag*/, i32 dim) : _data(dim), _dim(dim) {}
	DiagonalMatrix(WithDimUninit /*tag*/, i32 dim)
			: DiagonalMatrix(with_dim, dim) {}

	LDLT_INLINE auto as_view() const noexcept -> DiagonalMatrixView<Scalar> {
		return {_data.data(), _dim};
	}
	LDLT_INLINE auto as_mut() noexcept -> DiagonalMatrixViewMut<Scalar> {
		return {_data.data(), _dim};
	}
};

#ifdef __clang__
#define LDLT_FP_PRAGMA _Pragma("STDC FP_CONTRACT ON")
#else
#define LDLT_FP_PRAGMA
#endif

namespace detail {
template <Layout L>
struct MatrixLoadRowBlock;

template <>
struct MatrixLoadRowBlock<Layout::rowmajor> {
	template <usize N, typename Scalar>
	LDLT_INLINE static auto load_pack(Scalar const* p, i32 stride) noexcept
			-> Pack<Scalar, N> {
		(void)stride;
		return Pack<Scalar, N>::load_unaligned(p);
	}
};
template <>
struct MatrixLoadRowBlock<Layout::colmajor> {
	template <usize N, typename Scalar>
	LDLT_INLINE static auto load_pack(Scalar const* p, i32 stride) noexcept
			-> Pack<Scalar, N> {
		return Pack<Scalar, N>::load_gather(p, stride);
	}
};

template <typename Scalar, Layout L>
struct DiagAccGenerator {
	LowerTriangularMatrixView<Scalar, L> l;
	DiagonalMatrixView<Scalar> d;
	i32 j;
	i32 k;

	LDLT_INLINE auto add(Scalar acc) -> Scalar {
		Scalar ljk = l(j, k);
		Scalar dk = d(k);
		++k;

		Scalar prod = ljk * ljk;
		{
			LDLT_FP_PRAGMA
			return acc + prod * dk;
		}
	}
	LDLT_INLINE auto sub(Scalar comp) -> Scalar {
		Scalar ljk = l(j, k);
		Scalar dk = d(k);
		++k;

		Scalar prod = ljk * ljk;
		{
			LDLT_FP_PRAGMA
			return prod * dk - comp;
		}
	}

	using Pack = NativePack<Scalar>;
	using PackInfo = NativePackInfo<Scalar>;
	LDLT_INLINE auto add_pack(Pack acc) -> Pack {
		Pack ljk =
				MatrixLoadRowBlock<L>::template load_pack<PackInfo::N>(&l(j, k), l.dim);
		Pack dk = Pack::load_unaligned(&d(k));
		k += i32{NativePackInfo<Scalar>::N};

		return Pack::fmadd(ljk.mul(ljk), dk, acc);
	}

	LDLT_INLINE auto sub_pack(Pack acc) -> Pack {
		Pack ljk =
				MatrixLoadRowBlock<L>::template load_pack<PackInfo::N>(&l(j, k), l.dim);
		Pack dk = Pack::load_unaligned(&d(k));
		k += i32{NativePackInfo<Scalar>::N};

		return Pack::fmsub(ljk.mul(ljk), dk, acc);
	}
};

template <typename Scalar, Layout L>
struct LowerTriAccGenerator {
	LowerTriangularMatrixView<Scalar, L> l;
	DiagonalMatrixView<Scalar> d;
	i32 i;
	i32 j;
	i32 k;

	LDLT_INLINE auto add(Scalar acc) -> Scalar {
		Scalar lik = l(i, k);
		Scalar ljk = l(j, k);
		Scalar dk = d(k);
		++k;

		Scalar prod = lik * ljk;
		{
			LDLT_FP_PRAGMA
			return acc + prod * dk;
		}
	}
	LDLT_INLINE auto sub(Scalar comp) -> Scalar {
		Scalar lik = l(i, k);
		Scalar ljk = l(j, k);
		Scalar dk = d(k);
		++k;

		Scalar prod = lik * ljk;
		{
			LDLT_FP_PRAGMA
			return prod * dk - comp;
		}
	}

	using Pack = NativePack<Scalar>;
	using PackInfo = NativePackInfo<Scalar>;
	LDLT_INLINE auto add_pack(Pack acc) -> Pack {
		Pack lik =
				MatrixLoadRowBlock<L>::template load_pack<PackInfo::N>(&l(i, k), l.dim);
		Pack ljk =
				MatrixLoadRowBlock<L>::template load_pack<PackInfo::N>(&l(j, k), l.dim);
		Pack dk = Pack::load_unaligned(&d(k));
		k += i32{NativePackInfo<Scalar>::N};

		return Pack::fmadd(lik.mul(ljk), dk, acc);
	}

	LDLT_INLINE auto sub_pack(Pack acc) -> Pack {
		Pack lik =
				MatrixLoadRowBlock<L>::template load_pack<PackInfo::N>(&l(i, k), l.dim);
		Pack ljk =
				MatrixLoadRowBlock<L>::template load_pack<PackInfo::N>(&l(j, k), l.dim);
		Pack dk = Pack::load_unaligned(&d(k));
		k += NativePackInfo<Scalar>::N;

		return Pack::fmsub(lik.mul(ljk), dk, acc);
	}
};
} // namespace detail

namespace accumulators {
template <typename Scalar>
struct Sequential {
	template <typename Fn>
	LDLT_INLINE auto operator()(usize count, Fn fn) const -> Scalar {
		Scalar sum(0);
		for (usize k = 0; k < count; ++k) {
			sum = fn.add(sum);
		}
		return sum;
	}
};

template <typename Scalar>
struct SequentialVectorized {
	using Pack = detail::NativePack<Scalar>;
	using PackInfo = detail::NativePackInfo<Scalar>;

	template <typename Fn>
	LDLT_INLINE auto operator()(usize count, Fn fn) const -> Scalar {
		Pack psum = Pack::zero();
		constexpr usize N = PackInfo::N;
		usize div = count / N;
		usize rem = count % N;
		for (usize k = 0; k < div; ++k) {
			psum = fn.add_pack(psum);
		}

		Scalar rem_sum = Scalar(0);
		for (usize k = 0; k < rem; ++k) {
			rem_sum = fn.add(rem_sum);
		}
		return psum.sum() + rem_sum;
	}
};

template <typename Scalar>
struct Kahan {
	template <typename Fn>
	LDLT_INLINE auto operator()(usize count, Fn fn) const -> Scalar {
		Scalar sum(0);
		Scalar c(0);

		for (usize k = 0; k < count; ++k) {
			Scalar y = fn.sub(c);
			Scalar t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		return sum;
	}
};

template <typename Scalar>
struct KahanVectorized {
	using Pack = detail::NativePack<Scalar>;
	using PackInfo = detail::NativePackInfo<Scalar>;

	template <typename Fn>
	LDLT_INLINE auto operator()(usize count, Fn fn) const -> Scalar {
		constexpr usize N = PackInfo::N;

		Pack psum = Pack::zero();
		Pack pc = Pack::zero();
		usize div = count / N;
		usize rem = count % N;
		for (usize k = 0; k < div; ++k) {
			Pack py = fn.sub_pack(pc);
			Pack pt = psum.add(py);
			pc = (pt.sub(psum)).sub(py);
			psum = pt;
		}

		Scalar rem_sum(0);
		Scalar c(0);

		for (usize k = 0; k < rem; ++k) {
			Scalar y = fn.sub(c);
			Scalar t = rem_sum + y;
			c = (t - rem_sum) - y;
			rem_sum = t;
		}

		return psum.sum() + rem_sum;
	}
};
} // namespace accumulators

template <
		typename Scalar,
		Layout OutL,
		Layout InL,
		typename AccumuluateFn = accumulators::Sequential<Scalar>>
void factorize_ldlt_unblocked(
		LowerTriangularMatrixViewMut<Scalar, OutL> out_l,
		DiagonalMatrixViewMut<Scalar> out_d,
		MatrixView<Scalar, InL> in_matrix,
		AccumuluateFn acc = {}) {

	i32 dim = out_l.dim;
	auto in_l = out_l.as_const();
	auto in_d = out_d.as_const();

	for (i32 j = 0; j < dim; ++j) {
		Scalar acc_d =
				acc(usize(j), detail::DiagAccGenerator<Scalar, OutL>{in_l, in_d, j, 0});

		out_d(j) = in_matrix(j, j) - acc_d;

		for (i32 i = 0; i < j; ++i) {
			out_l(i, j) = Scalar(0);
		}
		out_l(j, j) = Scalar(1);
		for (i32 i = j + 1; i < dim; ++i) {
			Scalar acc_l =
					acc(usize(j),
			        detail::LowerTriAccGenerator<Scalar, OutL>{in_l, in_d, i, j, 0});

			out_l(i, j) = (in_matrix(i, j) - acc_l) / out_d(j);
		}
	}
}
} // namespace ldlt

#endif /* end of include guard LDLT_LDLT_HPP_FDFNWYGES */
