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
	offset(T* ptr, usize row, usize col, usize outer_stride) noexcept -> T* {
		return ptr + (row + col * outer_stride);
	}
};

template <>
struct MemoryOffset<Layout::rowmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, usize row, usize col, usize outer_stride) noexcept -> T* {
		return ptr + (col + row * outer_stride);
	}
};
} // namespace detail

template <typename Scalar, Layout L>
struct MatrixView {
	Scalar const* data;
	usize dim;

	LDLT_INLINE auto operator()(usize row, usize col) const noexcept
			-> Scalar const& {
		return *detail::MemoryOffset<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct LowerTriangularMatrixView {
	Scalar const* data;
	usize dim;

	LDLT_INLINE auto operator()(usize row, usize col) const noexcept
			-> Scalar const& {
		return *detail::MemoryOffset<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct LowerTriangularMatrixViewMut {
	Scalar* data;
	usize dim;

	LDLT_INLINE auto as_const() const noexcept
			-> LowerTriangularMatrixView<Scalar, L> {
		return {data, dim};
	}
	LDLT_INLINE auto operator()(usize row, usize col) const noexcept -> Scalar& {
		return *detail::MemoryOffset<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct LowerTriangularMatrix {
private:
	std::vector<Scalar> _data = {};
	usize _dim = 0;

public:
	LowerTriangularMatrix() noexcept = default;
	LowerTriangularMatrix(WithDim /*tag*/, usize dim)
			: _data(dim * dim), _dim(dim) {}
	LowerTriangularMatrix(WithDimUninit /*tag*/, usize dim)
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
	usize dim;

	HEDLEY_ALWAYS_INLINE auto operator()(usize index) const noexcept
			-> Scalar const& {
		return *(data + index);
	}
};

template <typename Scalar>
struct DiagonalMatrixViewMut {
	Scalar* data;
	usize dim;
	LDLT_INLINE auto as_const() const noexcept -> DiagonalMatrixView<Scalar> {
		return {data, dim};
	}
	HEDLEY_ALWAYS_INLINE auto operator()(usize index) const noexcept -> Scalar& {
		return *(data + index);
	}
};

template <typename Scalar>
struct DiagonalMatrix {
private:
	std::vector<Scalar> _data{};
	usize _dim = 0;

public:
	DiagonalMatrix() noexcept = default;
	DiagonalMatrix(WithDim /*tag*/, usize dim) : _data(dim), _dim(dim) {}
	DiagonalMatrix(WithDimUninit /*tag*/, usize dim)
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
template <typename Scalar, Layout L>
struct DiagAccGenerator {
	LowerTriangularMatrixView<Scalar, L> l;
	DiagonalMatrixView<Scalar> d;
	usize j;
	usize k;

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
	LDLT_INLINE auto add_pack(Pack acc) -> Pack {
		Pack ljk = Pack::load_unaligned(&l(j, k));
		Pack dk = Pack::load_unaligned(&d(k));
		k += NativePackInfo<Scalar>::N;

		return Pack::fmadd(ljk.mul(ljk), dk, acc);
	}

	LDLT_INLINE auto sub_pack(Pack acc) -> Pack {
		Pack ljk = Pack::load_unaligned(&l(j, k));
		Pack dk = Pack::load_unaligned(&d(k));
		k += NativePackInfo<Scalar>::N;

		return Pack::fmsub(ljk.mul(ljk), dk, acc);
	}
};

template <typename Scalar, Layout L>
struct LowerTriAccGenerator {
	LowerTriangularMatrixView<Scalar, L> l;
	DiagonalMatrixView<Scalar> d;
	usize i;
	usize j;
	usize k;

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
	LDLT_INLINE auto add_pack(Pack acc) -> Pack {
		Pack lik = Pack::load_unaligned(&l(i, k));
		Pack ljk = Pack::load_unaligned(&l(j, k));
		Pack dk = Pack::load_unaligned(&d(k));
		k += NativePackInfo<Scalar>::N;

		return Pack::fmadd(lik.mul(ljk), dk, acc);
	}

	LDLT_INLINE auto sub_pack(Pack acc) -> Pack {
		Pack lik = Pack::load_unaligned(&l(i, k));
		Pack ljk = Pack::load_unaligned(&l(j, k));
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
struct Vectorized {
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
		Scalar psum_reduced = psum.sum();
		return psum_reduced + rem_sum;
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

	usize dim = out_l.dim;
	auto in_l = out_l.as_const();
	auto in_d = out_d.as_const();

	for (usize j = 0; j < dim; ++j) {
		Scalar acc_d = Scalar(
				acc(j, detail::DiagAccGenerator<Scalar, OutL>{in_l, in_d, j, 0}));
		Scalar debug_acc_d = Scalar(accumulators::Sequential<Scalar>{}(
				j, detail::DiagAccGenerator<Scalar, OutL>{in_l, in_d, j, 0}));

		fmt::print(
				"count: {}, diff: {}, acc: {}, debug: {}\n",
				j,
				acc_d - debug_acc_d,
				acc_d,
				debug_acc_d);

		out_d(j) = in_matrix(j, j) - acc_d;

		for (usize i = 0; i < j; ++i) {
			out_l(i, j) = Scalar(0);
		}
		out_l(j, j) = Scalar(1);
		for (usize i = j + 1; i < dim; ++i) {
			Scalar acc_l = Scalar(acc(
					j, detail::LowerTriAccGenerator<Scalar, OutL>{in_l, in_d, i, j, 0}));
			Scalar debug_acc_l = Scalar(accumulators::Sequential<Scalar>{}(
					j, detail::LowerTriAccGenerator<Scalar, OutL>{in_l, in_d, i, j, 0}));

			// fmt::print(
			// 		"count: {}, diff: {}, acc: {}, debug: {}\n",
			// 		j,
			// 		acc_l - debug_acc_l,
			// 		acc_l,
			// 		debug_acc_l);
			out_l(i, j) = (in_matrix(i, j) - acc_l) / out_d(j);
		}
	}
}
} // namespace ldlt

#endif /* end of include guard LDLT_LDLT_HPP_FDFNWYGES */
