#ifndef LDLT_LDLT_HPP_FDFNWYGES
#define LDLT_LDLT_HPP_FDFNWYGES

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include <memory>
#include <vector>
#include <Eigen/Core>

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

template <typename Scalar>
struct ArrayGenerator {
	Scalar const* mem;

	LDLT_INLINE auto add(Scalar acc) -> Scalar {
		Scalar x = *mem + acc;
		++mem;
		return x;
	}
	LDLT_INLINE auto sub(Scalar comp) -> Scalar {
		Scalar x = *mem - comp;
		++mem;
		return x;
	}
};
} // namespace detail

namespace detail {
template <typename T>
using VecMapStridedMut = Eigen::Map<   //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				1                              //
				>,                             //
		Eigen::Unaligned,                  //
		Eigen::InnerStride<Eigen::Dynamic> //
		>;

template <typename T>
using VecMapStrided = Eigen::Map<      //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				1                              //
				> const,                       //
		Eigen::Unaligned,                  //
		Eigen::InnerStride<Eigen::Dynamic> //
		>;

template <typename T>
using VecMap = Eigen::Map< //
		Eigen::Matrix<         //
				T,                 //
				Eigen::Dynamic,    //
				1                  //
				> const,           //
		Eigen::Unaligned       //
		>;

template <typename T>
using VecMapMut = Eigen::Map< //
		Eigen::Matrix<            //
				T,                    //
				Eigen::Dynamic,       //
				1                     //
				>,                    //
		Eigen::Unaligned          //
		>;

template <typename T>
using MatMap = Eigen::Map<             //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				Eigen::Dynamic                 //
				> const,                       //
		Eigen::Unaligned,                  //
		Eigen::OuterStride<Eigen::Dynamic> //
		>;

template <typename T>
using MatMapMut = Eigen::Map<          //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				Eigen::Dynamic                 //
				>,                             //
		Eigen::Unaligned,                  //
		Eigen::OuterStride<Eigen::Dynamic> //
		>;
} // namespace detail

template <typename Scalar>
void factorize_ldlt_unblocked(
		LowerTriangularMatrixViewMut<Scalar, colmajor> out_l,
		DiagonalMatrixViewMut<Scalar> out_d,
		MatrixView<Scalar, colmajor> in_matrix) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2

	i32 dim = out_l.dim;

	auto workspace = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(dim);

	for (i32 j = 0; j < dim; ++j) {
		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		i32 m = dim - j - 1;

    // avoid buffer overflow UB when accessing the matrices
		i32 j_inc = ((j + 1) < dim) ? j + 1 : j;

		auto l01 = detail::VecMapMut<Scalar>{std::addressof(out_l(0, j)), j};
		l01.setZero();
		out_l(j, j) = Scalar(1);

		auto l10 = detail::VecMapStrided<Scalar>{
				std::addressof(out_l(j, 0)),
				j,
				1,
				Eigen::InnerStride<Eigen::Dynamic>(dim),
		};

		auto l20 = detail::MatMap<Scalar>{
				std::addressof(out_l(j_inc, 0)),
				m,
				j,
				Eigen::OuterStride<Eigen::Dynamic>{dim},
		};
		auto l21 = detail::VecMapMut<Scalar>{std::addressof(out_l(j_inc, j)), m};
		auto a21 = detail::VecMap<Scalar>{std::addressof(in_matrix(j_inc, j)), m};

		auto d = detail::VecMap<Scalar>{out_d.data, j};
		auto tmp_read = detail::VecMap<Scalar>{workspace.data(), j};
		auto tmp = detail::VecMapMut<Scalar>{workspace.data(), j};

		tmp.array() = l10.array().operator*(d.array());
		out_d(j) = in_matrix(j, j) - Scalar(tmp_read.dot(l10));
		l21 = a21;
		l21.noalias().operator-=(l20.operator*(tmp_read));
		l21 = l21.operator/(out_d(j));
	}
}
} // namespace ldlt

#endif /* end of include guard LDLT_LDLT_HPP_FDFNWYGES */
