#ifndef INRIA_LDLT_VIEWS_HPP_COA7NGHVS
#define INRIA_LDLT_VIEWS_HPP_COA7NGHVS

#include "ldlt/views.hpp"

namespace qp {

namespace detail {

struct EigenAllowAlloc {
	bool alloc_was_allowed;
	EigenAllowAlloc(EigenAllowAlloc&&) = delete;
	EigenAllowAlloc(EigenAllowAlloc const&) = delete;
	auto operator=(EigenAllowAlloc&&) -> EigenAllowAlloc& = delete;
	auto operator=(EigenAllowAlloc const&) -> EigenAllowAlloc& = delete;

#if defined(EIGEN_RUNTIME_NO_MALLOC)
	EigenAllowAlloc() noexcept
			: alloc_was_allowed(Eigen::internal::is_malloc_allowed()) {
		Eigen::internal::set_is_malloc_allowed(true);
	}
	~EigenAllowAlloc() noexcept {
		Eigen::internal::set_is_malloc_allowed(alloc_was_allowed);
	}
#else
	EigenAllowAlloc() = default;
#endif
};

using namespace ldlt::detail;
} // namespace detail
using ldlt::Layout;
using ldlt::i32;
using ldlt::i64;
using ldlt::isize;
using ldlt::usize;
using ldlt::MatrixViewMut;
using ldlt::MatrixView;
using ldlt::VectorViewMut;
using ldlt::VectorView;
using ldlt::LdltViewMut;
using ldlt::LdltView;
using ldlt::colmajor;
using ldlt::rowmajor;

template <typename T>
struct QpView {

	static constexpr Layout layout = rowmajor;

	MatrixView<T, layout> H;
	VectorView<T> g;

	MatrixView<T, layout> A;
	VectorView<T> b;
	MatrixView<T, layout> C;
	VectorView<T> d;
};

template <typename Scalar>
struct QpViewBox {
	static constexpr Layout layout = rowmajor;

	MatrixView<Scalar, layout> H;
	VectorView<Scalar> g;

	MatrixView<Scalar, layout> A;
	VectorView<Scalar> b;
	MatrixView<Scalar, layout> C;
	VectorView<Scalar> u;
	VectorView<Scalar> l;
};

template <typename T>
struct QpViewMut {
	static constexpr Layout layout = rowmajor;

	MatrixViewMut<T, layout> H;
	VectorViewMut<T> g;

	MatrixViewMut<T, layout> A;
	VectorViewMut<T> b;
	MatrixViewMut<T, layout> C;
	VectorViewMut<T> d;

	LDLT_INLINE constexpr auto as_const() const noexcept -> QpView<T> {
		return {
				H.as_const(),
				g.as_const(),
				A.as_const(),
				b.as_const(),
				C.as_const(),
				d.as_const(),
		};
	}
};

template <typename Scalar>
struct QpViewBoxMut {
	static constexpr Layout layout = rowmajor;

	MatrixViewMut<Scalar, layout> H;
	VectorViewMut<Scalar> g;

	MatrixViewMut<Scalar, layout> A;
	VectorViewMut<Scalar> b;
	MatrixViewMut<Scalar, layout> C;
	VectorViewMut<Scalar> u;
	VectorViewMut<Scalar> l;

	LDLT_INLINE constexpr auto as_const() const noexcept -> QpViewBox<Scalar> {
		return {
				H.as_const(),
				g.as_const(),
				A.as_const(),
				b.as_const(),
				C.as_const(),
				u.as_const(),
				l.as_const(),
		};
	}
};

namespace nb {
struct pow {
	template <typename T>
	auto operator()(T x, T y) const -> T {
		using std::pow;
		return pow(x, y);
	}
};
struct infty_norm {
	template <typename D>
	auto operator()(Eigen::MatrixBase<D> const& mat) const -> typename D::Scalar {
		if (mat.rows() == 0 || mat.cols() == 0) {
			return typename D::Scalar(0);
		} else {
			return mat.template lpNorm<Eigen::Infinity>();
		}
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(pow);
LDLT_DEFINE_NIEBLOID(infty_norm);
} // namespace qp

#endif /* end of include guard INRIA_LDLT_VIEWS_HPP_COA7NGHVS */
