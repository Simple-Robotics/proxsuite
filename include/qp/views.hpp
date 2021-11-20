#ifndef INRIA_LDLT_VIEWS_HPP_COA7NGHVS
#define INRIA_LDLT_VIEWS_HPP_COA7NGHVS

#include "ldlt/views.hpp"

namespace qp {
	
namespace detail {
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
	MatrixView<T, colmajor> H;
	VectorView<T> g;

	MatrixView<T, colmajor> A;
	VectorView<T> b;
	MatrixView<T, colmajor> C;
	VectorView<T> d;
};
template <typename Scalar>
struct QpViewBox {
	MatrixView<Scalar, colmajor> H;
	VectorView<Scalar> g;

	MatrixView<Scalar, colmajor> A;
	VectorView<Scalar> b;
	MatrixView<Scalar, colmajor> C;
	VectorView<Scalar> u;
	VectorView<Scalar> l;
};

template <typename T>
struct QpViewMut {
	MatrixViewMut<T, colmajor> H;
	VectorViewMut<T> g;

	MatrixViewMut<T, colmajor> A;
	VectorViewMut<T> b;
	MatrixViewMut<T, colmajor> C;
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
	MatrixViewMut<Scalar, colmajor> H;
	VectorViewMut<Scalar> g;

	MatrixViewMut<Scalar, colmajor> A;
	VectorViewMut<Scalar> b;
	MatrixViewMut<Scalar, colmajor> C;
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
