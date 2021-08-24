#ifndef INRIA_LDLT_VIEWS_HPP_COA7NGHVS
#define INRIA_LDLT_VIEWS_HPP_COA7NGHVS

#include "ldlt/views.hpp"

namespace qp {
namespace detail {
using namespace ldlt::detail;
} // namespace detail

using ldlt::Layout;
using ldlt::i32;
using ldlt::MatrixViewMut;
using ldlt::MatrixView;
using ldlt::VectorViewMut;
using ldlt::VectorView;
using ldlt::LdltViewMut;
using ldlt::LdltView;
using ldlt::colmajor;
using ldlt::rowmajor;

template <typename Scalar, Layout LH, Layout LC>
struct QpView {
	MatrixView<Scalar, LH> H;
	VectorView<Scalar> g;

	MatrixView<Scalar, LC> A;
	VectorView<Scalar> b;
	MatrixView<Scalar, LC> C;
	VectorView<Scalar> d;
};

template <typename Scalar, Layout LH, Layout LC>
struct QpViewMut {
	MatrixViewMut<Scalar, LH> H;
	VectorViewMut<Scalar> g;

	MatrixViewMut<Scalar, LC> A;
	VectorViewMut<Scalar> b;
	MatrixViewMut<Scalar, LC> C;
	VectorViewMut<Scalar> d;

	LDLT_INLINE constexpr auto as_const() const noexcept
			-> QpView<Scalar, LH, LC> {
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
} // namespace qp

#endif /* end of include guard INRIA_LDLT_VIEWS_HPP_COA7NGHVS */
