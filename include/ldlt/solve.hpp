#ifndef INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS
#define INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS

#include "ldlt/views.hpp"

namespace ldlt {
namespace detail {
template <typename T, Layout L>
void solve_impl( //
		VectorViewMut<T> x,
		MatrixView<T, L> l,
		VectorView<T> d,
		VectorView<T> b) {

	i32 dim = l.dim;
	bool inplace = x.data == b.data;

	auto x_e = detail::VecMapMut<T>{x.data, dim};
	auto d_e = detail::VecMap<T>{d.data, dim};
	auto l_e = EigenMatMap<T, L>{
			l.data,
			l.dim,
			l.dim,
			l.outer_stride,
	};
	auto lt_e = EigenMatMap<T, ldlt::flip_layout(L)>{
			l.data,
			l.dim,
			l.dim,
			l.outer_stride,
	};
	auto l_lower = l_e.template triangularView<Eigen::UnitLower>();
	auto lt_upper = lt_e.template triangularView<Eigen::UnitUpper>();

	// x = b
	if (!inplace) {
		x_e = detail::VecMap<T>{b.data, dim};
	}
	l_lower.solveInPlace(x_e);
	x_e.array().operator/=(d_e.array());
	lt_upper.solveInPlace(x_e);
}
} // namespace detail

namespace nb {
struct solve {
	template <typename T, Layout L>
	LDLT_INLINE void operator()( //
			VectorViewMut<T> x,
			MatrixView<T, L> l,
			VectorView<T> d,
			VectorView<T> b) const {
		detail::solve_impl(x, l, d, b);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(solve);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS */
