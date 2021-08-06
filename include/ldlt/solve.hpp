#ifndef INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS
#define INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS

#include "ldlt/views.hpp"

namespace ldlt {
namespace detail {
template <typename T, Layout L>
void solve_impl( //
		VectorViewMut<T> x,
		LdltView<T, L> ldlt,
		VectorView<T> b) {

	constexpr Layout LT = ::ldlt::flip_layout(L);

	i32 dim = ldlt.l.dim;
	bool inplace = x.data == b.data;

	auto x_e = detail::VecMapMut<T>{x.data, dim};
	auto d_e = detail::VecMap<T>{ldlt.d.data, dim};
	auto l_e = EigenMatMap<T, L>{ldlt.l.data, dim, dim, ldlt.l.outer_stride};
	auto lt_e = EigenMatMap<T, LT>{ldlt.l.data, dim, dim, ldlt.l.outer_stride};
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
			LdltView<T, L> ldlt,
			VectorView<T> b) const {
		detail::solve_impl(x, ldlt, b);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(solve);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS */
