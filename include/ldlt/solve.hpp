#ifndef INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS
#define INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS

#include "ldlt/views.hpp"

namespace ldlt {
namespace detail {
template <typename T>
LDLT_NO_INLINE void solve_impl( //
		VectorViewMut<T> x,
		LdltView<T> ldl,
		VectorView<T> b) {

	isize dim = ldl.l.rows;
	bool inplace = x.data == b.data;

	auto x_e = detail::VecMapMut<T>{x.data, dim};
	auto d_e = detail::VecMap<T>{ldl.d.data, dim};
	auto l_e = ldl.l.to_eigen();
	auto lt_e = ldl.l.trans().to_eigen();
	auto l_lower = l_e.template triangularView<Eigen::UnitLower>();
	auto lt_upper = lt_e.template triangularView<Eigen::UnitUpper>();

	// x = b
	if (!inplace) {
		x_e = detail::VecMap<T>{b.data, dim};
	}
	l_lower.solveInPlace(x_e);
	x_e = x_e.cwiseQuotient(d_e);
	lt_upper.solveInPlace(x_e);
}
extern template void
		solve_impl(VectorViewMut<f32>, LdltView<f32>, VectorView<f32>);
extern template void
		solve_impl(VectorViewMut<f64>, LdltView<f64>, VectorView<f64>);
} // namespace detail

namespace nb {
struct solve {
	template <typename T>
	LDLT_INLINE void operator()( //
			VectorViewMut<T> x,
			LdltView<T> ldl,
			VectorView<T> b) const {
		detail::solve_impl(x, ldl, b);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(solve);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_SOLVE_LDLT_HPP_L1KNDQRUS */
