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

	isize n = ldl.d().dim;
	bool inplace = x.data == b.data;

	auto x_e = x.to_eigen();
	auto d_e = ldl.d().to_eigen();
	auto l_e = ldl.l().to_eigen();
	auto lt_e = ldl.l().trans().to_eigen();
	auto l_lower = l_e.template triangularView<Eigen::UnitLower>();
	auto lt_upper = lt_e.template triangularView<Eigen::UnitUpper>();

	// x = b
	if (!inplace) {
		x_e = detail::VecMap<T>{b.data, n};
	}
	l_lower.solveInPlace(x_e);
	x_e = x_e.cwiseQuotient(d_e);
	lt_upper.solveInPlace(x_e);
}
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
