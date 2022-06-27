/** \file */
//
// Copyright (c) 2022, INRIA
//
#ifndef DENSE_LDLT_SOLVE_HPP_PXM02T7QS
#define DENSE_LDLT_SOLVE_HPP_PXM02T7QS

#include "linearsolver/dense/core.hpp"
#include <Eigen/Core>

namespace linearsolver {
namespace dense {
namespace _detail {
template <typename Mat, typename Rhs>
void solve_impl(Mat ld, Rhs rhs) {
	auto l = ld.template triangularView<Eigen::UnitLower>();
	auto lt = util::trans(ld).template triangularView<Eigen::UnitUpper>();
	auto d = util::diagonal(ld);

	l.solveInPlace(rhs);
	rhs = rhs.cwiseQuotient(d);
	lt.solveInPlace(rhs);
}
} // namespace _detail
template <typename Mat, typename Rhs>
void solve(Mat const& mat, Rhs&& rhs) {
	_detail::solve_impl(util::to_view(mat), util::to_view_dyn_rows(rhs));
}
} // namespace dense
} // namespace linearsolver

#endif /* end of include guard DENSE_LDLT_SOLVE_HPP_PXM02T7QS */
