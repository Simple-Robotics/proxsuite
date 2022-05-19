#ifndef PROXSUITE_TEST_OSQP_HPP
#define PROXSUITE_TEST_OSQP_HPP

#include "util.hpp"
#include <osqp.h>

namespace ldlt_test {
namespace osqp {

inline auto from_eigen(SparseMat<c_float>& mat) -> csc {
	return {
			mat.nonZeros(),
			mat.rows(),
			mat.cols(),
			mat.outerIndexPtr(),
			mat.innerIndexPtr(),
			mat.valuePtr(),
			-1,
	};
}

inline auto solve_eq_osqp_sparse(
		VectorViewMut<c_float> x,
		VectorViewMut<c_float> y,
		SparseMat<c_float> const& H_eigen,
		SparseMat<c_float> const& A_eigen,
		VectorView<c_float> g,
		VectorView<c_float> b,
		i64 max_iter,
		c_float eps_abs,
		c_float eps_rel) -> i64 {

	isize dim = isize(H_eigen.rows());
	isize n_eq = isize(A_eigen.rows());

	auto H =
			osqp::from_eigen(const_cast /* NOLINT */<SparseMat<c_float>&>(H_eigen));
	auto A =
			osqp::from_eigen(const_cast /* NOLINT */<SparseMat<c_float>&>(A_eigen));

	auto osqp = OSQPData{};
	osqp.n = dim;
	osqp.m = n_eq;
	osqp.P = &H;
	osqp.A = &A;
	osqp.q = const_cast /* NOLINT */<double*>(g.data);
	osqp.l = const_cast /* NOLINT */<double*>(b.data);
	osqp.u = const_cast /* NOLINT */<double*>(b.data);

	OSQPSettings osqp_settings{};
	osqp_set_default_settings(&osqp_settings);
	osqp_settings.eps_rel = eps_rel;
	osqp_settings.eps_abs = eps_abs;
	osqp_settings.max_iter = max_iter;
	osqp_settings.warm_start = 0;
	osqp_settings.verbose = 0;
	OSQPWorkspace* osqp_work{};

	osqp_setup(&osqp_work, &osqp, &osqp_settings);
	osqp_solve(osqp_work);
	i64 n_iter = i64(osqp_work->info->iter);
	std::memcpy(x.data, osqp_work->solution->x, usize(dim) * sizeof(c_float));
	std::memcpy(y.data, osqp_work->solution->y, usize(n_eq) * sizeof(c_float));
	osqp_cleanup(osqp_work);
	return n_iter;
};

} // namespace osqp
} // namespace ldlt_test
#endif /* end of include guard PROXSUITE_TEST_OSQP_HPP */
