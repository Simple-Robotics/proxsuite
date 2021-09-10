#ifndef INRIA_LDLT_QPALM_HPP_7H6RYKIYS
#define INRIA_LDLT_QPALM_HPP_7H6RYKIYS

#include "util.hpp"
#include <qpalm.h>

namespace ldlt_test {
namespace qpalm {
using osqp::to_sparse;
using osqp::to_sparse_sym;

inline auto from_eigen(SparseMat<c_float>& mat) -> ladel_sparse_matrix {
	ladel_sparse_matrix out{};
	out.nzmax = mat.nonZeros();
	out.nrow = mat.rows();
	out.ncol = mat.cols();
	out.p = mat.outerIndexPtr();
	out.i = mat.innerIndexPtr();
	out.x = mat.valuePtr();
	out.nz = nullptr;
	out.values = TRUE;
	out.symmetry = UNSYMMETRIC;
	return out;
}

inline auto from_eigen_sym(SparseMat<c_float>& mat) -> ladel_sparse_matrix {
	auto out = qpalm::from_eigen(mat);
	out.symmetry = UPPER;
	return out;
}

inline auto solve_eq_qpalm_sparse(
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

	auto H = qpalm::from_eigen_sym(
			const_cast /* NOLINT */<SparseMat<c_float>&>(H_eigen));
	auto A =
			qpalm::from_eigen(const_cast /* NOLINT */<SparseMat<c_float>&>(A_eigen));

	QPALMSettings settings{};
	qpalm_set_default_settings(&settings);
	settings.max_iter = max_iter;
	settings.eps_abs = eps_abs;
	settings.eps_rel = eps_rel;
	settings.warm_start = 0;
	settings.verbose = 0;

	QPALMData data{};
	data.n = usize(dim);
	data.m = usize(n_eq);
	data.Q = &H;
	data.A = &A;
	data.q = const_cast /* NOLINT */<c_float*>(g.data);
	data.bmin = const_cast /* NOLINT */<c_float*>(b.data);
	data.bmax = const_cast /* NOLINT */<c_float*>(b.data);
	data.c = 0;
	QPALMWorkspace* work = qpalm_setup(&data, &settings);
	qpalm_solve(work);
	auto n_iter = i64(work->info->iter);

	std::memcpy(x.data, work->solution->x, usize(dim) * sizeof(c_float));
	std::memcpy(y.data, work->solution->y, usize(n_eq) * sizeof(c_float));
	qpalm_cleanup(work);
	return n_iter;
};
} // namespace qpalm
} // namespace ldlt_test
#endif /* end of include guard INRIA_LDLT_QPALM_HPP_7H6RYKIYS */
