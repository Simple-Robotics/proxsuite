#pragma once

#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

namespace proxsuite {
namespace proxqp {
namespace utils {

namespace eigen {
LDLT_EXPLICIT_TPL_DECL(2, llt_compute<Mat<f32, colmajor>>);
LDLT_EXPLICIT_TPL_DECL(2, ldlt_compute<Mat<f32, colmajor>>);
LDLT_EXPLICIT_TPL_DECL(2, llt_compute<Mat<f32, rowmajor>>);
LDLT_EXPLICIT_TPL_DECL(2, ldlt_compute<Mat<f32, rowmajor>>);
} // namespace eigen
namespace rand {
LDLT_EXPLICIT_TPL_DECL(2, matrix_rand<f32>);
LDLT_EXPLICIT_TPL_DECL(1, vector_rand<f32>);
LDLT_EXPLICIT_TPL_DECL(2, positive_definite_rand<f32>);
LDLT_EXPLICIT_TPL_DECL(1, orthonormal_rand<f32>);
LDLT_EXPLICIT_TPL_DECL(3, sparse_matrix_rand<f32>);
LDLT_EXPLICIT_TPL_DECL(3, sparse_positive_definite_rand<f32>);
} // namespace rand

LDLT_EXPLICIT_TPL_DECL(2, matmul_impl<long double>);
LDLT_EXPLICIT_TPL_DECL(1, mat_cast<proxqp::f32, long double>);

} // namespace utils
} // namespace proxqp
} // namespace proxsuite
