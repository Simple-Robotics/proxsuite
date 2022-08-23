#pragma once

#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

namespace proxsuite {
namespace proxqp {
namespace utils {

namespace eigen {
LDLT_EXPLICIT_TPL_DECL(2, llt_compute<Mat<f64, colmajor>>);
LDLT_EXPLICIT_TPL_DECL(2, ldlt_compute<Mat<f64, colmajor>>);
LDLT_EXPLICIT_TPL_DECL(2, llt_compute<Mat<f64, rowmajor>>);
LDLT_EXPLICIT_TPL_DECL(2, ldlt_compute<Mat<f64, rowmajor>>);
} // namespace eigen

namespace rand {
LDLT_EXPLICIT_TPL_DECL(2, matrix_rand<f64>);
LDLT_EXPLICIT_TPL_DECL(1, vector_rand<f64>);
LDLT_EXPLICIT_TPL_DECL(2, positive_definite_rand<f64>);
LDLT_EXPLICIT_TPL_DECL(1, orthonormal_rand<f64>);
LDLT_EXPLICIT_TPL_DECL(3, sparse_matrix_rand<f64>);
LDLT_EXPLICIT_TPL_DECL(3, sparse_positive_definite_rand<f64>);
} // namespace rand

LDLT_EXPLICIT_TPL_DECL(1, mat_cast<f64, long double>);

} // namespace utils
} // namespace proxqp
} // namespace proxsuite
