#include "util_f64.hpp"

namespace proxsuite {
namespace proxqp {
namespace utils {

namespace eigen {
LDLT_EXPLICIT_TPL_DEF(2, llt_compute<Mat<f64, colmajor>>);
LDLT_EXPLICIT_TPL_DEF(2, ldlt_compute<Mat<f64, colmajor>>);
LDLT_EXPLICIT_TPL_DEF(2, llt_compute<Mat<f64, rowmajor>>);
LDLT_EXPLICIT_TPL_DEF(2, ldlt_compute<Mat<f64, rowmajor>>);
} // namespace eigen

namespace rand {
LDLT_EXPLICIT_TPL_DEF(2, matrix_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(1, vector_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(2, positive_definite_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(1, orthonormal_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(3, sparse_matrix_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(3, sparse_positive_definite_rand<f64>);
} // namespace rand

LDLT_EXPLICIT_TPL_DEF(1, mat_cast<f64, long double>);

} // namespace utils
} // namespace proxqp
} // namespace proxsuite
