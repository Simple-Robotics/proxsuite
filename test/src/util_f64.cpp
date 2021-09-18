#include <util.hpp>

namespace ldlt_test {
namespace eigen {
LDLT_EXPLICIT_TPL_DEF(2, llt_compute<colmajor, f64>);
LDLT_EXPLICIT_TPL_DEF(2, ldlt_compute<colmajor, f64>);
LDLT_EXPLICIT_TPL_DEF(2, llt_compute<rowmajor, f64>);
LDLT_EXPLICIT_TPL_DEF(2, ldlt_compute<rowmajor, f64>);
} // namespace eigen
namespace rand {
LDLT_EXPLICIT_TPL_DEF(2, matrix_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(1, vector_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(2, positive_definite_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(1, orthonormal_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(3, sparse_matrix_rand<f64>);
LDLT_EXPLICIT_TPL_DEF(3, sparse_positive_definite_rand<f64>);
} // namespace rand
} // namespace ldlt_test

LDLT_EXPLICIT_TPL_DEF(1, mat_cast<ldlt::f64, long double>);
