#include <util.hpp>

namespace proxsuite {
namespace proxqp {
namespace test {

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

namespace osqp {
auto
to_sparse(Mat<c_float, colmajor> const& mat) -> SparseMat<c_float>
{
  SparseMat<c_float> out(mat.rows(), mat.cols());

  using Eigen::Index;
  for (Index j = 0; j < mat.cols(); ++j) {
    for (Index i = 0; i < mat.rows(); ++i) {
      if (mat(i, j) != 0) {
        out.insert(i, j) = mat(i, j);
      }
    }
  }
  out.makeCompressed();
  return out;
}

auto
to_sparse_sym(Mat<c_float, colmajor> const& mat) -> SparseMat<c_float>
{
  SparseMat<c_float> out(mat.rows(), mat.cols());
  using Eigen::Index;
  for (Index j = 0; j < mat.cols(); ++j) {
    for (Index i = 0; i < j + 1; ++i) {
      if (mat(i, j) != 0) {
        out.insert(i, j) = mat(i, j);
      }
    }
  }
  out.makeCompressed();
  return out;
}
} // namespace osqp

LDLT_EXPLICIT_TPL_DEF(1, mat_cast<f64, long double>);

} // namespace test
} // namespace proxqp
} // namespace proxsuite
