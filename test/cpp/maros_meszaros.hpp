#include <Eigen/SparseCore>
#include <matio.h>
#include <string>
#include <proxsuite/linalg/veg/util/assert.hpp>
#include <iostream>

struct MarosMeszarosQp
{
  using Mat = Eigen::SparseMatrix<double, Eigen::ColMajor, mat_int32_t>;
  using Vec = Eigen::VectorXd;

  std::string filename;

  Mat P;
  Vec q;
  Mat A;
  Vec l;
  Vec u;

  bool skip = false;
};

auto
load_qp(char const* filename, bool check_size = false) -> MarosMeszarosQp
{
  using Mat = MarosMeszarosQp::Mat;
  using Vec = MarosMeszarosQp::Vec;

  mat_t* mat_fp = Mat_Open(filename, MAT_ACC_RDONLY);
  VEG_ASSERT(mat_fp != nullptr);
  auto&& _mat_fp_cleanup =
    proxsuite::linalg::veg::defer([&] { Mat_Close(mat_fp); });
  proxsuite::linalg::veg::unused(_mat_fp_cleanup);

  bool skip = false;

  auto load_mat = [&](char const* name) -> Mat {
    matvar_t* mat_var = Mat_VarRead(mat_fp, name);
    VEG_ASSERT(mat_var != nullptr);
    auto&& _mat_var_cleanup =
      proxsuite::linalg::veg::defer([&] { Mat_VarFree(mat_var); });
    proxsuite::linalg::veg::unused(_mat_var_cleanup);

    VEG_ASSERT(int(mat_var->class_type) == int(matio_classes::MAT_C_SPARSE));
    auto const* ptr = static_cast<mat_sparse_t const*>(mat_var->data);

    using proxsuite::linalg::veg::isize;

    isize nrows = isize(mat_var->dims[0]);
    isize ncols = isize(mat_var->dims[1]);
    Mat out;

    if (!check_size || (nrows <= 1000 && ncols <= 1000)) {
      auto optr = reinterpret_cast<mat_int32_t const*>(ptr->jc); // NOLINT
      auto iptr = reinterpret_cast<mat_int32_t const*>(ptr->ir); // NOLINT
      auto vptr = static_cast<double const*>(ptr->data);         // NOLINT

      out.resize(nrows, ncols);
      out.reserve(ptr->nzmax);
      for (isize j = 0; j < ncols; ++j) {
        isize col_start = optr[j];
        isize col_end = optr[j + 1];

        for (isize p = col_start; p < col_end; ++p) {

          isize i = iptr[p];
          double v = vptr[p];

          out.insert(i, j) = v;
        }
      }
    } else {
      skip = true;
    }
    return out;
  };

  auto load_vec = [&](char const* name) -> Vec {
    matvar_t* mat_var = Mat_VarRead(mat_fp, name);
    VEG_ASSERT(mat_var != nullptr);
    auto&& _mat_var_cleanup =
      proxsuite::linalg::veg::defer([&] { Mat_VarFree(mat_var); });
    proxsuite::linalg::veg::unused(_mat_var_cleanup);

    VEG_ASSERT(int(mat_var->data_type) == int(matio_types::MAT_T_DOUBLE));
    auto const* ptr = static_cast<double const*>(mat_var->data);

    auto view = Eigen::Map<Vec const>{
      ptr,
      long(mat_var->dims[0]),
    };
    return view;
  };

  return {
    filename,      load_mat("P"), load_vec("q"), load_mat("A"),
    load_vec("l"), load_vec("u"), skip,
  };
}

struct PreprocessedQp
{
  using Mat = MarosMeszarosQp::Mat::DenseMatrixType;
  using Vec = MarosMeszarosQp::Vec;

  Mat H;
  Mat A;
  Mat C;
  Vec g;
  Vec b;
  Vec u;
  Vec l;
};

struct PreprocessedQpSparse
{
  using Mat = MarosMeszarosQp::Mat;
  using Vec = MarosMeszarosQp::Vec;

  Mat H;
  Mat AT;
  Mat CT;
  Vec g;
  Vec b;
  Vec u;
  Vec l;
};

auto
preprocess_qp(MarosMeszarosQp& qp) -> PreprocessedQp
{
  using Mat = MarosMeszarosQp::Mat;
  using Vec = MarosMeszarosQp::Vec;
  using proxsuite::linalg::veg::isize;

  auto eq = qp.l.array().cwiseEqual(qp.u.array()).eval();

  isize n = qp.P.rows();
  isize n_eq = eq.count();
  isize n_in = eq.rows() - n_eq;

  Mat::DenseMatrixType A{ n_eq, n };
  Vec b{ n_eq };

  Mat::DenseMatrixType C{ n_in, n };
  Vec u{ n_in };
  Vec l{ n_in };

  isize eq_idx = 0;
  isize in_idx = 0;
  for (isize i = 0; i < eq.rows(); ++i) {
    if (eq[i]) {
      A.row(eq_idx) = qp.A.row(i);
      b[eq_idx] = qp.l[i];
      ++eq_idx;
    } else {
      C.row(in_idx) = qp.A.row(i);
      l[in_idx] = qp.l[i];
      u[in_idx] = qp.u[i];
      ++in_idx;
    }
  }

  return {
    qp.P.toDense(), VEG_FWD(A), VEG_FWD(C), VEG_FWD(qp.q),
    VEG_FWD(b),     VEG_FWD(u), VEG_FWD(l),
  };
}

auto
preprocess_qp_sparse(MarosMeszarosQp&& qp) -> PreprocessedQpSparse
{
  using Mat = MarosMeszarosQp::Mat;
  using Vec = MarosMeszarosQp::Vec;
  using proxsuite::linalg::veg::isize;

  auto eq = qp.l.array().cwiseEqual(qp.u.array()).eval();

  isize n = qp.P.rows();
  isize n_eq = eq.count();
  isize n_in = eq.rows() - n_eq;

  qp.A = Mat(qp.A.transpose());

  Mat AT{ n, n_eq };
  Vec b{ n_eq };

  Mat CT{ n, n_in };
  Vec u{ n_in };
  Vec l{ n_in };

  isize eq_idx = 0;
  isize in_idx = 0;
  for (isize i = 0; i < n_eq + n_in; ++i) {
    if (eq[i]) {
      AT.col(eq_idx) = qp.A.col(i);
      b[eq_idx] = qp.l[i];
      ++eq_idx;
    } else {
      CT.col(in_idx) = qp.A.col(i);
      l[in_idx] = qp.l[i];
      u[in_idx] = qp.u[i];
      ++in_idx;
    }
  }
  AT.makeCompressed();
  CT.makeCompressed();

  return {
    qp.P.triangularView<Eigen::Upper>(),
    VEG_FWD(AT),
    VEG_FWD(CT),
    VEG_FWD(qp.q),
    VEG_FWD(b),
    VEG_FWD(u),
    VEG_FWD(l),
  };
}
