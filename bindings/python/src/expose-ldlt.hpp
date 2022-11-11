//
// Copyright (c) 2022 INRIA
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <proxsuite/proxqp/sparse/wrapper.hpp>

namespace proxsuite {
namespace linalg {
namespace sparse {
namespace python {

template<typename T, typename I>
auto
to_eigen(proxsuite::linalg::sparse::MatRef<T, I> a) noexcept
  -> Eigen::Matrix<T, -1, -1>
{
  return a.to_eigen();
}
template<typename I>
auto
to_eigen_perm(proxsuite::linalg::veg::Slice<I> perm)
  -> Eigen::PermutationMatrix<-1, -1, I>
{
  Eigen::PermutationMatrix<-1, -1, I> perm_eigen;
  perm_eigen.indices().resize(perm.len());
  std::memmove( //
    perm_eigen.indices().data(),
    perm.ptr(),
    proxsuite::linalg::veg::usize(perm.len()) * sizeof(I));
  return perm_eigen;
}

template<typename T, typename I>
auto
reconstruct_with_perm(proxsuite::linalg::veg::Slice<I> perm_inv,
                      proxsuite::linalg::sparse::MatRef<T, I> ld)
  -> Eigen::Matrix<T, -1, -1, Eigen::ColMajor>
{
  using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
  Mat ld_eigen = to_eigen(ld);
  auto perm_inv_eigen = to_eigen_perm(perm_inv);
  Mat l = ld_eigen.template triangularView<Eigen::UnitLower>();
  Mat d = ld_eigen.diagonal().asDiagonal();
  Mat ldlt = l * d * l.transpose();
  return perm_inv_eigen.inverse() * ldlt * perm_inv_eigen;
}

template<typename T, typename I>
struct SparseLDLT
{

  ///// Cholesky Factorization
  proxsuite::proxqp::sparse::Ldlt<T, I> ldl;
  proxsuite::linalg::veg::Vec<proxsuite::linalg::veg::mem::byte> storage;
  isize n_tot;
  isize nnz_tot;

  proxsuite::linalg::veg::Vec<I> kkt_col_ptrs;
  proxsuite::linalg::veg::Vec<I> kkt_row_indices;
  proxsuite::linalg::veg::Vec<T> kkt_values;
  proxsuite::linalg::veg::Vec<I> kkt_nnz_counts;

  proxsuite::linalg::veg::Tag<I> itag;
  proxsuite::linalg::veg::Tag<T> xtag;

  SparseLDLT(isize n_tot_)
    : ldl{}
    , n_tot(n_tot_)
  {
    PROXSUITE_THROW_PRETTY(n_tot == 0,
                           std::invalid_argument,
                           "wrong argument size: the dimension of the "
                           "matrix to factorize should be strictly positive.");
  }
  void factorize(proxsuite::proxqp::sparse::SparseMat<T, I>& mat_)
  {

    proxsuite::proxqp::sparse::SparseMat<T, I> Mat =
      mat_.template triangularView<Eigen::Upper>();
    proxsuite::linalg::sparse::MatRef<T, I> mat = {
      proxsuite::linalg::sparse::from_eigen, Mat
    };
    using SR = proxsuite::linalg::veg::dynstack::StackReq;
    using namespace proxsuite::linalg::veg::dynstack;
    using namespace proxsuite::linalg::sparse::util;
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      mat.nrows(),
      n_tot,
      "the dimension of the matrix to factorize is different from "
      "the one given to initialize the Sparse LDLT object.");
    nnz_tot = mat.nnz();
    isize lnnz = 0;
    {

      kkt_col_ptrs.resize_for_overwrite(n_tot + 1);
      kkt_row_indices.resize_for_overwrite(nnz_tot);
      kkt_values.resize_for_overwrite(nnz_tot);

      I* kktp = kkt_col_ptrs.ptr_mut();
      I* kkti = kkt_row_indices.ptr_mut();
      T* kktx = kkt_values.ptr_mut();

      kktp[0] = 0;
      usize col = 0;
      usize pos = 0;

      auto insert_submatrix = [&](proxsuite::linalg::sparse::MatRef<T, I> m,
                                  bool assert_sym_hi) -> void {
        I const* mi = m.row_indices();
        T const* mx = m.values();
        isize ncols = m.ncols();

        for (usize j = 0; j < usize(ncols); ++j) {
          usize col_start = m.col_start(j);
          usize col_end = m.col_end(j);

          kktp[col + 1] =
            checked_non_negative_plus(kktp[col], I(col_end - col_start));
          ++col;

          for (usize p = col_start; p < col_end; ++p) {
            usize i = zero_extend(mi[p]);
            if (assert_sym_hi) {
              VEG_ASSERT(i <= j);
            }

            kkti[pos] = proxsuite::linalg::veg::nb::narrow<I>{}(i);
            kktx[pos] = mx[p];

            ++pos;
          }
        }
      };
      insert_submatrix(mat, true);
      storage.resize_for_overwrite( //
        (StackReq::with_len(itag, n_tot) &
         proxsuite::linalg::sparse::factorize_symbolic_req( //
           itag,                                            //
           n_tot,                                           //
           nnz_tot,                                         //
           proxsuite::linalg::sparse::Ordering::amd))       //
          .alloc_req()                                      //
      );

      ldl.col_ptrs.resize_for_overwrite(n_tot + 1);
      ldl.perm_inv.resize_for_overwrite(n_tot);

      proxsuite::linalg::veg::dynstack::DynStackMut stack =
        proxsuite::linalg::veg::dynstack::DynStackMut{
          proxsuite::linalg::veg::tags::from_slice_mut, storage.as_mut()
        };

      ldl.etree.resize_for_overwrite(n_tot);
      auto etree_ptr = ldl.etree.ptr_mut();

      using namespace proxsuite::linalg::veg::literals;
      auto kkt_sym = proxsuite::linalg::sparse::SymbolicMatRef<I>{
        proxsuite::linalg::sparse::from_raw_parts,
        n_tot,
        n_tot,
        nnz_tot,
        kkt_col_ptrs.ptr(),
        nullptr,
        kkt_row_indices.ptr(),
      };

      proxsuite::linalg::sparse::factorize_symbolic_non_zeros( //
        ldl.col_ptrs.ptr_mut() + 1,
        etree_ptr,
        ldl.perm_inv.ptr_mut(),
        static_cast<I const*>(nullptr),
        kkt_sym,
        stack);
      auto pcol_ptrs = ldl.col_ptrs.ptr_mut();
      pcol_ptrs[0] = I(0); // pcol_ptrs +1: pointor towards the nbr of non zero
                           // elts per column of the ldlt
      // we need to compute its cumulative sum below to determine if there could
      // be an overflow

      using proxsuite::linalg::veg::u64;
      u64 acc = 0;

      for (usize i = 0; i < usize(n_tot); ++i) {
        acc += u64(zero_extend(pcol_ptrs[i + 1]));
        // if (acc != u64(I(acc))) {
        //	overflow = true;
        // }
        pcol_ptrs[(i + 1)] = I(acc);
      }

      lnnz = isize(zero_extend(ldl.col_ptrs[n_tot]));
    }
#define PROX_QP_ALL_OF(...)                                                    \
  ::proxsuite::linalg::veg::dynstack::StackReq::and_(                          \
    ::proxsuite::linalg::veg::init_list(__VA_ARGS__))
#define PROX_QP_ANY_OF(...)                                                    \
  ::proxsuite::linalg::veg::dynstack::StackReq::or_(                           \
    ::proxsuite::linalg::veg::init_list(__VA_ARGS__))
    auto refactorize_req = PROX_QP_ANY_OF({
      proxsuite::linalg::sparse::factorize_symbolic_req( // symbolic ldl
        itag,
        n_tot,
        nnz_tot,
        proxsuite::linalg::sparse::Ordering::user_provided),
      PROX_QP_ALL_OF({
        SR::with_len(xtag, n_tot),                        // diag
        proxsuite::linalg::sparse::factorize_numeric_req( // numeric ldl
          xtag,
          itag,
          n_tot,
          nnz_tot,
          proxsuite::linalg::sparse::Ordering::user_provided),
      }),
    });

    auto x_vec = [&](isize n) noexcept -> StackReq {
      return proxsuite::linalg::dense::temp_vec_req(xtag, n);
    };

    auto ldl_solve_in_place_req = PROX_QP_ALL_OF({
      x_vec(n_tot), // tmp
      x_vec(n_tot), // err
      x_vec(n_tot), // work
    });
    auto req =                                    //
      PROX_QP_ALL_OF({ SR::with_len(itag, n_tot), // kkt nnz counts
                       refactorize_req,
                       ldl_solve_in_place_req,
                       PROX_QP_ALL_OF({
                         SR::with_len(itag, n_tot), // perm
                         SR::with_len(itag, n_tot), // etree
                         SR::with_len(itag, n_tot), // ldl nnz counts
                         SR::with_len(itag, lnnz),  // ldl row indices
                         SR::with_len(xtag, lnnz),  // ldl values
                       }) });
    storage.resize_for_overwrite(req.alloc_req());
    proxsuite::linalg::veg::dynstack::DynStackMut stack =
      proxsuite::linalg::veg::dynstack::DynStackMut{
        proxsuite::linalg::veg::tags::from_slice_mut, storage.as_mut()
      };
    kkt_nnz_counts.resize_for_overwrite(n_tot);
    auto zx = proxsuite::linalg::sparse::util::zero_extend; // ?
    auto max_lnnz = isize(zx(ldl.col_ptrs[n_tot]));
    isize ldlt_ntot = n_tot;
    isize ldlt_lnnz = max_lnnz;
    ldl.nnz_counts.resize_for_overwrite(ldlt_ntot);
    ldl.row_indices.resize_for_overwrite(ldlt_lnnz);
    ldl.values.resize_for_overwrite(ldlt_lnnz);
    ldl.perm.resize_for_overwrite(ldlt_ntot);
    if (true) {
      // compute perm from perm_inv
      for (isize i = 0; i < n_tot; ++i) {
        ldl.perm[isize(zx(ldl.perm_inv[i]))] = I(i);
      }
    }
    auto nnz =
      isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
    proxsuite::linalg::sparse::MatMut<T, I> kkt = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz,
      kkt_col_ptrs.ptr_mut(),
      nullptr,
      kkt_row_indices.ptr_mut(),
      kkt_values.ptr_mut(),
    };

    I* kkt_nnz_counts_ = kkt_nnz_counts.ptr_mut();
    for (usize j = 0; j < usize(n_tot); ++j) {
      kkt_nnz_counts_[isize(j)] = I(kkt.col_end(j) - kkt.col_start(j));
    }
    proxsuite::linalg::sparse::MatMut<T, I> kkt_active = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz_tot,
      kkt.col_ptrs_mut(),
      kkt_nnz_counts_,
      kkt.row_indices_mut(),
      kkt.values_mut(),
    };

    proxsuite::linalg::sparse::factorize_symbolic_non_zeros(
      ldl.nnz_counts.ptr_mut(),
      ldl.etree.ptr_mut(),
      ldl.perm_inv.ptr_mut(),
      ldl.perm.ptr_mut(),
      kkt_active.symbolic(),
      stack);

    proxsuite::linalg::sparse::factorize_numeric(ldl.values.ptr_mut(),
                                                 ldl.row_indices.ptr_mut(),
                                                 nullptr, // diag
                                                 ldl.perm.ptr_mut(),
                                                 ldl.col_ptrs.ptr(),
                                                 ldl.etree.ptr_mut(),
                                                 ldl.perm_inv.ptr_mut(),
                                                 kkt_active.as_const(),
                                                 stack);
  }
  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> reconstruct_factorized_matrix()
  {

    I* ldl_nnz_counts = ldl.nnz_counts.ptr_mut();
    I* ldl_row_indices = ldl.row_indices.ptr_mut();
    T* ldl_values = ldl.values.ptr_mut();
    auto ldl_col_ptrs = ldl.col_ptrs.ptr_mut();
    proxsuite::linalg::sparse::MatMut<T, I> ld = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      0,
      ldl_col_ptrs,
      ldl_nnz_counts,
      ldl_row_indices,
      ldl_values,
    };
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> a =
      reconstruct_with_perm(ldl.perm_inv.as_ref(), ld.as_const());
    return a;
  }
  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> l()
  {

    I* ldl_nnz_counts = ldl.nnz_counts.ptr_mut();
    I* ldl_row_indices = ldl.row_indices.ptr_mut();
    T* ldl_values = ldl.values.ptr_mut();
    auto ldl_col_ptrs = ldl.col_ptrs.ptr_mut();
    proxsuite::linalg::sparse::MatMut<T, I> ld = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      0,
      ldl_col_ptrs,
      ldl_nnz_counts,
      ldl_row_indices,
      ldl_values,
    };
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> ld_eigen =
      to_eigen(ld.as_const());
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> l =
      ld_eigen.template triangularView<Eigen::UnitLower>();
    return l;
  }
  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> lt()
  {

    I* ldl_nnz_counts = ldl.nnz_counts.ptr_mut();
    I* ldl_row_indices = ldl.row_indices.ptr_mut();
    T* ldl_values = ldl.values.ptr_mut();
    auto ldl_col_ptrs = ldl.col_ptrs.ptr_mut();
    proxsuite::linalg::sparse::MatMut<T, I> ld = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      0,
      ldl_col_ptrs,
      ldl_nnz_counts,
      ldl_row_indices,
      ldl_values,
    };
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> ld_eigen =
      to_eigen(ld.as_const());
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> l =
      ld_eigen.template triangularView<Eigen::UnitLower>();
    return l.transpose();
  }
  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> d()
  {

    I* ldl_nnz_counts = ldl.nnz_counts.ptr_mut();
    I* ldl_row_indices = ldl.row_indices.ptr_mut();
    T* ldl_values = ldl.values.ptr_mut();
    auto ldl_col_ptrs = ldl.col_ptrs.ptr_mut();
    proxsuite::linalg::sparse::MatMut<T, I> ld = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      0,
      ldl_col_ptrs,
      ldl_nnz_counts,
      ldl_row_indices,
      ldl_values,
    };
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> ld_eigen =
      to_eigen(ld.as_const());
    Eigen::Matrix<T, -1, -1, Eigen::ColMajor> d =
      ld_eigen.diagonal().asDiagonal();
    return d;
  }

  Eigen::ArrayXi p()
  {
    Eigen::PermutationMatrix<-1, -1, I> perm_inv_eigen =
      to_eigen_perm(ldl.perm_inv.as_ref());
    Eigen::ArrayXi p = perm_inv_eigen.indices().array();
    return p;
  }
  Eigen::ArrayXi pt()
  {
    Eigen::PermutationMatrix<-1, -1, I> perm_inv_eigen =
      to_eigen_perm(ldl.perm_inv.as_ref()).inverse();
    Eigen::ArrayXi pt = perm_inv_eigen.indices().array();
    return pt;
  }
  void solve_in_place(proxsuite::proxqp::sparse::VecRefMut<T> rhs_e)
  {
    proxsuite::linalg::veg::dynstack::DynStackMut stack{
      proxsuite::linalg::veg::from_slice_mut, storage.as_mut()
    };
    LDLT_TEMP_VEC_UNINIT(T, work_, n_tot, stack);
    auto zx = proxsuite::linalg::sparse::util::zero_extend;

    I* ldl_nnz_counts = ldl.nnz_counts.ptr_mut();
    I* ldl_row_indices = ldl.row_indices.ptr_mut();
    T* ldl_values = ldl.values.ptr_mut();
    I* perm = ldl.perm.ptr_mut();
    I* perm_inv = ldl.perm_inv.ptr_mut();
    auto ldl_col_ptrs = ldl.col_ptrs.ptr_mut();
    proxsuite::linalg::sparse::MatMut<T, I> ldlt = {
      // change name
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      0,
      ldl_col_ptrs,
      ldl_nnz_counts, // si do_ldlt est vrai do ldl_nnz_counts
      ldl_row_indices,
      ldl_values,
    };

    for (isize i = 0; i < n_tot; ++i) {
      work_[i] = rhs_e[isize(zx(perm[i]))];
    }

    proxsuite::linalg::sparse::dense_lsolve<T, I>( //
      { proxsuite::linalg::sparse::from_eigen, work_ },
      ldlt.as_const());

    for (isize i = 0; i < n_tot; ++i) {
      work_[i] /= ldl_values[isize(zx(ldl_col_ptrs[i]))];
    }

    proxsuite::linalg::sparse::dense_ltsolve<T, I>( //
      { proxsuite::linalg::sparse::from_eigen, work_ },
      ldlt.as_const());

    for (isize i = 0; i < n_tot; ++i) {
      rhs_e[i] = work_[isize(zx(perm_inv[i]))];
    }
  }
};

template<typename T, typename I>
void
exposeSparseLDLT(pybind11::module_ m)
{

  ::pybind11::class_<SparseLDLT<T, I>>(
    m, "SparseLDLT", pybind11::module_local())
    .def(::pybind11::init<proxsuite::linalg::veg::i64>(),
         pybind11::arg_v("n_tot", 0, "dimension of the matrix to factorize."),
         "Constructor for defining sparse LDLT object.") // constructor)
    .def("factorize",
         &SparseLDLT<T, I>::factorize,
         "Factorizes a sparse symmetric and invertible matrix.")
    .def("reconstruct_factorized_matrix",
         &SparseLDLT<T, I>::reconstruct_factorized_matrix,
         "Reconstructs the factorized matrix in dense format.")
    .def("solve_in_place",
         &SparseLDLT<T, I>::solve_in_place,
         "Solve in place a linear system using the sparse factorization.")
    .def("l",
         &SparseLDLT<T, I>::l,
         "Outputs the lower triangular part of the sparse LDLT factorization "
         "in dense format.")
    .def("lt",
         &SparseLDLT<T, I>::lt,
         "Outputs the transpose of the lower triangular part of the sparse "
         "LDLT factorization "
         "in dense format.")
    .def("d",
         &SparseLDLT<T, I>::d,
         "Outputs the diagonal part of the sparse LDLT factorization "
         "in dense format.")
    .def("p",
         &SparseLDLT<T, I>::p,
         "Outputs the permutation matrix of the LDLT factorization "
         "in vector format.")
    .def("pt",
         &SparseLDLT<T, I>::pt,
         "Outputs the inverse permutation matrix of the LDLT factorization "
         "in dense format.");
}

} // namespace python
} // namespace sparse

} // namespace linalg
} // namespace proxsuite
