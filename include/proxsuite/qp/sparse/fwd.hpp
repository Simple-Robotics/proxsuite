//
// Copyright (c) 2022, INRIA
//
/** \file */
#ifndef PROXSUITE_QP_SPARSE_FWD_HPP
#define PROXSUITE_QP_SPARSE_FWD_HPP

#include <Eigen/Sparse>
#include <proxsuite/veg/vec.hpp>
#include <proxsuite/qp/dense/views.hpp>

namespace proxsuite {
namespace qp {
namespace sparse {

using dense::infty_norm;
using veg::isize;
using veg::usize;
using veg::i64;

template <typename T>
using DMat = Eigen::Matrix<T, -1, -1>;

static constexpr auto DYN = Eigen::Dynamic;
enum { layout = Eigen::RowMajor };
template <typename T, typename I>
using SparseMat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
//using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor, I>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template <typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template <typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

template <typename T, typename I>
using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
//using Mat = Eigen::SparseMatrix<T, Eigen::RowMajor, I>;

} // namespace sparse
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_FWD_HPP */
