//
// Copyright (c) 2022 INRIA
//
/** \file */
#ifndef PROXSUITE_QP_DENSE_FWD_HPP
#define PROXSUITE_QP_DENSE_FWD_HPP

#include <Eigen/Sparse>

namespace proxsuite {
namespace proxqp {
namespace dense {

static constexpr auto DYN = Eigen::Dynamic;
enum
{
  layout = Eigen::RowMajor
};
template<typename T>
using SparseMat = Eigen::SparseMatrix<T, 1>;
template<typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template<typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template<typename T>
using Mat = Eigen::Matrix<T, DYN, DYN, layout>;
template<typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

using proxsuite::linalg::veg::isize;

template<typename T>
using VecMap = Eigen::Map<Vec<T> const>;
template<typename T>
using VecMapMut = Eigen::Map<Vec<T>>;

template<typename T>
using MatMap = Eigen::Map<Mat<T> const>;
template<typename T>
using MatMapMut = Eigen::Map<Mat<T>>;

using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
using VecISize = Eigen::Matrix<isize, DYN, 1>;

using VecMapBool = Eigen::Map<Eigen::Matrix<bool, DYN, 1> const>;
using VecBool = Eigen::Matrix<bool, DYN, 1>;

} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_FWD_HPP */
