//
// Copyright (c) 2022 INRIA
//
/**
 * @file eigen.hpp
 */

#ifndef PROXSUITE_SERIALIZATION_EIGEN_HPP
#define PROXSUITE_SERIALIZATION_EIGEN_HPP

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <cereal/cereal.hpp>

namespace cereal {

// dense matrices
template<class Archive,
         class _Scalar,
         int _Rows,
         int _Cols,
         int _Options,
         int _MaxRows,
         int _MaxCols>
inline void
save(
  Archive& ar,
  Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const& m)
{
  Eigen::Index rows = m.rows();
  Eigen::Index cols = m.cols();
  ar(rows);
  ar(cols);
  ar(_Options);

  for (Eigen::Index i = 0; i < m.size(); i++)
    ar(m.data()[i]);
}

template<class Archive,
         class _Scalar,
         int _Rows,
         int _Cols,
         int _Options,
         int _MaxRows,
         int _MaxCols>
inline void
load(Archive& ar,
     Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m)
{
  Eigen::Index rows;
  Eigen::Index cols;
  int _OptionsLoaded;
  ar(rows);
  ar(cols);
  ar(_OptionsLoaded);

  m.resize(rows, cols);

  for (Eigen::Index i = 0; i < m.size(); i++)
    ar(m.data()[i]);

  // Account for different storage orders
  if (!_OptionsLoaded == _Options) {
    m.transposeInPlace();
  }
}

// sparse matrices
template<class Archive, typename _Scalar, int _Options, typename _StorageIndex>
inline void
save(Archive& ar,
     Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> const& m)
{
  Eigen::Index innerSize = m.innerSize();
  Eigen::Index outerSize = m.outerSize();
  typedef typename Eigen::Triplet<_Scalar> Triplet;
  std::vector<Triplet> triplets;

  for (Eigen::Index i = 0; i < outerSize; ++i) {
    for (typename Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex>::
           InnerIterator it(m, i);
         it;
         ++it) {
      triplets.push_back(Triplet(it.row(), it.col(), it.value()));
    }
  }
  ar(innerSize);
  ar(outerSize);
  ar(triplets);
}

template<class Archive, typename _Scalar, int _Options, typename _StorageIndex>
inline void
load(Archive& ar, Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex>& m)
{
  Eigen::Index innerSize;
  Eigen::Index outerSize;
  ar(innerSize);
  ar(outerSize);
  Eigen::Index rows = m.IsRowMajor ? outerSize : innerSize;
  Eigen::Index cols = m.IsRowMajor ? innerSize : outerSize;
  m.resize(rows, cols);
  typedef typename Eigen::Triplet<_Scalar> Triplet;
  std::vector<Triplet> triplets;
  ar(triplets);
  m.setFromTriplets(triplets.begin(), triplets.end());
}

} // namespace cereal

#endif /* end of include guard PROXSUITE_SERIALIZATION_EIGEN_HPP */
