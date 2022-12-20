//
// Copyright (c) 2022 INRIA
//
/**
 * @file serialize.hpp
 */

#ifndef PROXSUITE_SPARSE_SERIALIZE_HPP
#define PROXSUITE_SPARSE_SERIALIZE_HPP

#include <Eigen/Sparse>
#include <cereal/cereal.hpp>

namespace cereal {
template<class Archive,
         int _Options,
         typename _StorageIndex inline void save(
           Archive& ar,
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

template<class Archive,
         int _Options,
         typename _StorageIndex inline void load(
           Archive& ar,
           Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex>& m)
{
  Eigen::Index rows;
  Eigen::Index cols;
  ar(rows);
  ar(cols);

  m.resize(rows, cols);

  for (Eigen::Index i = 0; i < m.size(); i++)
    ar(m.data()[i]);

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

#endif /* end of include guard PROXSUITE_SPARSE_SERIALIZE_HPP */
