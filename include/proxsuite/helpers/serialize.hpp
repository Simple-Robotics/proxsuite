//
// Copyright (c) 2022 INRIA
//
/**
 * @file serialize.hpp
 */

#ifndef PROXSUITE_HELPERS_SERIALIZE_HPP
#define PROXSUITE_HELPERS_SERIALIZE_HPP

#ifdef PROXSUITE_WITH_SERIALIZATION
#include <Eigen/Eigen>
#include <cereal/cereal.hpp>

namespace cereal {
// only working for binary
// template<class Archive, class Derived>
// inline typename std::enable_if<
//   cereal::traits::is_output_serializable<BinaryData<typename
//   Derived::Scalar>,
//                                          Archive>::value,
//   void>::type
// save(Archive& ar, Eigen::PlainObjectBase<Derived> const& m)
// {
//   typedef Eigen::PlainObjectBase<Derived> ArrT;
//   if (ArrT::RowsAtCompileTime == Eigen::Dynamic)
//     ar(m.rows());
//   if (ArrT::ColsAtCompileTime == Eigen::Dynamic)
//     ar(m.cols());
//   ar(binary_data(m.data(), m.size() * sizeof(typename Derived::Scalar)));
// }

// template<class Archive, class Derived>
// inline typename std::enable_if<
//   cereal::traits::is_input_serializable<BinaryData<typename Derived::Scalar>,
//                                         Archive>::value,
//   void>::type
// load(Archive& ar, Eigen::PlainObjectBase<Derived>& m)
// {
//   typedef Eigen::PlainObjectBase<Derived> ArrT;
//   Eigen::Index rows = ArrT::RowsAtCompileTime, cols =
//   ArrT::ColsAtCompileTime; if (rows == Eigen::Dynamic)
//     ar(rows);
//   if (cols == Eigen::Dynamic)
//     ar(cols);
//   m.resize(rows, cols);
//   ar(binary_data(
//     m.data(),
//     static_cast<std::size_t>(rows * cols * sizeof(typename
//     Derived::Scalar))));
// }

// For JSON serialization this code is working
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
  int32_t rows = m.rows();
  int32_t cols = m.cols();
  ar(rows);
  ar(cols);

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      ar(m(i, j));
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
  int32_t rows;
  int32_t cols;
  ar(rows);
  ar(cols);

  m.resize(rows, cols);

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      ar(m(i, j));
}
}

#endif
#endif /* end of include guard PROXSUITE_HELPERS_SERIALIZE_HPP */
