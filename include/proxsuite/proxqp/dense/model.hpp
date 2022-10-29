//
// Copyright (c) 2022 INRIA
//
/** \file */
#ifndef PROXSUITE_PROXQP_DENSE_MODEL_HPP
#define PROXSUITE_PROXQP_DENSE_MODEL_HPP

#include <Eigen/Core>
#include "proxsuite/linalg/veg/type_traits/core.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
#include "proxsuite/proxqp/sparse/model.hpp"

namespace proxsuite {
namespace proxqp {
namespace dense {
///
/// @brief This class stores the model of the QP problem.
///
/*!
 * Model class of the dense solver storing the QP problem.
 */
template<typename T>
struct Model
{

  ///// QP STORAGE
  Mat<T> H;
  Vec<T> g;
  Mat<T> A;
  Mat<T> C;
  Vec<T> b;
  Vec<T> u;
  Vec<T> l;

  ///// model sizes
  isize dim;
  isize n_eq;
  isize n_in;
  isize n_total;

  /*!
   * Default constructor.
   * @param dim primal variable dimension.
   * @param n_eq number of equality constraints.
   * @param n_in number of inequality constraints.
   */
  Model(isize dim, isize n_eq, isize n_in)
    : H(dim, dim)
    , g(dim)
    , A(n_eq, dim)
    , C(n_in, dim)
    , b(n_eq)
    , u(n_in)
    , l(n_in)
    , dim(dim)
    , n_eq(n_eq)
    , n_in(n_in)
    , n_total(dim + n_eq + n_in)
  {
    PROXSUITE_THROW_PRETTY(dim == 0,
                           std::invalid_argument,
                           "wrong argument size: the dimension wrt the primal "
                           "variable x should be strictly positive.");

    const T infinite_bound_value = helpers::infinite_bound<T>::value();

    H.setZero();
    g.setZero();
    A.setZero();
    C.setZero();
    b.setZero();
    u.fill(+infinite_bound_value); // in case it appears u is nullopt (i.e., the
                                   // problem is only lower bounded)
    l.fill(-infinite_bound_value); // in case it appears l is nullopt (i.e., the
                                   // problem is only upper bounded)
  }

  proxsuite::proxqp::sparse::SparseModel<T> to_sparse()
  {
    SparseMat<T> H_sparse = H.sparseView();
    SparseMat<T> A_sparse = A.sparseView();
    SparseMat<T> C_sparse = C.sparseView();
    proxsuite::proxqp::sparse::SparseModel<T> res{ H_sparse, g, A_sparse, b,
                                                   C_sparse, u, l };
    return res;
  }
};
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_MODEL_HPP */
