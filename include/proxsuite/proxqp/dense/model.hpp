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

  bool is_valid()
  {
    // check that all matrices and vectors of qpmodel have the correct size
    // and that H and C have expected properties
    if (g.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        g.size(),
        dim,
        "the dimension wrt the primal variable x variable for initializing g "
        "is not valid.");
    }
    if (b.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        b.size(),
        n_eq,
        "the dimension wrt equality constrained variables for initializing b "
        "is not valid.");
    }
    if (u.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        u.size(),
        n_in,
        "the dimension wrt inequality constrained variables for initializing u "
        "is not valid.");
    }
    if (l.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        l.size(),
        n_in,
        "the dimension wrt inequality constrained variables for initializing l "
        "is not valid.");
    }
    if (H.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.rows(), dim, "the row dimension for initializing H is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.cols(), dim, "the column dimension for initializing H is not valid.");

      // Check that H is not only upper/lower triangular matrix
      assert(H.isApprox(H.transpose()) && "H is not symmetric");
    }
    if (A.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.rows(), n_eq, "the row dimension for initializing A is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.cols(), dim, "the column dimension for initializing A is not valid.");
    }
    if (C.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.rows(), n_in, "the row dimension for initializing C is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.cols(), dim, "the column dimension for initializing C is not valid.");
      // If user inputs C it cannot be only zeros
      assert(!C.isZero() && "C contains only zeros, remove it.");
    }
    return true;
  }
};
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_MODEL_HPP */
