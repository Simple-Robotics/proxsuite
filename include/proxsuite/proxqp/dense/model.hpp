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
#include "proxsuite/proxqp/dense/backward_data.hpp"
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
  Vec<T> u_box;
  Vec<T> l_box;

  ///// model sizes
  isize dim;
  isize n_eq;
  isize n_in;
  isize n_total;

  ///// Derivative data
  BackwardData<T> backward_data;

  /*!
   * Default constructor.
   * @param dim primal variable dimension.
   * @param n_eq number of equality constraints.
   * @param n_in number of inequality constraints.
   */
  Model(isize dim, isize n_eq, isize n_in, bool box_constraints = false)
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

    if (box_constraints) {
      u_box.resize(dim);
      l_box.resize(dim);
      u_box.fill(
        +infinite_bound_value); // in case it appears u is nullopt (i.e., the
                                // problem is only lower bounded)
      l_box.fill(
        -infinite_bound_value); // in case it appears l is nullopt (i.e., the
                                // problem is only upper bounded)
    }
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

  bool is_valid(const bool box_constraints)
  {
    // check that all matrices and vectors of qpmodel have the correct size
    // and that H and C have expected properties
    PROXSUITE_CHECK_ARGUMENT_SIZE(g.size(), dim, "g has not the expected size.")
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      b.size(), n_eq, "b has not the expected size.")
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      l.size(), n_in, "l has not the expected size.")
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      u.size(), n_in, "u has not the expected size.")
    if (box_constraints) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        u_box.size(), dim, "u_box has not the expected size");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        l_box.size(), dim, "l_box has not the expected size");
    }
    if (H.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.rows(), dim, "H has not the expected number of rows.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.cols(), dim, "H has not the expected number of cols.");
      PROXSUITE_THROW_PRETTY(
        (!H.isApprox(
          H.transpose(),
          std::numeric_limits<typename decltype(H)::Scalar>::epsilon())),
        std::invalid_argument,
        "H is not symmetric.");
    }
    if (A.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.rows(), n_eq, "A has not the expected number of rows.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.cols(), dim, "A has not the expected number of cols.");
    }
    if (C.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.rows(), n_in, "C has not the expected number of rows.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.cols(), dim, "C has not the expected number of cols.");
      PROXSUITE_THROW_PRETTY(
        C.isZero(), std::invalid_argument, "C is zero, while n_in != 0.");
    }
    return true;
  }
};

template<typename T>
bool
operator==(const Model<T>& model1, const Model<T>& model2)
{
  bool value = model1.dim == model2.dim && model1.n_eq == model2.n_eq &&
               model1.n_in == model2.n_in && model1.n_total == model2.n_total &&
               model1.H == model2.H && model1.g == model2.g &&
               model1.A == model2.A && model1.b == model2.b &&
               model1.C == model2.C && model1.l == model2.l &&
               model1.u == model2.u && model1.l_box == model2.l_box &&
               model1.u_box == model2.u_box;
  return value;
}

template<typename T>
bool
operator!=(const Model<T>& model1, const Model<T>& model2)
{
  return !(model1 == model2);
}

} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_MODEL_HPP */
