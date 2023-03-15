//
// Copyright (c) 2022 INRIA
//
/** \file */
#ifndef PROXSUITE_PROXQP_DENSE_LINESEARCH_HPP
#define PROXSUITE_PROXQP_DENSE_LINESEARCH_HPP

#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include <cmath>
namespace proxsuite {
namespace proxqp {
namespace dense {
namespace linesearch {
///
/// @brief This class stores the results of the primal-dual line-search.
///
/*!
 * Stores results of the line-search.
 *
 * @param a second order polynomial coefficient of the merit function used in
 * the linesearch.
 * @param b first order polynomial coefficient of the merit function used in the
 * linesearch.
 * @param grad derivative of the merit function used in the linesearch.
 */
template<typename T>
struct PrimalDualDerivativeResult
{
  T a;
  T b;
  T grad;
  VEG_REFLECT(PrimalDualDerivativeResult, a, b, grad);
};
/*!
 * Stores first derivative and coefficient of the univariate second order
 * polynomial merit function to be canceled by the exact generalized primal-dual
 * linesearch.
 *
 * @param a second order polynomial coefficient of the merit function used in
 * the linesearch.
 * @param b first order polynomial coefficient of the merit function used in the
 * linesearch.
 * @param grad derivative of the merit function used in the linesearch.
 */
template<typename T>
auto
gpdal_derivative_results(const Model<T>& qpmodel,
                         Results<T>& qpresults,
                         Workspace<T>& qpwork,
                         const Settings<T>& qpsettings,
                         isize n_constraints,
                         T alpha) -> PrimalDualDerivativeResult<T>
{

  /*
   * the function computes the first derivative of phi(alpha) at outer step k
   * and inner step l
   *
   * phi(alpha) = f(x_l+alpha dx) + rho/2 |x_l + alpha dx - x_k|**2
   *              + mu_eq_inv/2 (|A(x_l+alpha dx)-d+y_k * mu_eq|**2)
   *              + mu_eq_inv * nu /2 (|A(x_l+alpha dx)-d+y_k * mu_eq -
   * (y_l+alpha dy)
   * |**2)
   *              + mu_in_inv/2 ( | [C(x_l+alpha dx) - u + z_k * mu_in]_+ |**2
   *                         +| [C(x_l+alpha dx) - l + z_k * mu_in]_- |**2
   *                         )
   * 				+ mu_in_inv * nu / 2 ( | [C(x_l+alpha dx) - u +
   * z_k
   * * mu_in]_+
   * + [C(x_l+alpha dx) - l + z_k * mu_in]_- - (z+alpha dz) * mu_in |**2 with
   * f(x) = 0.5 * x^THx + g^Tx phi is a second order polynomial in alpha. Below
   * are computed its coefficients a0 and b0 in order to compute the desired
   * gradient a0 * alpha + b0
   */

  qpwork.primal_residual_in_scaled_up_plus_alphaCdx =
    qpwork.primal_residual_in_scaled_up + qpwork.Cdx * alpha;
  qpwork.primal_residual_in_scaled_low_plus_alphaCdx =
    qpresults.si + qpwork.Cdx * alpha;

  T a(qpwork.dw_aug.head(qpmodel.dim).dot(qpwork.Hdx) +
      qpresults.info.mu_eq_inv * (qpwork.Adx).squaredNorm() +
      qpresults.info.rho *
        qpwork.dw_aug.head(qpmodel.dim)
          .squaredNorm()); // contains now: a = dx.dot(H.dot(dx)) + rho *
                           // norm(dx)**2 + (mu_eq_inv) * norm(Adx)**2

  qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) =
    qpwork.Adx -
    qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) * qpresults.info.mu_eq;
  a +=
    qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).squaredNorm() *
    qpresults.info
      .mu_eq_inv; // contains now: a = dx.dot(H.dot(dx)) + rho * norm(dx)**2 +
  // (mu_eq_inv) * norm(Adx)**2 + mu_eq_inv * norm(Adx-dy*mu_eq)**2
  qpwork.err.head(qpmodel.dim) =
    qpresults.info.rho * (qpresults.x - qpwork.x_prev) + qpwork.g_scaled;
  T b(qpresults.x.dot(qpwork.Hdx) +
      (qpwork.err.head(qpmodel.dim)).dot(qpwork.dw_aug.head(qpmodel.dim)) +
      qpresults.info.mu_eq_inv *
        (qpwork.Adx)
          .dot(qpresults.se +
               qpresults.y * qpresults.info.mu_eq)); // contains now: b =
                                                     // dx.dot(H.dot(x) +
                                                     // rho*(x-xe) +  g)  +
  // mu_eq_inv * Adx.dot(res_eq)

  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpresults.se;
  b += qpresults.info.mu_eq_inv *
       qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)
         .dot(qpwork.rhs.segment(
           qpmodel.dim,
           qpmodel.n_eq)); // contains now: b = dx.dot(H.dot(x) + rho*(x-xe)
  // +  g)  + mu_eq_inv * Adx.dot(res_eq) + nu*mu_eq_inv *
  // (Adx-dy*mu_eq).dot(res_eq-y*mu_eq)

  // derive Cdx_act
  qpwork.err.tail(n_constraints) =
    ((qpwork.primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.)) ||
     (qpwork.primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.)))
      .select(qpwork.Cdx,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_constraints));

  a += qpresults.info.mu_in_inv * qpwork.err.tail(n_constraints).squaredNorm() /
       qpsettings.alpha_gpdal; // contains now: a = dx.dot(H.dot(dx)) + rho *
  // norm(dx)**2 + (mu_eq_inv) * norm(Adx)**2 + nu*mu_eq_inv *
  // norm(Adx-dy*mu_eq)**2 +
  // norm(dw_act)**2 / (mu_in * (alpha_gpdal))
  a += qpresults.info.mu_in * (1. - qpsettings.alpha_gpdal) *
       qpwork.dw_aug.tail(n_constraints).squaredNorm();
  // add norm(z)**2 * mu_in * (1-alpha)

  // derive vector [w-u]_+ + [w-l]--
  qpwork.active_part_z =
    (qpwork.primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.))
      .select(qpwork.primal_residual_in_scaled_up,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_constraints)) +
    (qpwork.primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.))
      .select(qpresults.si,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_constraints));

  b +=
    qpresults.info.mu_in_inv *
    qpwork.active_part_z.dot(qpwork.err.tail(n_constraints)) /
    qpsettings.alpha_gpdal; // contains now: b = dx.dot(H.dot(x) + rho*(x-xe) +
  // g)  + mu_eq_inv * Adx.dot(res_eq) + nu*mu_eq_inv *
  // (Adx-dy*mu_eq).dot(res_eq-y*mu_eq) + mu_in
  // * dw_act.dot([w-u]_+ + [w-l]--) / alpha_gpdal

  // contains now b =  dx.dot(H.dot(x) + rho*(x-xe) +  g)  + mu_eq_inv *
  // Adx.dot(res_eq) + nu*mu_eq_inv * (Adx-dy*mu_eq).dot(res_eq-y*mu_eq) +
  // mu_in_inv
  // * Cdx_act.dot([Cx-u+ze*mu_in]_+ + [Cx-l+ze*mu_in]--) + nu*mu_in_inv
  // (Cdx_act-dz*mu_in).dot([Cx-u+ze*mu_in]_+ + [Cx-l+ze*mu_in]-- - z*mu_in)
  b += qpresults.info.mu_in * (1. - qpsettings.alpha_gpdal) *
       qpwork.dw_aug.tail(n_constraints).dot(qpresults.z);

  return {
    a,
    b,
    a * alpha + b,
  };
}
/*!
 * Stores first derivative and coefficient of the univariate second order
 * polynomial merit function to be canceled by the exact primal-dual linesearch.
 *
 * @param a second order polynomial coefficient of the merit function used in
 * the linesearch.
 * @param b first order polynomial coefficient of the merit function used in the
 * linesearch.
 * @param grad derivative of the merit function used in the linesearch.
 */
template<typename T>
auto
primal_dual_derivative_results(const Model<T>& qpmodel,
                               Results<T>& qpresults,
                               Workspace<T>& qpwork,
                               isize n_constraints,
                               T alpha) -> PrimalDualDerivativeResult<T>
{

  /*
   * the function computes the first derivative of phi(alpha) at outer step k
   * and inner step l
   *
   * phi(alpha) = f(x_l+alpha dx) + rho/2 |x_l + alpha dx - x_k|**2
   *              + mu_eq_inv/2 (|A(x_l+alpha dx)-d+y_k * mu_eq|**2)
   *              + mu_eq_inv * nu /2 (|A(x_l+alpha dx)-d+y_k * mu_eq -
   * (y_l+alpha dy)
   * |**2)
   *              + mu_in_inv/2 ( | [C(x_l+alpha dx) - u + z_k * mu_in]_+ |**2
   *                         +| [C(x_l+alpha dx) - l + z_k * mu_in]_- |**2
   *                         )
   * 				+ mu_in_inv * nu / 2 ( | [C(x_l+alpha dx) - u +
   * z_k
   * * mu_in]_+
   * + [C(x_l+alpha dx) - l + z_k * mu_in]_- - (z+alpha dz) * mu_in |**2 with
   * f(x) = 0.5 * x^THx + g^Tx phi is a second order polynomial in alpha. Below
   * are computed its coefficients a0 and b0 in order to compute the desired
   * gradient a0 * alpha + b0
   */

  qpwork.primal_residual_in_scaled_up_plus_alphaCdx =
    qpwork.primal_residual_in_scaled_up + qpwork.Cdx * alpha;
  qpwork.primal_residual_in_scaled_low_plus_alphaCdx =
    qpresults.si + qpwork.Cdx * alpha;

  T a(qpwork.dw_aug.head(qpmodel.dim).dot(qpwork.Hdx) +
      qpresults.info.mu_eq_inv * (qpwork.Adx).squaredNorm() +
      qpresults.info.rho *
        qpwork.dw_aug.head(qpmodel.dim)
          .squaredNorm()); // contains now: a = dx.dot(H.dot(dx)) + rho *
                           // norm(dx)**2 + (mu_eq_inv) * norm(Adx)**2

  qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) =
    qpwork.Adx -
    qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) * qpresults.info.mu_eq;
  a += qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).squaredNorm() *
       qpresults.info.mu_eq_inv *
       qpresults.info
         .nu; // contains now: a = dx.dot(H.dot(dx)) + rho * norm(dx)**2 +
  // (mu_eq_inv) * norm(Adx)**2 + nu*mu_eq_inv * norm(Adx-dy*mu_eq)**2
  qpwork.err.head(qpmodel.dim) =
    qpresults.info.rho * (qpresults.x - qpwork.x_prev) + qpwork.g_scaled;
  // T b(qpresults.x.dot(qpwork.Hdx) +
  //     (qpwork.err.head(qpmodel.dim)).dot(qpwork.dw_aug.head(qpmodel.dim)) +
  //     qpresults.info.mu_eq_inv *
  //       (qpwork.Adx)
  //         .dot(qpwork.primal_residual_eq_scaled +
  //              qpresults.y * qpresults.info.mu_eq)); // contains now: b =
  //                                                    // dx.dot(H.dot(x) +
  //                                                    // rho*(x-xe) +  g)  +

  T b(qpresults.x.dot(qpwork.Hdx) +
      (qpwork.err.head(qpmodel.dim)).dot(qpwork.dw_aug.head(qpmodel.dim)) +
      qpresults.info.mu_eq_inv *
        (qpwork.Adx)
          .dot(qpresults.se +
               qpresults.y * qpresults.info.mu_eq)); // contains now: b =
                                                     // dx.dot(H.dot(x) +
                                                     // rho*(x-xe) +  g)  +
  // mu_eq_inv * Adx.dot(res_eq)
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpresults.se;
  b += qpresults.info.nu * qpresults.info.mu_eq_inv *
       qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)
         .dot(qpwork.rhs.segment(
           qpmodel.dim,
           qpmodel.n_eq)); // contains now: b = dx.dot(H.dot(x) + rho*(x-xe)
  // +  g)  + mu_eq_inv * Adx.dot(res_eq) + nu*mu_eq_inv *
  // (Adx-dy*mu_eq).dot(res_eq-y*mu_eq)

  // derive Cdx_act
  qpwork.err.tail(n_constraints) =
    ((qpwork.primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.)) ||
     (qpwork.primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.)))
      .select(qpwork.Cdx,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_constraints));

  a += qpresults.info.mu_in_inv *
       qpwork.err.tail(n_constraints)
         .squaredNorm(); // contains now: a = dx.dot(H.dot(dx)) + rho *
  // norm(dx)**2 + (mu_eq_inv) * norm(Adx)**2 + nu*mu_eq_inv *
  // norm(Adx-dy*mu_eq)**2 + mu_in *
  // norm(Cdx_act)**2

  // derive vector [Cx-u+ze/mu]_+ + [Cx-l+ze/mu]--
  qpwork.active_part_z =
    (qpwork.primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.))
      .select(qpwork.primal_residual_in_scaled_up,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_constraints)) +
    (qpwork.primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.))
      .select(qpresults.si,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_constraints));

  b += qpresults.info.mu_in_inv *
       qpwork.active_part_z.dot(qpwork.err.tail(
         n_constraints)); // contains now: b = dx.dot(H.dot(x) + rho*(x-xe) +
  // g)  + mu_eq_inv * Adx.dot(res_eq) + nu*mu_eq_inv *
  // (Adx-dy*mu_eq).dot(res_eq-y*mu_eq) + mu_in
  // * Cdx_act.dot([Cx-u+ze/mu]_+ + [Cx-l+ze*mu_in]--)

  // derive Cdx_act - dz*mu_in
  qpwork.err.tail(n_constraints) -=
    qpwork.dw_aug.tail(n_constraints) * qpresults.info.mu_in;
  // derive [Cx-u+ze*mu_in]_+ + [Cx-l+ze*mu_in]-- -z*mu_in
  qpwork.active_part_z -= qpresults.z * qpresults.info.mu_in;

  // contains now a = dx.dot(H.dot(dx)) + rho * norm(dx)**2 + (mu_eq_inv) *
  // norm(Adx)**2 + nu*mu_eq_inv * norm(Adx-dy*mu_eq)**2 + mu_in_inv *
  // norm(Cdx_act)**2 + nu*mu_in_inv * norm(Cdx_act-dz*mu_in)**2
  a += qpresults.info.nu * qpresults.info.mu_in_inv *
       qpwork.err.tail(n_constraints).squaredNorm();
  // contains now b =  dx.dot(H.dot(x) + rho*(x-xe) +  g)  + mu_eq_inv *
  // Adx.dot(res_eq) + nu*mu_eq_inv * (Adx-dy*mu_eq).dot(res_eq-y*mu_eq) +
  // mu_in_inv
  // * Cdx_act.dot([Cx-u+ze*mu_in]_+ + [Cx-l+ze*mu_in]--) + nu*mu_in_inv
  // (Cdx_act-dz*mu_in).dot([Cx-u+ze*mu_in]_+ + [Cx-l+ze*mu_in]-- - z*mu_in)
  b += qpresults.info.nu * qpresults.info.mu_in_inv *
       qpwork.err.tail(n_constraints).dot(qpwork.active_part_z);

  return {
    a,
    b,
    a * alpha + b,
  };
}
/*!
 * Performs the exact primaldual linesearch algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
void
primal_dual_ls(const Model<T>& qpmodel,
               Results<T>& qpresults,
               Workspace<T>& qpwork,
               const Settings<T>& qpsettings,
               const isize n_constraints)
{

  /*
   * The algorithm performs the following step
   *
   * 1/
   * 1.1/ Store solutions of equations
   * C(x+alpha dx) - l + ze/mu_in = 0
   * C(x+alpha dx) - u + ze/mu_in = 0
   *
   * 1.2/ Sort the alpha
   * 2/
   * 2.1
   * For each positive alpha compute the first derivative of
   * phi(alpha) = [proximal primal dual augmented lagrangian of the subproblem
   * evaluated at x_k + alpha dx, y_k + alpha dy, z_k + alpha dz] using function
   * "gradient_norm" By construction for alpha = 0, phi'(alpha) <= 0 and
   * phi'(alpha) goes to infinity with alpha hence it cancels uniquely at one
   * optimal alpha*
   *
   * while phi'(alpha)<=0 store the derivative (noted last_grad_neg) and
   * alpha (last_alpha_neg)
   * the first time phi'(alpha) > 0 store the derivative (noted
   * first_grad_pos) and alpha (first_alpha_pos), and break the loo
   *
   * 2.2
   * If first_alpha_pos corresponds to the first positive alpha of previous
   * loop, then do
   *   last_alpha_neg = 0
   *   last_grad_neg = phi'(0)
   *
   * 2.3
   * the optimal alpha is within the interval
   * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi' is
   * an affine function in alph
   * alpha* = alpha_last_neg
   *        - last_neg_grad * (alpha_first_pos - alpha_last_neg) /
   *                          (first_pos_grad - last_neg_grad);
   */

  const T machine_eps = std::numeric_limits<T>::epsilon();

  qpwork.alpha = T(1);
  T alpha_(1.);

  qpwork.alphas.clear();

  ///////// STEP 1 /////////
  // 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
  // dx)-u +ze/mu_in = 0

  for (isize i = 0; i < n_constraints; i++) {

    if (qpwork.Cdx(i) != 0.) {
      alpha_ =
        -qpwork.primal_residual_in_scaled_up(i) / (qpwork.Cdx(i) + machine_eps);
      if (alpha_ > machine_eps) {
        qpwork.alphas.push(alpha_);
      }
      alpha_ = -qpresults.si(i) / (qpwork.Cdx(i) + machine_eps);
      if (alpha_ > machine_eps) {
        qpwork.alphas.push(alpha_);
      }
    }
  }

  isize n_alpha = qpwork.alphas.len();

  // 1.2 sort the alphas

  std::sort(qpwork.alphas.ptr_mut(), qpwork.alphas.ptr_mut() + n_alpha);
  isize new_len = std::unique( //
                    qpwork.alphas.ptr_mut(),
                    qpwork.alphas.ptr_mut() + n_alpha) -
                  qpwork.alphas.ptr_mut();
  qpwork.alphas.resize(new_len);

  n_alpha = qpwork.alphas.len();
  if (n_alpha == 0) { //
    switch (qpsettings.merit_function_type) {
      case MeritFunctionType::GPDAL: {
        auto res = gpdal_derivative_results(
          qpmodel, qpresults, qpwork, qpsettings, n_constraints, T(0));
        qpwork.alpha = -res.b / res.a;
      } break;
      case MeritFunctionType::PDAL: {
        auto res = primal_dual_derivative_results(
          qpmodel, qpresults, qpwork, n_constraints, T(0));
        qpwork.alpha = -res.b / res.a;
      } break;
    }
    return;
  }
  ////////// STEP 2 ///////////
  auto infty = std::numeric_limits<T>::infinity();

  T last_neg_grad = 0;
  T alpha_last_neg = 0;
  T first_pos_grad = 0;
  T alpha_first_pos = infty;
  for (isize i = 0; i < n_alpha; ++i) {
    alpha_ = qpwork.alphas[i];

    /*
     * 2.1
     * For each positive alpha compute the first derivative of
     * phi(alpha) = [proximal augmented lagrangian of the
     *               subproblem evaluated at x_k + alpha dx]
     *
     * (By construction for alpha = 0,  phi'(alpha) <= 0 and
     * phi'(alpha) goes to infinity with alpha hence it cancels
     * uniquely at one optimal alpha*
     *
     * while phi'(alpha)<=0 store the derivative (noted
     * last_grad_neg) and alpha (last_alpha_neg
     * the first time phi'(alpha) > 0 store the derivative
     * (noted first_grad_pos) and alpha (first_alpha_pos), and
     * break the loop
     */
    T gr(0);
    switch (qpsettings.merit_function_type) {
      case MeritFunctionType::GPDAL:
        gr = gpdal_derivative_results(
               qpmodel, qpresults, qpwork, qpsettings, n_constraints, alpha_)
               .grad;
        break;
      case MeritFunctionType::PDAL:
        gr = primal_dual_derivative_results(
               qpmodel, qpresults, qpwork, n_constraints, alpha_)
               .grad;
        break;
    }

    if (gr < T(0)) {
      alpha_last_neg = alpha_;
      last_neg_grad = gr;
    } else {
      first_pos_grad = gr;
      alpha_first_pos = alpha_;
      break;
    }
  }

  /*
   * 2.2
   * If first_alpha_pos corresponds to the first positive alpha of
   * previous loop, then do
   * last_alpha_neg = 0 and last_grad_neg = phi'(0) using function
   * "gradient_norm"
   */
  if (alpha_last_neg == T(0)) {
    switch (qpsettings.merit_function_type) {
      case MeritFunctionType::GPDAL:
        last_neg_grad = gpdal_derivative_results(qpmodel,
                                                 qpresults,
                                                 qpwork,
                                                 qpsettings,
                                                 n_constraints,
                                                 alpha_last_neg)
                          .grad;
        break;
      case MeritFunctionType::PDAL:
        last_neg_grad =
          primal_dual_derivative_results(
            qpmodel, qpresults, qpwork, n_constraints, alpha_last_neg)
            .grad;
        break;
    }
  }
  if (alpha_first_pos == infty) {
    /*
     * 2.3
     * the optimal alpha is within the interval
     * [last_alpha_neg, +∞)
     */
    switch (qpsettings.merit_function_type) {
      case MeritFunctionType::GPDAL: {
        PrimalDualDerivativeResult<T> res =
          gpdal_derivative_results(qpmodel,
                                   qpresults,
                                   qpwork,
                                   qpsettings,
                                   n_constraints,
                                   2 * alpha_last_neg + 1);
        auto& a = res.a;
        auto& b = res.b;
        // grad = a * alpha + b
        // grad = 0 => alpha = -b/a
        qpwork.alpha = -b / a;
      } break;
      case MeritFunctionType::PDAL: {
        PrimalDualDerivativeResult<T> res = primal_dual_derivative_results(
          qpmodel, qpresults, qpwork, n_constraints, 2 * alpha_last_neg + 1);
        auto& a = res.a;
        auto& b = res.b;
        // grad = a * alpha + b
        // grad = 0 => alpha = -b/a
        qpwork.alpha = -b / a;
      } break;
    }
  } else {
    /*
     * 2.3
     * the optimal alpha is within the interval
     * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi'
     * is an affine function in alpha
     */
    qpwork.alpha = std::abs(alpha_last_neg -
                            last_neg_grad * (alpha_first_pos - alpha_last_neg) /
                              (first_pos_grad - last_neg_grad));
  }
}

/*!
 * Performs the active set change of the factorized KKT matrix (using rank one
 * updates or downgrades).
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
void
active_set_change(const Model<T>& qpmodel,
                  Results<T>& qpresults,
                  const DenseBackend& dense_backend,
                  const isize n_constraints,
                  Workspace<T>& qpwork)
{

  /*
   * arguments
   * 1/ new_active_set : a vector which contains new active set of the
   * problem, namely if
   * new_active_set_u = Cx_k-u +z_k*mu_in>= 0
   * new_active_set_l = Cx_k-l +z_k*mu_in<=
   * then new_active_set = new_active_set_u OR new_active_set_
   *
   * 2/ current_bijection_map : a vector for which each entry corresponds to
   * the current row of C of the current factorization
   *
   * for example, naming C_initial the initial C matrix of the problem, and
   * C_current the one of the current factorization, the
   * C_initial[i,:] = C_current[current_bijection_mal[i],:] for all
   *
   * 3/ n_c : the current number of active_inequalities
   * This algorithm ensures that for all new version of C_current in the LDLT
   * factorization all row index i < n_c correspond to current active indexes
   * (all other correspond to inactive rows
   *
   * To do so,
   * 1/ for initialization
   * 1.1/ new_bijection_map = current_bijection_map
   * 1.2/ n_c_f = n_
   *
   * 2/ All active indexes of the current bijection map (i.e
   * current_bijection_map(i) < n_c by assumption) which are not active
   * anymore in the new active set (new_active_set(i)=false are put at the
   * end of new_bijection_map, i.
   *
   * 2.1/ for all j if new_bijection_map(j) > new_bijection_map(i), then
   * new_bijection_map(j)-=1
   * 2.2/ n_c_f -=1
   * 2.3/ new_bijection_map(i) = n_in-1
   *
   * 3/ All active indexe of the new active set (new_active_set(i) == true)
   * which are not active in the new_bijection_map (new_bijection_map(i) >=
   * n_c_f) are put at the end of the current version of C, i.e
   * 3.1/ if new_bijection_map(j) < new_bijection_map(i) &&
   * new_bijection_map(j) >= n_c_f then new_bijection_map(j)+=1
   * 3.2/ new_bijection_map(i) = n_c_f
   * 3.3/ n_c_f +=1
   *
   * It returns finally the new_bijection_map, for which
   * new_bijection_map(n_in) = n_c_f
   */

  isize n_c_f = qpwork.n_c;
  qpwork.new_bijection_map = qpwork.current_bijection_map;

  // suppression pour le nouvel active set, ajout dans le nouvel unactive set

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  {
    auto _planned_to_delete = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, isize(n_constraints));
    isize* planned_to_delete = _planned_to_delete.ptr_mut();
    isize planned_to_delete_count = 0;

    for (isize i = 0; i < n_constraints; i++) {
      if (qpwork.current_bijection_map(i) < qpwork.n_c) {
        if (!qpwork.active_inequalities(i)) {
          // delete current_bijection_map(i)

          planned_to_delete[planned_to_delete_count] =
            qpwork.current_bijection_map(i) + qpmodel.dim + qpmodel.n_eq;
          ++planned_to_delete_count;

          for (isize j = 0; j < n_constraints; j++) {
            if (qpwork.new_bijection_map(j) > qpwork.new_bijection_map(i)) {
              qpwork.new_bijection_map(j) -= 1;
            }
          }
          n_c_f -= 1;
          qpwork.new_bijection_map(i) = n_constraints - 1;
        }
      }
    }
    std::sort(planned_to_delete, planned_to_delete + planned_to_delete_count);
    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT:
        qpwork.ldl.delete_at(planned_to_delete, planned_to_delete_count, stack);
        break;
      case DenseBackend::PrimalLDLT: {
        // for (isize i=0; i < planned_to_delete_count; i++){
        //   isize index = planned_to_delete[i] - (qpmodel.dim + qpmodel.n_eq);
        //   if (index >= qpmodel.n_in){
        //     // bow constraint
        //     qpwork.kkt(index-qpmodel.n_in,index-qpmodel.n_in) -=
        //     qpresults.info.mu_in_inv *
        //     std::pow(qpwork.i_scaled[index-qpmodel.n_in],2) ;
        //   }else{
        //     // generic ineq constraints
        //     qpwork.kkt.noalias() -= qpresults.info.mu_in_inv *
        //     qpwork.C_scaled.row(index).transpose() *
        //     qpwork.C_scaled.row(index) ;
        //   }
        // }
        LDLT_TEMP_MAT_UNINIT(
          T, new_cols, qpmodel.dim, planned_to_delete_count, stack);
        qpwork.dw_aug.head(planned_to_delete_count).setOnes();
        T mu_in_inv_neg(-qpresults.info.mu_in_inv);
        qpwork.dw_aug.head(planned_to_delete_count).array() *= mu_in_inv_neg;
        for (isize i = 0; i < planned_to_delete_count; ++i) {
          isize index = planned_to_delete[i] - (qpmodel.dim + qpmodel.n_eq);
          auto col = new_cols.col(i);
          if (index >= qpmodel.n_in) {
            // box constraint
            col.setZero();
            col[index - qpmodel.n_in] = qpwork.i_scaled[index - qpmodel.n_in];
          } else {
            // generic ineq constraints
            col = qpwork.C_scaled.row(index);
          }
        }
        qpwork.ldl.rank_r_update(
          new_cols, qpwork.dw_aug.head(planned_to_delete_count), stack);
      } break;
      case DenseBackend::Automatic:
        break;
    }
    if (planned_to_delete_count > 0) {
      qpwork.constraints_changed = true;
    }
  }
  // ajout au nouvel active set, suppression pour le nouvel unactive set

  {
    auto _planned_to_add = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, n_constraints);
    auto planned_to_add = _planned_to_add.ptr_mut();

    isize planned_to_add_count = 0;
    T mu_in_neg(-qpresults.info.mu_in);

    isize n_c = n_c_f;
    for (isize i = 0; i < n_constraints; i++) {
      if (qpwork.active_inequalities(i)) {
        if (qpwork.new_bijection_map(i) >= n_c_f) {
          // add at the end
          planned_to_add[planned_to_add_count] = i;
          ++planned_to_add_count;

          for (isize j = 0; j < n_constraints; j++) {
            if (qpwork.new_bijection_map(j) < qpwork.new_bijection_map(i) &&
                qpwork.new_bijection_map(j) >= n_c_f) {
              qpwork.new_bijection_map(j) += 1;
            }
          }
          qpwork.new_bijection_map(i) = n_c_f;
          n_c_f += 1;
        }
      }
    }
    {
      switch (dense_backend) {
        case DenseBackend::PrimalDualLDLT: {
          isize n = qpmodel.dim;
          isize n_eq = qpmodel.n_eq;
          LDLT_TEMP_MAT_UNINIT(
            T, new_cols, n + n_eq + n_c_f, planned_to_add_count, stack);

          for (isize k = 0; k < planned_to_add_count; ++k) {
            isize index = planned_to_add[k];
            auto col = new_cols.col(k);
            if (index >= qpmodel.n_in) {
              col.head(n).setZero();
              // I_scaled = ED which is the diagonal matrix
              col[index - qpmodel.n_in] = qpwork.i_scaled[index - qpmodel.n_in];
            } else {
              col.head(n) = (qpwork.C_scaled.row(index));
            }
            col.tail(n_eq + n_c_f).setZero();
            col[n + n_eq + n_c + k] = mu_in_neg;
          }
          qpwork.ldl.insert_block_at(n + n_eq + n_c, new_cols, stack);
        } break;
        case DenseBackend::PrimalLDLT: {
          // too slow
          // for (isize i=0; i < planned_to_add_count; ++i){
          //   isize index = planned_to_add[i];
          //   if (index >= qpmodel.n_in){
          //     // bow constraint
          //     qpwork.kkt(index-qpmodel.n_in,index-qpmodel.n_in) +=
          //     qpresults.info.mu_in_inv *
          //     std::pow(qpwork.i_scaled[index-qpmodel.n_in],2) ;
          //   }else{
          //     // generic ineq constraints
          //     qpwork.kkt.noalias() += qpresults.info.mu_in_inv *
          //     qpwork.C_scaled.row(index).transpose() *
          //     qpwork.C_scaled.row(index) ;
          //   }
          // }
          LDLT_TEMP_MAT_UNINIT(
            T, new_cols, qpmodel.dim, planned_to_add_count, stack);
          qpwork.dw_aug.head(planned_to_add_count).setOnes();
          qpwork.dw_aug.head(planned_to_add_count).array() *=
            qpresults.info.mu_in_inv;
          for (isize i = 0; i < planned_to_add_count; ++i) {
            isize index = planned_to_add[i];
            auto col = new_cols.col(i);
            if (index >= qpmodel.n_in) {
              // box constraint
              col.setZero();
              col[index - qpmodel.n_in] = qpwork.i_scaled[index - qpmodel.n_in];
            } else {
              // generic ineq constraints
              col.head(qpmodel.dim) = qpwork.C_scaled.row(index);
            }
          }
          qpwork.ldl.rank_r_update(
            new_cols, qpwork.dw_aug.head(planned_to_add_count), stack);
        }
        // qpwork.ldl.factorize(qpwork.kkt.transpose(), stack);
        break;
        case DenseBackend::Automatic:
          break;
      }
    }
    if (planned_to_add_count > 0) {
      qpwork.constraints_changed = true;
    }
  }

  qpwork.n_c = n_c_f;
  qpwork.current_bijection_map = qpwork.new_bijection_map;
}

} // namespace linesearch
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_LINESEARCH_HPP */
