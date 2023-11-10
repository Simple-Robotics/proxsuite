//
// Copyright (c) 2022 INRIA
//
/**
 * @file ruiz.hpp
 */
#ifndef PROXSUITE_PROXQP_DENSE_PRECOND_RUIZ_HPP
#define PROXSUITE_PROXQP_DENSE_PRECOND_RUIZ_HPP

#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <ostream>
#include <iostream>
#include <Eigen/Core>

namespace proxsuite {
namespace proxqp {
enum struct Symmetry
{
  general,
  lower,
  upper,
};
namespace dense {
namespace detail {

template<typename T>
auto
ruiz_scale_qp_in_place( //
  VectorViewMut<T> delta_,
  std::ostream* logger_ptr,
  QpViewBoxMut<T> qp,
  T epsilon,
  isize max_iter,
  bool preconditioning_for_infeasible_problems,
  Symmetry sym,
  HessianType HessianType,
  const bool box_constraints,
  proxsuite::linalg::veg::dynstack::DynStackMut stack) -> T
{
  T c(1);
  auto S = delta_.to_eigen();
  auto H = qp.H.to_eigen();
  auto g = qp.g.to_eigen();
  auto A = qp.A.to_eigen();
  auto b = qp.b.to_eigen();
  auto C = qp.C.to_eigen();
  auto u = qp.u.to_eigen();
  auto l = qp.l.to_eigen();

  auto l_box = qp.l_box.to_eigen();
  auto u_box = qp.u_box.to_eigen();
  auto i_scaled = qp.I.to_eigen();
  i_scaled.setOnes();

  static constexpr T machine_eps = std::numeric_limits<T>::epsilon();
  /*
   * compute equilibration parameters and scale in place the qp following
   * algorithm
   *
   * modified: removed g in gamma computation
   */

  isize n = qp.H.rows;
  isize n_eq = qp.A.rows;
  isize n_in = qp.C.rows;
  isize n_constraints(n_in);
  if (box_constraints) {
    n_constraints += n;
  }

  T gamma = T(1);
  LDLT_TEMP_VEC(T, delta, n + n_eq + n_constraints, stack);
  i64 iter = 1;
  while (infty_norm((1 - delta.array()).matrix()) > epsilon) {

    if (logger_ptr != nullptr) {
      *logger_ptr                                   //
        << "j : "                                   //
        << iter                                     //
        << " ; error : "                            //
        << infty_norm((1 - delta.array()).matrix()) //
        << "\n\n";
    }
    if (iter == max_iter) {
      break;
    } else {
      ++iter;
    }
    // normalization vector
    {
      switch (HessianType) {
        case HessianType::Zero:
          for (isize k = 0; k < n; ++k) {
            T aux = sqrt(std::max({ n_eq > 0 ? infty_norm(A.col(k)) : T(0),
                                    n_in > 0 ? infty_norm(C.col(k)) : T(0),
                                    box_constraints ? i_scaled[k] : T(0) }));
            if (aux == T(0)) {
              delta(k) = T(1);
            } else {
              delta(k) = T(1) / (aux + machine_eps);
            }
          }
          break;
        case HessianType::Dense:
          for (isize k = 0; k < n; ++k) {
            switch (sym) {
              case Symmetry::upper: { // upper triangular part
                T aux =
                  sqrt(std::max({ infty_norm(H.col(k).head(k)),
                                  infty_norm(H.row(k).tail(n - k)),
                                  n_eq > 0 ? infty_norm(A.col(k)) : T(0),
                                  n_in > 0 ? infty_norm(C.col(k)) : T(0),
                                  box_constraints ? i_scaled[k] : T(0) }));
                if (aux == T(0)) {
                  delta(k) = T(1);
                } else {
                  delta(k) = T(1) / (aux + machine_eps);
                }
                break;
              }
              case Symmetry::lower: { // lower triangular part

                T aux =
                  sqrt(std::max({ infty_norm(H.col(k).head(k)),
                                  infty_norm(H.col(k).tail(n - k)),
                                  n_eq > 0 ? infty_norm(A.col(k)) : T(0),
                                  n_in > 0 ? infty_norm(C.col(k)) : T(0),
                                  box_constraints ? i_scaled[k] : T(0) }));
                if (aux == T(0)) {
                  delta(k) = T(1);
                } else {
                  delta(k) = T(1) / (aux + machine_eps);
                }
                break;
              }
              case Symmetry::general: {

                T aux =
                  sqrt(std::max({ infty_norm(H.col(k)),
                                  n_eq > 0 ? infty_norm(A.col(k)) : T(0),
                                  n_in > 0 ? infty_norm(C.col(k)) : T(0),
                                  box_constraints ? i_scaled[k] : T(0) }));
                if (aux == T(0)) {
                  delta(k) = T(1);
                } else {
                  delta(k) = T(1) / (aux + machine_eps);
                }
                break;
              }
            }
          }
          break;
        case HessianType::Diagonal:
          for (isize k = 0; k < n; ++k) {
            T aux = sqrt(std::max({ std::abs(H(k, k)),
                                    n_eq > 0 ? infty_norm(A.col(k)) : T(0),
                                    n_in > 0 ? infty_norm(C.col(k)) : T(0),
                                    box_constraints ? i_scaled[k] : T(0) }));
            if (aux == T(0)) {
              delta(k) = T(1);
            } else {
              delta(k) = T(1) / (aux + machine_eps);
            }
          }
          break;
      }
      if (preconditioning_for_infeasible_problems) {
        delta.tail(n_eq + n_constraints).setOnes();
      } else {
        for (isize k = 0; k < n_eq; ++k) {
          T aux = sqrt(infty_norm(A.row(k)));
          if (aux == T(0)) {
            delta(n + k) = T(1);
          } else {
            delta(n + k) = T(1) / (aux + machine_eps);
          }
        }
        for (isize k = 0; k < n_in; ++k) {
          T aux = sqrt(infty_norm(C.row(k)));
          if (aux == T(0)) {
            delta(k + n + n_eq) = T(1);
          } else {
            delta(k + n + n_eq) = T(1) / (aux + machine_eps);
          }
        }
        if (box_constraints) {
          for (isize k = 0; k < n; ++k) {
            delta(k + n + n_eq + n_in) = T(1) / sqrt(i_scaled[k] + machine_eps);
          }
        }
      }
      // removed as non deterministic when using avx
      // https://gitlab.com/libeigen/eigen/-/issues/1728
      // if (preconditioning_for_infeasible_problems) {
      //   T mean = delta.segment(n, n_eq_in).mean();
      //   delta.segment(n,n_eq_in).setConstant(mean);
      // }
    }
    {

      // normalize A and C
      A = delta.segment(n, n_eq).asDiagonal() * A * delta.head(n).asDiagonal();
      C = delta.segment(n + n_eq, n_in).asDiagonal() * C *
          delta.head(n).asDiagonal();
      if (box_constraints) {
        i_scaled.array() *= delta.head(n).array();
        i_scaled.array() *= delta.tail(n).array();
        u_box.array() *= delta.tail(n).array();
        l_box.array() *= delta.tail(n).array();
      }
      // normalize vectors
      g.array() *= delta.head(n).array();
      b.array() *= delta.segment(n, n_eq).array();
      u.array() *= delta.segment(n + n_eq, n_in).array();
      l.array() *= delta.segment(n + n_eq, n_in).array();

      // normalize H
      switch (HessianType) {
        case HessianType::Zero:
          break;
        case HessianType::Dense:
          switch (sym) {
            case Symmetry::upper: {
              // upper triangular part
              for (isize j = 0; j < n; ++j) {
                H.col(j).head(j + 1) *= delta(j);
              }
              // normalisation des lignes
              for (isize i = 0; i < n; ++i) {
                H.row(i).tail(n - i) *= delta(i);
              }
              break;
            }
            case Symmetry::lower: {
              // lower triangular part
              for (isize j = 0; j < n; ++j) {
                H.col(j).tail(n - j) *= delta(j);
              }
              // normalisation des lignes
              for (isize i = 0; i < n; ++i) {
                H.row(i).head(i + 1) *= delta(i);
              }
              break;
            }
            case Symmetry::general: {
              // all matrix
              H = delta.head(n).asDiagonal() * H * delta.head(n).asDiagonal();
              break;
            }
            default:
              break;
          }
          // additional normalization for the cost function
          switch (sym) {
            case Symmetry::upper: {
              // upper triangular part
              T tmp = T(0);
              for (isize j = 0; j < n; ++j) {
                tmp += proxqp::dense::infty_norm(H.row(j).tail(n - j));
              }
              gamma = 1 / std::max(tmp / T(n), T(1));
              break;
            }
            case Symmetry::lower: {
              // lower triangular part
              T tmp = T(0);
              for (isize j = 0; j < n; ++j) {
                tmp += proxqp::dense::infty_norm(H.col(j).tail(n - j));
              }
              gamma = 1 / std::max(tmp / T(n), T(1));
              break;
            }
            case Symmetry::general: {
              // all matrix
              gamma =
                1 / std::max(
                      T(1),
                      (H.colwise().template lpNorm<Eigen::Infinity>()).mean());
              break;
            }
            default:
              break;
          }
          break;
        case HessianType::Diagonal:
          // H = delta.head(n).asDiagonal() * H.asDiagonal() *
          // delta.head(n).asDiagonal();
          H.diagonal().array() *=
            delta.head(n)
              .array(); //* H.asDiagonal() * delta.head(n).asDiagonal();
          H.diagonal().array() *=
            delta.head(n)
              .array(); //* H.asDiagonal() * delta.head(n).asDiagonal();
          gamma =
            1 /
            std::max(T(1),
                     (H.diagonal().template lpNorm<Eigen::Infinity>()) / T(n));
          H *= gamma;
          break;
      }
      g *= gamma;

      S.array() *= delta.array(); // coefficientwise product
      c *= gamma;
    }
  }
  return c;
}
} // namespace detail

namespace preconditioner {

template<typename T>
struct RuizEquilibration
{
  Vec<T> delta;
  T c;
  isize dim;
  isize n_eq;
  isize n_in;
  T epsilon;
  i64 max_iter;
  Symmetry sym;

  std::ostream* logger_ptr = nullptr;
  /*!
   * Default constructor.
   * @param dim primal variable dimension.
   * @param n_eq_constraints number of equality and inequality constraints
   * (n_in+n_constraints if box constraints are present, or n_in).
   * @param epsilon_ accuracy required for stopping the ruiz equilibration
   * algorithm.
   * @param max_iter_ maximum number of ruiz equilibration iterations.
   * @param sym_ symetry option format of quadratic cost matrix.
   * @param logger parameter for printing or not intermediary results.
   */
  explicit RuizEquilibration(isize dim_,
                             isize n_eq_,
                             isize n_in_,
                             bool box_constraints,
                             T epsilon_ = T(1e-3),
                             i64 max_iter_ = 10,
                             Symmetry sym_ = Symmetry::general,
                             std::ostream* logger = nullptr)
    : delta(Vec<T>::Ones(dim_ + n_eq_ + n_in_ + (box_constraints ? dim_ : 0)))
    , c(1)
    , dim(dim_)
    , n_eq(n_eq_)
    , n_in(n_in_)
    , epsilon(epsilon_)
    , max_iter(max_iter_)
    , sym(sym_)
    , logger_ptr(logger)
  {
  }
  /*!
   * Prints ruiz equilibrator scaling variables.
   */
  void print() const
  {
    // CHANGE: endl to newline
    *logger_ptr << " delta : " << delta << "\n\n";
    *logger_ptr << " c : " << c << "\n\n";
  }
  /*!
   * Determines memory requirements for executing the equilibrator.
   * @param tag tag for specifying entry type.
   * @param n dimension of the primal variable of the model.
   * @param n_eq number of equality constraints.
   * @param n_in number of inequality constraints.
   */
  static auto scale_qp_in_place_req(proxsuite::linalg::veg::Tag<T> tag,
                                    isize n,
                                    isize n_eq,
                                    isize n_in,
                                    bool box_constraints)
    -> proxsuite::linalg::veg::dynstack::StackReq
  {
    if (box_constraints) {
      return proxsuite::linalg::dense::temp_vec_req(tag, 2 * n + n_eq + n_in);
    } else {
      return proxsuite::linalg::dense::temp_vec_req(tag, n + n_eq + n_in);
    }
  }

  // H_new = c * head @ H @ head
  // A_new = tail @ A @ head
  // g_new = c * head @ g
  // b_new = tail @ b
  /*!
   * Scales the qp performing the ruiz equilibrator algorithm considering user
   * options.
   * @param qp qp to be scaled (in place).
   * @param execute_preconditioner bool variable specifying whether the qp is
   * scaled using current equilibrator scaling variables, or performing anew the
   * algorithm.
   * @param settings solver's settings.
   * @param stack stack variable used by the equilibrator.
   */
  void scale_qp_in_place(QpViewBoxMut<T> qp,
                         bool execute_preconditioner,
                         bool preconditioning_for_infeasible_problems,
                         const isize max_iter,
                         const T epsilon,
                         const HessianType& HessianType,
                         const bool box_constraints,
                         proxsuite::linalg::veg::dynstack::DynStackMut stack)
  {
    if (execute_preconditioner) {
      delta.setOnes();
      c =
        detail::ruiz_scale_qp_in_place({ proxqp::from_eigen, delta },
                                       logger_ptr,
                                       qp,
                                       epsilon,
                                       max_iter,
                                       preconditioning_for_infeasible_problems,
                                       sym,
                                       HessianType,
                                       box_constraints,
                                       stack);
    } else {

      auto H = qp.H.to_eigen();
      auto g = qp.g.to_eigen();
      auto A = qp.A.to_eigen();
      auto b = qp.b.to_eigen();
      auto C = qp.C.to_eigen();
      auto u = qp.u.to_eigen();
      auto l = qp.l.to_eigen();
      auto l_box = qp.l_box.to_eigen();
      auto u_box = qp.u_box.to_eigen();
      auto i_scaled = qp.I.to_eigen(); // it is a vector
      isize n = qp.H.rows;
      isize n_eq = qp.A.rows;
      isize n_in = qp.C.rows;

      // normalize A and C
      A = delta.segment(n, n_eq).asDiagonal() * A * delta.head(n).asDiagonal();
      C = delta.segment(n + n_eq, n_in).asDiagonal() * C *
          delta.head(n).asDiagonal();

      // normalize H
      switch (HessianType) {
        case HessianType::Dense:
          switch (sym) {
            case Symmetry::upper: {
              // upper triangular part
              for (isize j = 0; j < n; ++j) {
                H.col(j).head(j + 1) *= delta(j);
              }
              // normalisation des lignes
              for (isize i = 0; i < n; ++i) {
                H.row(i).tail(n - i) *= delta(i);
              }
              break;
            }
            case Symmetry::lower: {
              // lower triangular part
              for (isize j = 0; j < n; ++j) {
                H.col(j).tail(n - j) *= delta(j);
              }
              // normalisation des lignes
              for (isize i = 0; i < n; ++i) {
                H.row(i).head(i + 1) *= delta(i);
              }
              break;
            }
            case Symmetry::general: {
              // all matrix
              H = delta.head(n).asDiagonal() * H * delta.head(n).asDiagonal();
              break;
            }
            default:
              break;
          }
          break;

        case HessianType::Zero:
          break;
        case HessianType::Diagonal:
          // H = delta.head(n).asDiagonal() * H.asDiagonal() *
          // delta.head(n).asDiagonal();
          H.diagonal().array() *=
            delta.head(n)
              .array(); //* H.asDiagonal() * delta.head(n).asDiagonal();
          H.diagonal().array() *=
            delta.head(n)
              .array(); //* H.asDiagonal() * delta.head(n).asDiagonal();
          break;
      }

      // normalize vectors
      g.array() *= delta.head(n).array();
      b.array() *= delta.segment(n, n_eq).array();
      l.array() *= delta.segment(n + n_eq, n_in).array();
      u.array() *= delta.segment(n + n_eq, n_in).array();

      if (box_constraints) {
        u_box.array() *= delta.tail(n).array();
        l_box.array() *= delta.tail(n).array();
        i_scaled.array() *= delta.tail(n).array();
        i_scaled.array() *= delta.head(n).array();
      }

      g *= c;
      H *= c;
    }
  }
  // modifies variables in place
  /*!
   * Scales a primal variable in place.
   * @param primal primal variable.
   */
  void scale_primal_in_place(VectorViewMut<T> primal) const
  {
    primal.to_eigen().array() /= delta.array().head(dim);
  }
  /*!
   * Scales a dual variable in place.
   * @param dual dual variable (includes all equalities and inequalities
   * constraints).
   */
  void scale_dual_in_place(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() /
                              delta.tail(delta.size() - dim).array() * c;
  }
  /*!
   * Scales a dual equality constrained variable in place.
   * @param dual dual variable (includes equalities constraints only).
   */
  void scale_dual_in_place_eq(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() /
                              delta.middleRows(dim, n_eq).array() * c;
  }
  /*!
   * Scales a dual inequality constrained variable in place.
   * @param dual dual variable (includes inequalities constraints only).
   */
  void scale_dual_in_place_in(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() /
                              delta.segment(dim + n_eq, n_in).array() * c;
  }
  /*!
   * Unscales a primal variable in place.
   * @param primal primal variable.
   */
  void unscale_primal_in_place(VectorViewMut<T> primal) const
  {
    primal.to_eigen().array() *= delta.array().head(dim);
  }
  /*!
   * Unscales a dual variable in place.
   * @param dual dual variable (includes equalities constraints only).
   */
  void unscale_dual_in_place(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() *
                              delta.tail(delta.size() - dim).array() / c;
  }
  /*!
   * Unscales a dual variable in place for box inequality constraints.
   * @param dual dual variable (includes equalities constraints only).
   */
  void unscale_box_dual_in_place_in(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() =
      delta.tail(dim).array() * dual.as_const().to_eigen().array() / c;
  }
  /*!
   * scales a dual variable in place for box inequality constraints.
   * @param dual dual variable (includes equalities constraints only).
   */
  void scale_box_dual_in_place_in(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() =
      dual.as_const().to_eigen().array() / delta.tail(dim).array() * c;
  }
  /*!
   * Unscales a dual equality constrained variable in place.
   * @param dual dual variable (includes equalities constraints only).
   */
  void unscale_dual_in_place_eq(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() *
                              delta.middleRows(dim, n_eq).array() / c;
  }
  /*!
   * Unscales a dual inequality constrained variable in place.
   * @param dual dual variable (includes inequalities constraints only).
   */
  void unscale_dual_in_place_in(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() *
                              delta.segment(dim + n_eq, n_in).array() / c;
  }
  // modifies residuals in place
  /*!
   * Scales a primal residual in place.
   * @param primal primal residual (includes equality and inequality
   * constraints)
   */
  void scale_primal_residual_in_place(VectorViewMut<T> primal) const
  {
    primal.to_eigen().array() *= delta.tail(delta.size() - dim).array();
  }

  /*!
   * Scales a primal equality constraint residual in place.
   * @param primal primal equality constraint residual.
   */
  void scale_primal_residual_in_place_eq(VectorViewMut<T> primal_eq) const
  {
    primal_eq.to_eigen().array() *= delta.middleRows(dim, n_eq).array();
  }
  /*!
   * Scales a primal inequality constraint residual in place.
   * @param primal primal inequality constraint residual.
   */
  void scale_primal_residual_in_place_in(VectorViewMut<T> primal_in) const
  {
    primal_in.to_eigen().array() *= delta.segment(dim + n_eq, n_in).array();
  }
  /*!
   * Scales a primal box inequality constraint residual in place.
   * @param primal primal inequality constraint residual.
   */
  void scale_box_primal_residual_in_place_in(VectorViewMut<T> primal_in) const
  {
    primal_in.to_eigen().array() *= delta.tail(dim).array();
  }
  /*!
   * Scales a dual residual in place.
   * @param dual dual residual.
   */
  void scale_dual_residual_in_place(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() *= delta.head(dim).array() * c;
  }
  /*!
   * Unscales a primal residual in place.
   * @param primal primal residual (includes equality and inequality
   * constraints).
   */
  void unscale_primal_residual_in_place(VectorViewMut<T> primal) const
  {
    primal.to_eigen().array() /= delta.tail(delta.size() - dim).array();
  }
  /*!
   * Unscales a primal equality constraint residual in place.
   * @param primal primal equality constraint residual.
   */
  void unscale_box_primal_residual_in_place(VectorViewMut<T> primal) const
  {
    primal.to_eigen().array() /= delta.tail(dim).array();
  }
  /*!
   * Unscales a primal equality constraint residual in place.
   * @param primal primal equality constraint residual.
   */
  void unscale_primal_residual_in_place_eq(VectorViewMut<T> primal_eq) const
  {
    primal_eq.to_eigen().array() /= delta.middleRows(dim, n_eq).array();
  }
  /*!
   * Unscales a primal inequality constraint residual in place.
   * @param primal primal inequality constraint residual.
   */
  void unscale_primal_residual_in_place_in(VectorViewMut<T> primal_in) const
  {
    primal_in.to_eigen().array() /= delta.middleRows(dim + n_eq, n_in).array();
  }
  /*!
   * Unscales a primal inequality constraint residual in place.
   * @param primal primal inequality constraint residual.
   */
  void unscale_box_primal_residual_in_place_in(VectorViewMut<T> primal_in) const
  {
    primal_in.to_eigen().array() /= delta.tail(dim).array();
  }
  /*!
   * Unscales a dual residual in place.
   * @param dual dual residual.
   */
  void unscale_dual_residual_in_place(VectorViewMut<T> dual) const
  {
    dual.to_eigen().array() /= delta.head(dim).array() * c;
  }
};

template<typename T>
bool
operator==(const RuizEquilibration<T>& ruiz1, const RuizEquilibration<T>& ruiz2)
{
  bool value =
    // ruiz1.delta == ruiz2.delta &&
    ruiz1.c == ruiz2.c
    // ruiz1.dim == ruiz2.dim
    // ruiz1.epsilon == ruiz2.epsilon &&
    // ruiz1.max_iter == ruiz2.max_iter &&
    // ruiz1.sym == ruiz2.sym &&
    // ruiz1.logger_ptr == ruiz2.logger_ptr
    ;
  return value;
}

template<typename T>
bool
operator!=(const RuizEquilibration<T>& ruiz1, const RuizEquilibration<T>& ruiz2)
{
  return !(ruiz1 == ruiz2);
}

} // namespace preconditioner
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_PRECOND_RUIZ_HPP */
