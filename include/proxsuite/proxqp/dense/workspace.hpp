//
// Copyright (c) 2022 INRIA
//
/**
 * @file workspace.hpp
 */
#ifndef PROXSUITE_PROXQP_DENSE_WORKSPACE_HPP
#define PROXSUITE_PROXQP_DENSE_WORKSPACE_HPP

#include <Eigen/Core>
#include <proxsuite/linalg/dense/ldlt.hpp>
#include <proxsuite/proxqp/timings.hpp>
#include <proxsuite/linalg/veg/vec.hpp>
#include <proxsuite/proxqp/settings.hpp>
namespace proxsuite {
namespace proxqp {
namespace dense {
///
/// @brief This class defines the workspace of the dense solver.
///
/*!
 * Workspace class of the dense solver.
 */
template<typename T>
struct Workspace
{

  ///// Cholesky Factorization
  proxsuite::linalg::dense::Ldlt<T> ldl{};
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  Timer<T> timer;

  ///// QP STORAGE
  Mat<T> H_scaled;
  Vec<T> g_scaled;
  Mat<T> A_scaled;
  Mat<T> C_scaled;
  Vec<T> b_scaled;
  Vec<T> u_scaled;
  Vec<T> l_scaled;

  Vec<T> u_box_scaled;
  Vec<T> l_box_scaled;
  Vec<T> i_scaled;

  ///// Initial variable loading

  Vec<T> x_prev;
  Vec<T> y_prev;
  Vec<T> z_prev;

  ///// KKT system storage
  Mat<T> kkt;

  //// Active set & permutation vector
  VecISize current_bijection_map;
  VecISize new_bijection_map;

  VecBool active_set_up;
  VecBool active_set_low;
  VecBool active_inequalities;

  //// First order residuals for line search

  Vec<T> Hdx;
  Vec<T> Cdx;
  Vec<T> Adx;

  Vec<T> active_part_z;
  proxsuite::linalg::veg::Vec<T> alphas;

  ///// Newton variables
  Vec<T> dw_aug;
  Vec<T> rhs;
  Vec<T> err;

  //// Relative residuals constants

  T dual_feasibility_rhs_2;
  T correction_guess_rhs_g;
  T correction_guess_rhs_b;
  T alpha;

  Vec<T> dual_residual_scaled;
  Vec<T> primal_residual_in_scaled_up;

  Vec<T> primal_residual_in_scaled_up_plus_alphaCdx;
  Vec<T> primal_residual_in_scaled_low_plus_alphaCdx;
  Vec<T> CTz;

  bool constraints_changed;
  bool dirty;
  bool refactorize;
  bool proximal_parameter_update;
  bool is_initialized;

  sparse::isize n_c; // final number of active inequalities
  /*!
   * Default constructor.
   * @param dim primal variable dimension.
   * @param n_eq number of equality constraints.
   * @param n_in number of inequality constraints.
   */
  Workspace(isize dim = 0,
            isize n_eq = 0,
            isize n_in = 0,
            bool box_constraints = false,
            DenseBackend dense_backend = DenseBackend::PrimalDualLDLT)
    : ldl{}
    , H_scaled(dim, dim)
    , g_scaled(dim)
    , A_scaled(n_eq, dim)
    , C_scaled(n_in, dim)
    , b_scaled(n_eq)
    , u_scaled(n_in)
    , l_scaled(n_in)
    , x_prev(dim)
    , y_prev(n_eq)
    , Hdx(dim)
    , Adx(n_eq)
    , dual_residual_scaled(dim)
    , CTz(dim)
    , constraints_changed(false)
    , dirty(false)
    , refactorize(false)
    , proximal_parameter_update(false)
    , is_initialized(false)
  {

    if (box_constraints) {
      u_box_scaled.resize(dim);
      u_box_scaled.setZero();
      l_box_scaled.resize(dim);
      l_box_scaled.setZero();
      i_scaled.resize(dim);
      i_scaled.setOnes();

      z_prev.resize(dim + n_in);
      // TODO appropriate heuristic for automatic choice
      switch (dense_backend) {
        case DenseBackend::PrimalDualLDLT:
          kkt.resize(dim + n_eq, dim + n_eq);
          ldl.reserve_uninit(dim + n_eq + n_in + dim);
          ldl_stack.resize_for_overwrite(
            proxsuite::linalg::veg::dynstack::StackReq(
              // optimize here
              proxsuite::linalg::dense::Ldlt<T>::factorize_req(dim + n_eq +
                                                               n_in + dim) |

              (proxsuite::linalg::dense::temp_vec_req(
                 proxsuite::linalg::veg::Tag<T>{}, n_eq + n_in + dim) &
               proxsuite::linalg::veg::dynstack::StackReq{
                 isize{ sizeof(isize) } * (n_eq + n_in + dim),
                 alignof(isize) } &
               proxsuite::linalg::dense::Ldlt<T>::diagonal_update_req(
                 dim + n_eq + n_in + dim, n_eq + n_in + dim)) |

              (proxsuite::linalg::dense::temp_mat_req(
                 proxsuite::linalg::veg::Tag<T>{},
                 dim + n_eq + n_in + dim,
                 n_in + dim) &
               proxsuite::linalg::dense::Ldlt<T>::insert_block_at_req(
                 dim + n_eq + n_in + dim, n_in + dim)) |

              proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(dim + n_eq +
                                                                    n_in + dim))
              // TODO optimize here
              .alloc_req());
          break;
        case DenseBackend::PrimalLDLT:
          kkt.resize(dim, dim);
          ldl.reserve_uninit(dim);
          ldl_stack.resize_for_overwrite(
            proxsuite::linalg::veg::dynstack::StackReq(

              proxsuite::linalg::dense::Ldlt<T>::factorize_req(dim) |
              // check simplification possible
              (proxsuite::linalg::dense::temp_vec_req(
                 proxsuite::linalg::veg::Tag<T>{}, n_eq + n_in + dim) &
               proxsuite::linalg::veg::dynstack::StackReq{
                 isize{ sizeof(isize) } * (n_eq + n_in + dim),
                 alignof(isize) } &
               proxsuite::linalg::dense::Ldlt<T>::diagonal_update_req(
                 dim + n_eq + n_in + dim, n_eq + n_in + dim)) |

              (proxsuite::linalg::dense::temp_mat_req(
                 proxsuite::linalg::veg::Tag<T>{},
                 dim + n_eq + n_in + dim,
                 n_in + dim) &
               proxsuite::linalg::dense::Ldlt<T>::insert_block_at_req(
                 dim + n_eq + n_in + dim, n_in + dim)) |
              // end check
              proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(dim))

              .alloc_req());
          break;
        case DenseBackend::Automatic:
          break;
      }
      current_bijection_map.resize(n_in + dim);
      new_bijection_map.resize(n_in + dim);
      for (isize i = 0; i < n_in + dim; i++) {
        current_bijection_map(i) = i;
        new_bijection_map(i) = i;
      }
      active_set_up.resize(n_in + dim);
      active_set_low.resize(n_in + dim);
      active_inequalities.resize(n_in + dim);
      active_part_z.resize(n_in + dim);
      dw_aug.resize(dim + n_eq + n_in + dim);
      rhs.resize(dim + n_eq + n_in + dim);
      err.resize(dim + n_eq + n_in + dim);
      primal_residual_in_scaled_up.resize(dim + n_in);
      primal_residual_in_scaled_up_plus_alphaCdx.resize(dim + n_in);
      primal_residual_in_scaled_low_plus_alphaCdx.resize(dim + n_in);
      Cdx.resize(n_in + dim);
      alphas.reserve(2 * n_in + 2 * dim);
    } else {
      z_prev.resize(n_in);

      switch (dense_backend) {
        case DenseBackend::PrimalDualLDLT:
          kkt.resize(dim + n_eq, dim + n_eq);
          ldl.reserve_uninit(dim + n_eq + n_in);
          ldl_stack.resize_for_overwrite(
            proxsuite::linalg::veg::dynstack::StackReq(
              // todo optimize here
              proxsuite::linalg::dense::Ldlt<T>::factorize_req(dim + n_eq +
                                                               n_in) |

              (proxsuite::linalg::dense::temp_vec_req(
                 proxsuite::linalg::veg::Tag<T>{}, n_eq + n_in) &
               proxsuite::linalg::veg::dynstack::StackReq{
                 isize{ sizeof(isize) } * (n_eq + n_in), alignof(isize) } &
               proxsuite::linalg::dense::Ldlt<T>::diagonal_update_req(
                 dim + n_eq + n_in, n_eq + n_in)) |

              (proxsuite::linalg::dense::temp_mat_req(
                 proxsuite::linalg::veg::Tag<T>{}, dim + n_eq + n_in, n_in) &
               proxsuite::linalg::dense::Ldlt<T>::insert_block_at_req(
                 dim + n_eq + n_in, n_in)) |

              proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(dim + n_eq +
                                                                    n_in))
              // end todo optimize here
              .alloc_req());
          break;
        case DenseBackend::PrimalLDLT:
          kkt.resize(dim, dim);
          ldl.reserve_uninit(dim);
          ldl_stack.resize_for_overwrite(
            proxsuite::linalg::veg::dynstack::StackReq(

              proxsuite::linalg::dense::Ldlt<T>::factorize_req(dim) |
              // check if it can be more simplified
              (proxsuite::linalg::dense::temp_vec_req(
                 proxsuite::linalg::veg::Tag<T>{}, n_eq + n_in) &
               proxsuite::linalg::veg::dynstack::StackReq{
                 isize{ sizeof(isize) } * (n_eq + n_in), alignof(isize) } &
               proxsuite::linalg::dense::Ldlt<T>::diagonal_update_req(
                 dim + n_eq + n_in, n_eq + n_in)) |
              (proxsuite::linalg::dense::temp_mat_req(
                 proxsuite::linalg::veg::Tag<T>{}, dim + n_eq + n_in, n_in) &
               proxsuite::linalg::dense::Ldlt<T>::insert_block_at_req(
                 dim + n_eq + n_in, n_in)) |
              // end check
              proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(dim))

              .alloc_req());
          break;
        case DenseBackend::Automatic:
          break;
      }

      current_bijection_map.resize(n_in);
      new_bijection_map.resize(n_in);
      for (isize i = 0; i < n_in; i++) {
        current_bijection_map(i) = i;
        new_bijection_map(i) = i;
      }
      active_set_up.resize(n_in);
      active_set_low.resize(n_in);
      active_inequalities.resize(n_in);
      active_part_z.resize(n_in);
      dw_aug.resize(dim + n_eq + n_in);
      rhs.resize(dim + n_eq + n_in);
      err.resize(dim + n_eq + n_in);
      primal_residual_in_scaled_up.resize(n_in);
      primal_residual_in_scaled_up_plus_alphaCdx.resize(n_in);
      primal_residual_in_scaled_low_plus_alphaCdx.resize(n_in);
      Cdx.resize(n_in);
      alphas.reserve(2 * n_in);
    }

    H_scaled.setZero();
    g_scaled.setZero();
    A_scaled.setZero();
    C_scaled.setZero();
    b_scaled.setZero();
    u_scaled.setZero();
    l_scaled.setZero();
    x_prev.setZero();
    y_prev.setZero();
    z_prev.setZero();
    kkt.setZero();
    Hdx.setZero();
    Cdx.setZero();
    Adx.setZero();
    active_part_z.setZero();
    dw_aug.setZero();
    rhs.setZero();
    err.setZero();

    dual_feasibility_rhs_2 = 0;
    correction_guess_rhs_g = 0;
    correction_guess_rhs_b = 0;
    alpha = 1.;

    dual_residual_scaled.setZero();
    primal_residual_in_scaled_up.setZero();

    primal_residual_in_scaled_up_plus_alphaCdx.setZero();
    primal_residual_in_scaled_low_plus_alphaCdx.setZero();
    CTz.setZero();
    n_c = 0;
  }
  /*!
   * Clean-ups solver's workspace.
   */
  void cleanup(const bool box_constraints)
  {
    isize n_in = C_scaled.rows();
    isize dim = H_scaled.rows();
    H_scaled.setZero();
    g_scaled.setZero();
    A_scaled.setZero();
    C_scaled.setZero();
    b_scaled.setZero();
    u_scaled.setZero();
    l_scaled.setZero();
    Hdx.setZero();
    Cdx.setZero();
    Adx.setZero();
    active_part_z.setZero();
    dw_aug.setZero();
    rhs.setZero();
    err.setZero();

    alpha = 1.;

    dual_residual_scaled.setZero();
    primal_residual_in_scaled_up.setZero();

    primal_residual_in_scaled_up_plus_alphaCdx.setZero();
    primal_residual_in_scaled_low_plus_alphaCdx.setZero();
    CTz.setZero();

    x_prev.setZero();
    y_prev.setZero();
    z_prev.setZero();
    isize n_constraints(n_in);
    if (box_constraints) {
      n_constraints += dim;
    }
    for (isize i = 0; i < n_constraints; i++) {
      current_bijection_map(i) = i;
      new_bijection_map(i) = i;
      active_inequalities(i) = false;
    }

    constraints_changed = false;
    dirty = false;
    refactorize = false;
    proximal_parameter_update = false;
    is_initialized = false;
    n_c = 0;
  }
};
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_WORKSPACE_HPP */
