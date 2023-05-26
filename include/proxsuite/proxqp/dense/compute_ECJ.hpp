//
// Copyright (c) 2023 INRIA
//
/**
 * @file compute_ECJ.hpp
 */

#ifndef PROXSUITE_PROXQP_DENSE_COMPUTE_ECJ_HPP
#define PROXSUITE_PROXQP_DENSE_COMPUTE_ECJ_HPP
#include <proxsuite/proxqp/dense/wrapper.hpp>

namespace proxsuite {
namespace proxqp {
namespace dense {


template<typename T>
void
compute_backward(dense::QP<T>& solved_qp,
                 Vec<T>& loss_derivative,
                 T eps = 1.E-4)
{
  bool check =
    solved_qp.results.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
  if (check) {
    PROXSUITE_THROW_PRETTY(
      true,
      std::invalid_argument,
      "the QP problem is not feasible, so computing the derivatives "
      "is not valid in this setting. Try enabling infeasible solving if "
      "the problem is only primally infeasible.");
  } else {
    solved_qp.model.backward_data.initialize(
      solved_qp.model.dim, solved_qp.model.n_eq, solved_qp.model.n_in);

    /// derive solution
    solved_qp.work.CTz.noalias() =
      solved_qp.model.C * solved_qp.results.x + solved_qp.results.z;
    solved_qp.work.active_set_up.array() =
      (solved_qp.work.CTz - solved_qp.model.u).array() >= 0.;
    solved_qp.work.active_set_low.array() =
      (solved_qp.work.CTz - solved_qp.model.l).array() <= 0.;
    solved_qp.work.active_inequalities =
      solved_qp.work.active_set_up || solved_qp.work.active_set_low;
    isize numactive_inequalities = solved_qp.work.active_inequalities.count();
    isize inner_pb_dim =
      solved_qp.model.dim + solved_qp.model.n_eq + numactive_inequalities;
    solved_qp.work.rhs.setZero();
    // work.dw_aug.setZero(); zeroed in active_set_change
    T rho_new = 1.e-3;
    solved_qp.results.info.mu_eq = 5.E-5;
    solved_qp.results.info.mu_in = 5.E-5;
    solved_qp.results.info.rho = rho_new;

    // a large amount of constraints might have changed
    // so in order to avoid to much refactorization later in the
    // iterative refinement, a factorization from scratch is directly
    // performed with new mu and rho as well to enable more stability
    proxsuite::proxqp::dense::setup_factorization(
      solved_qp.work, solved_qp.model, solved_qp.results);
    solved_qp.work.n_c = 0;
    for (isize i = 0; i < solved_qp.model.n_in; i++) {
      solved_qp.work.current_bijection_map(i) = i;
      solved_qp.work.new_bijection_map(i) = i;
    }
    linesearch::active_set_change(
      solved_qp.model, solved_qp.results, solved_qp.work);
    solved_qp.work.constraints_changed = false; // no refactorization afterwords

    solved_qp.work.rhs = loss_derivative; // take full derivatives
    solved_qp.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{
      from_eigen, solved_qp.work.rhs.head(solved_qp.model.dim) });
    if (!solved_qp.work.rhs.segment(solved_qp.model.dim, solved_qp.model.n_eq)
           .isZero()) {
      solved_qp.work.rhs.segment(solved_qp.model.dim, solved_qp.model.n_eq) =
        loss_derivative.segment(solved_qp.model.dim, solved_qp.model.n_eq);
      solved_qp.ruiz.scale_primal_residual_in_place_eq(
        VectorViewMut<T>{ from_eigen,
                          solved_qp.work.rhs.segment(solved_qp.model.dim,
                                                     solved_qp.model.n_eq) });
    }
    if (!solved_qp.work.rhs.tail(solved_qp.model.n_in).isZero()) {
      for (isize i = 0; i < solved_qp.model.n_in; i++) {
        isize j = solved_qp.work.current_bijection_map(i);
        if (j < solved_qp.work.n_c) {
          solved_qp.work.rhs(j + solved_qp.model.dim + solved_qp.model.n_eq) =
            loss_derivative(i + solved_qp.model.dim + solved_qp.model.n_eq);
        }
        solved_qp.ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{
          from_eigen, solved_qp.work.rhs.tail(solved_qp.model.n_in) });
      }
    }
    iterative_solve_with_permut_fact( //
      solved_qp.settings,
      solved_qp.model,
      solved_qp.results,
      solved_qp.work,
      eps,
      inner_pb_dim); // /!\ the full rhs is zeroed inside
    compute_backward_ESG(solved_qp, loss_derivative);
  }
}

template<typename T>
void
compute_backward_loss_ESG(dense::QP<T>& solved_qp, Vec<T>& loss_derivative)
{
  // use active_part_z as a temporary variable to derive unpermutted dz step
  solved_qp.work.active_part_z.setZero();
  for (isize j = 0; j < solved_qp.model.n_in; ++j) {
    isize i = solved_qp.work.current_bijection_map(j);
    if (i < solved_qp.work.n_c) {
      solved_qp.work.active_part_z(j) =
        solved_qp.work.dw_aug(solved_qp.model.dim + solved_qp.model.n_eq + i);
    } else {
      solved_qp.work.active_part_z(j) =
        loss_derivative(solved_qp.model.dim + solved_qp.model.n_eq + i);
    }
  }
  solved_qp.work.dw_aug.tail(solved_qp.model.n_in) =
    solved_qp.work.active_part_z;
  solved_qp.ruiz.unscale_primal_in_place(VectorViewMut<T>{
    from_eigen, solved_qp.work.dw_aug.head(solved_qp.model.dim) });
  solved_qp.ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{
    from_eigen,
    solved_qp.work.dw_aug.segment(solved_qp.model.dim, solved_qp.model.n_eq) });
  solved_qp.ruiz.unscale_dual_in_place_in(VectorViewMut<T>{
    from_eigen, solved_qp.work.dw_aug.tail(solved_qp.model.n_in) });

  /// compute backward derivatives
  solved_qp.model.backward_data.dL_dC.noalias() =
    solved_qp.work.dw_aug.tail(solved_qp.model.n_in) *
    solved_qp.results.x.transpose();
  solved_qp.model.backward_data.dL_dC.noalias() +=
    solved_qp.results.z *
    solved_qp.work.dw_aug.head(solved_qp.model.dim).transpose();

  solved_qp.model.backward_data.dL_du =
    (solved_qp.work.active_set_up)
      .select(-solved_qp.work.dw_aug.tail(solved_qp.model.n_in),
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(solved_qp.model.n_in));

  solved_qp.model.backward_data.dL_dl =
    (solved_qp.work.active_set_low)
      .select(-solved_qp.work.dw_aug.tail(solved_qp.model.n_in),
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(solved_qp.model.n_in));

  solved_qp.model.backward_data.dL_dA.noalias() =
    solved_qp.work.dw_aug.segment(solved_qp.model.dim, solved_qp.model.n_eq) *
    solved_qp.results.x.transpose();
  solved_qp.model.backward_data.dL_dA.noalias() +=
    solved_qp.results.y *
    solved_qp.work.dw_aug.head(solved_qp.model.dim).transpose();

  solved_qp.model.backward_data.dL_db =
    -solved_qp.work.dw_aug.segment(solved_qp.model.dim, solved_qp.model.n_eq);
  solved_qp.model.backward_data.dL_dH.noalias() =
    0.5 * (solved_qp.work.dw_aug.head(solved_qp.model.dim) *
             solved_qp.results.x.transpose() +
           solved_qp.results.x *
             solved_qp.work.dw_aug.head(solved_qp.model.dim).transpose());

  solved_qp.model.backward_data.dL_dg =
    solved_qp.work.dw_aug.head(solved_qp.model.dim);
}

} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_COMPUTE_ECJ_HPP */
