//
// Copyright (c) 2022 INRIA
//
/**
 * @file model.hpp
 */

#ifndef PROXSUITE_SERIALIZATION_WORKSPACE_HPP
#define PROXSUITE_SERIALIZATION_WORKSPACE_HPP

#include <cereal/cereal.hpp>
#include <proxsuite/proxqp/dense/workspace.hpp>

namespace cereal {

template<class Archive, typename T>
void
serialize(Archive& archive, proxsuite::proxqp::dense::Workspace<T>& work)
{
  archive(
    // CEREAL_NVP(work.ldl),
    // CEREAL_NVP(work.ldl_stack),
    CEREAL_NVP(work.H_scaled),
    CEREAL_NVP(work.g_scaled),
    CEREAL_NVP(work.A_scaled),
    CEREAL_NVP(work.b_scaled),
    CEREAL_NVP(work.C_scaled),
    CEREAL_NVP(work.l_scaled),
    CEREAL_NVP(work.u_scaled),
    CEREAL_NVP(work.x_prev),
    CEREAL_NVP(work.y_prev),
    CEREAL_NVP(work.z_prev),
    CEREAL_NVP(work.kkt),
    CEREAL_NVP(work.current_bijection_map),
    CEREAL_NVP(work.new_bijection_map),
    CEREAL_NVP(work.active_set_up),
    CEREAL_NVP(work.active_set_low),
    CEREAL_NVP(work.active_inequalities),
    CEREAL_NVP(work.Hdx),
    CEREAL_NVP(work.Cdx),
    CEREAL_NVP(work.Adx),
    CEREAL_NVP(work.active_part_z),
    CEREAL_NVP(work.alphas),
    CEREAL_NVP(work.dw_aug),
    CEREAL_NVP(work.rhs),
    CEREAL_NVP(work.err),

    CEREAL_NVP(work.dual_feasibility_rhs_2),
    CEREAL_NVP(work.correction_guess_rhs_g),
    CEREAL_NVP(work.correction_guess_rhs_b),
    CEREAL_NVP(work.alpha),

    CEREAL_NVP(work.dual_residual_scaled),
    CEREAL_NVP(work.primal_residual_in_scaled_up),
    CEREAL_NVP(work.primal_residual_in_scaled_up_plus_alphaCdx),
    CEREAL_NVP(work.primal_residual_in_scaled_low_plus_alphaCdx),
    CEREAL_NVP(work.CTz),
    CEREAL_NVP(work.constraints_changed),
    CEREAL_NVP(work.dirty),
    CEREAL_NVP(work.refactorize),
    CEREAL_NVP(work.proximal_parameter_update),
    CEREAL_NVP(work.is_initialized),
    CEREAL_NVP(work.n_c));
}

template<typename T, class Archive>
void
save(Archive& ar, proxsuite::linalg::veg::Vec<T> const& vec_T)
{
  proxsuite::linalg::veg::isize len = vec_T.len();
  ar(CEREAL_NVP(len));
  for (proxsuite::linalg::veg::isize i = 0; i < len; i++)
    ar(vec_T[i]);
}

template<typename T, class Archive>
void
load(Archive& ar, proxsuite::linalg::veg::Vec<T>& vec_T)
{
  proxsuite::linalg::veg::isize len;
  ar(len);
  vec_T.reserve(len);
  for (proxsuite::linalg::veg::isize i = 0; i < len; i++)
    ar(vec_T[i]);
}

} // namespace cereal

#endif /* end of include guard PROXSUITE_SERIALIZATION_WORKSPACE_HPP */
