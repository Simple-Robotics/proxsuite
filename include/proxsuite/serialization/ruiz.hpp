//
// Copyright (c) 2022 INRIA
//
/**
 * @file ruiz.hpp
 */

#ifndef PROXSUITE_SERIALIZATION_RUIZ_HPP
#define PROXSUITE_SERIALIZATION_RUIZ_HPP

#include <cereal/cereal.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>

namespace cereal {

template<class Archive, typename T>
void
serialize(Archive& archive,
          proxsuite::proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz)
{
  archive(
    // CEREAL_NVP(ruiz.delta),
    CEREAL_NVP(ruiz.c)
    // CEREAL_NVP(ruiz.dim)
    // CEREAL_NVP(ruiz.epsilon),
    // CEREAL_NVP(ruiz.max_iter)
    // CEREAL_NVP(ruiz.sym)
  );
}
} // namespace cereal

#endif /* end of include guard PROXSUITE_SERIALIZATION_RUIZ_HPP */
