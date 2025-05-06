//
// Copyright (c) 2025 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_SOLVER_HPP
#define PROXSUITE_OSQP_DENSE_SOLVER_HPP

#include <proxsuite/linalg/dense/ldlt.hpp>
#include <proxsuite/linalg/veg/internal/typedefs.hpp>
#include "proxsuite/proxqp/dense/solver.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
#include "proxsuite/proxqp/dense/preconditioner/ruiz.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/results.hpp"
#include <iostream>

namespace proxsuite {
namespace osqp {
namespace dense {

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::dense;

/*!
 * Executes the OSQP algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 */
template<typename T>
void
qp_solve( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const bool box_constraints,
  const DenseBackend& dense_backend,
  const HessianType& hessian_type,
  preconditioner::RuizEquilibration<T>& ruiz)
{
  PROXSUITE_EIGEN_MALLOC_NOT_ALLOWED();

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */