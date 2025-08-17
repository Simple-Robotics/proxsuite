//
// Copyright (c) 2025 INRIA
//
/**
 * @file aliases.hpp
 */

#ifndef PROXSUITE_PROXQP_SPARSE_ALIASES_HPP
#define PROXSUITE_PROXQP_SPARSE_ALIASES_HPP

#include "proxsuite/common/solvers.hpp"
#include "proxsuite/common/status.hpp"
#include "proxsuite/common/settings.hpp"
#include "proxsuite/common/results.hpp"
#include "proxsuite/common/dense/views.hpp"
#include "proxsuite/common/timings.hpp"

namespace proxsuite {
namespace proxqp {
namespace sparse {

using proxsuite::common::from_eigen;
using proxsuite::common::isize;

using proxsuite::common::HessianType;
using proxsuite::common::InitialGuessStatus;
using proxsuite::common::MeritFunctionType;
using proxsuite::common::PreconditionerStatus;
using proxsuite::common::QPSolver;
using proxsuite::common::QPSolverOutput;
using proxsuite::common::SparseBackend;
using proxsuite::common::Timer;

using proxsuite::common::Results;
using proxsuite::common::Settings;

using proxsuite::common::VectorView;
using proxsuite::common::VectorViewMut;

} // namespace sparse
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_SPARSE_ALIASES_HPP */
