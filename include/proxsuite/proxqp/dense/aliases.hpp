//
// Copyright (c) 2025 INRIA
//
/**
 * @file aliases.hpp
 */

#ifndef PROXSUITE_PROXQP_DENSE_ALIASES_HPP
#define PROXSUITE_PROXQP_DENSE_ALIASES_HPP

#include "proxsuite/common/status.hpp"
#include "proxsuite/common/settings.hpp"
#include "proxsuite/common/results.hpp"
#include "proxsuite/common/dense/views.hpp"
#include "proxsuite/common/dense/model.hpp"
#include "proxsuite/common/dense/workspace.hpp"

namespace proxsuite {
namespace proxqp {
namespace dense {

using proxsuite::common::from_eigen;
using proxsuite::common::i32;
using proxsuite::common::i64;
using proxsuite::common::isize;
using proxsuite::common::dense::infty_norm;

using proxsuite::common::DenseBackend;
using proxsuite::common::HessianType;
using proxsuite::common::InitialGuessStatus;
using proxsuite::common::MeritFunctionType;
using proxsuite::common::PolishStatus;
using proxsuite::common::PreconditionerStatus;
using proxsuite::common::QPSolverOutput;
using proxsuite::common::SparseBackend;
using proxsuite::common::Timer;

using proxsuite::common::Results;
using proxsuite::common::Settings;
using proxsuite::common::dense::Model;
using proxsuite::common::dense::Workspace;

using proxsuite::common::VectorView;
using proxsuite::common::VectorViewMut;
using proxsuite::common::dense::Mat;
using proxsuite::common::dense::MatRef;
using proxsuite::common::dense::Vec;
using proxsuite::common::dense::VecRef;

} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_ALIASES_HPP */
