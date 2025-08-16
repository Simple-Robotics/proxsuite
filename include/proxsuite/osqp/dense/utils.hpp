//
// Copyright (c) 2022-2024 INRIA
//
/**
 * @file utils.hpp
 */
#ifndef PROXSUITE_OSQP_DENSE_UTILS_HPP
#define PROXSUITE_OSQP_DENSE_UTILS_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <type_traits>

#include "proxsuite/common/status.hpp"
#include "proxsuite/helpers/common.hpp"
#include "proxsuite/common/dense/views.hpp"
#include "proxsuite/common/dense/workspace.hpp"
#include <proxsuite/proxqp/dense/model.hpp>
#include <proxsuite/common/results.hpp>
#include <proxsuite/common/settings.hpp>
#include <proxsuite/common/dense/preconditioner/ruiz.hpp>

namespace proxsuite {
namespace osqp {
namespace dense {

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::dense;

using proxsuite::common::DenseBackend;
using proxsuite::common::HessianType;
using proxsuite::common::InitialGuessStatus;
using proxsuite::common::Results;
using proxsuite::common::Settings;
using proxsuite::common::dense::Workspace;

template<typename T>
void
setup_factorization_complete_kkt(Results<T>& qpresults,
                                 const Model<T>& qpmodel,
                                 Workspace<T>& qpwork,
                                 const isize n_constraints,
                                 const DenseBackend& dense_backend)
{
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

  // Delete columns (from potential previous solve)
  if (qpwork.dirty == true) {
    auto _planned_to_delete = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, isize(n_constraints));
    isize* planned_to_delete = _planned_to_delete.ptr_mut();

    for (isize i = 0; i < n_constraints; i++) {
      planned_to_delete[i] = qpmodel.dim + qpmodel.n_eq + i;
    }

    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT: {
        qpwork.ldl.delete_at(planned_to_delete, n_constraints, stack);
      } break;
      case DenseBackend::PrimalLDLT:
        break;
      case DenseBackend::Automatic:
        break;
    }
  }

  // Add columns
  {
    T mu_in_neg(-qpresults.info.mu_in);
    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT: {
        isize n = qpmodel.dim;
        isize n_eq = qpmodel.n_eq;
        LDLT_TEMP_MAT_UNINIT(
          T, new_cols, n + n_eq + n_constraints, n_constraints, stack);

        for (isize k = 0; k < n_constraints; ++k) {
          auto col = new_cols.col(k);
          if (k >= qpmodel.n_in) {
            col.head(n).setZero();
            col[k - qpmodel.n_in] = qpwork.i_scaled[k - qpmodel.n_in];
          } else {
            col.head(n) = (qpwork.C_scaled.row(k));
          }
          col.tail(n_eq + n_constraints).setZero();
          col[n + n_eq + k] = mu_in_neg;
        }
        qpwork.ldl.insert_block_at(n + n_eq, new_cols, stack);
      } break;
      case DenseBackend::PrimalLDLT:
        break;
      case DenseBackend::Automatic:
        break;
    }
  }

  qpwork.n_c = n_constraints;
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_UTILS_HPP */
