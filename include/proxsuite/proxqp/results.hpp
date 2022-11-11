//
// Copyright (c) 2022 INRIA
//
/**
 * @file results.hpp
 */
#ifndef PROXSUITE_PROXQP_RESULTS_HPP
#define PROXSUITE_PROXQP_RESULTS_HPP

#include <proxsuite/helpers/optional.hpp>
#include <Eigen/Core>
#include <proxsuite/linalg/veg/type_traits/core.hpp>
#include <proxsuite/linalg/veg/vec.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include "proxsuite/proxqp/status.hpp"
#include "proxsuite/proxqp/sparse/fwd.hpp"

namespace proxsuite {
namespace proxqp {
///
/// @brief This class stores the results statistics of PROXQP solvers with
/// sparse and dense backends.
///
/*!
 * Info class of dense and sparse solver.
 */
template<typename T>
struct Info
{
  ///// final proximal regularization parameters
  T mu_eq;
  T mu_eq_inv;
  T mu_in;
  T mu_in_inv;
  T rho;
  T nu;

  ///// iteration count
  sparse::isize iter;
  sparse::isize iter_ext;
  sparse::isize mu_updates;
  sparse::isize rho_updates;
  QPSolverOutput status;

  //// timings
  T setup_time;
  T solve_time;
  T run_time;
  T objValue;
  T pri_res;
  T dua_res;
  T duality_gap;
  //// sparse backend used by solver, either CholeskySparse or MatrixFree
  SparseBackend sparse_backend;
};
///
/// @brief This class stores all the results of PROXQP solvers with sparse and
/// dense backends.
///
/*!
 * Results class of dense and sparse solver.
 */
template<typename T>
struct Results
{

  ///// SOLUTION STORAGE

  sparse::Vec<T> x;
  sparse::Vec<T> y;
  sparse::Vec<T> z;
  proxsuite::linalg::veg::Vec<bool> active_constraints;

  Info<T> info;

  ////// SOLUTION STATUS
  /*!
   * Default constructor.
   * @param dim dimension of the primal variable.
   * @param n_eq dimension of the number of equality constraints.
   * @param n_in dimension of the number of inequality constraints.
   */
  Results(isize dim = 0, isize n_eq = 0, isize n_in = 0)
    : x(dim)
    , y(n_eq)
    , z(n_in)
  {

    x.setZero();
    y.setZero();
    z.setZero();

    info.rho = 1e-6;
    info.mu_eq_inv = 1e3;
    info.mu_eq = 1e-3;
    info.mu_in_inv = 1e1;
    info.mu_in = 1e-1;
    info.nu = 1.;
    info.iter = 0;
    info.iter_ext = 0;
    info.mu_updates = 0;
    info.rho_updates = 0;
    info.run_time = 0;
    info.setup_time = 0;
    info.solve_time = 0;
    info.objValue = 0.;
    info.pri_res = 0.;
    info.dua_res = 0.;
    info.duality_gap = 0.;
    info.status = QPSolverOutput::PROXQP_NOT_RUN;
    info.sparse_backend = SparseBackend::Automatic;
  }
  /*!
   * cleanups the Result variables and set the info variables to their initial
   * values.
   */
  void cleanup(optional<Settings<T>> settings = nullopt)
  {
    x.setZero();
    y.setZero();
    z.setZero();
    cold_start(settings);
  }
  void cleanup_statistics()
  {
    info.run_time = 0;
    info.setup_time = 0;
    info.solve_time = 0;
    info.objValue = 0.;
    info.iter = 0;
    info.iter_ext = 0;
    info.mu_updates = 0;
    info.rho_updates = 0;
    info.pri_res = 0.;
    info.dua_res = 0.;
    info.duality_gap = 0.;
    info.status = QPSolverOutput::PROXQP_MAX_ITER_REACHED;
    info.sparse_backend = SparseBackend::Automatic;
  }
  void cold_start(optional<Settings<T>> settings = nullopt)
  {
    info.rho = 1e-6;
    info.mu_eq_inv = 1e3;
    info.mu_eq = 1e-3;
    info.mu_in_inv = 1e1;
    info.mu_in = 1e-1;
    info.nu = 1.;
    if (settings != nullopt) {
      info.rho = settings.value().default_rho;
      info.mu_eq = settings.value().default_mu_eq;
      info.mu_eq_inv = T(1) / info.mu_eq;
      info.mu_in = settings.value().default_mu_in;
      info.mu_in_inv = T(1) / info.mu_in;
    }
    cleanup_statistics();
  }
  void cleanup_all_except_prox_parameters()
  {
    x.setZero();
    y.setZero();
    z.setZero();
    cleanup_statistics();
  }
};

} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_RESULTS_HPP */
