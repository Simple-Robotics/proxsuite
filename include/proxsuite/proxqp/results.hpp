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
  T iterative_residual;
  //// sparse backend used by solver, either CholeskySparse or MatrixFree
  SparseBackend sparse_backend;
  //// quadratic cost minimal eigenvalue estimate
  T minimal_H_eigenvalue_estimate;
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
  sparse::Vec<T> se; // optimal shift to the closest feasible problem wrt
                     // equality constraints
  sparse::Vec<T> si; // optimal shift to the closest feasible problem wrt
                     // inequality constraints
  proxsuite::linalg::veg::Vec<bool> active_constraints;

  Info<T> info;

  ////// SOLUTION STATUS
  /*!
   * Default constructor.
   * @param dim dimension of the primal variable.
   * @param n_eq dimension of the number of equality constraints.
   * @param n_in dimension of the number of inequality constraints.
   */
  Results(isize dim = 0,
          isize n_eq = 0,
          isize n_in = 0,
          bool box_constraints = false,
          DenseBackend dense_backend = DenseBackend::PrimalDualLDLT)
    : x(dim)
    , y(n_eq)
    , z(n_in)
    , se(n_eq)
    , si(n_in)
  {
    if (box_constraints) {
      z.resize(dim + n_in);
      si.resize(dim + n_in);
    } else {
      z.resize(n_in);
      si.resize(n_in);
    }
    x.setZero();
    y.setZero();
    z.setZero();
    se.setZero();
    si.setZero();
    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT:
        info.rho = 1e-6;
        break;
      case DenseBackend::PrimalLDLT:
        info.rho = 1.E-5;
        break;
      case DenseBackend::Automatic:
        info.rho = 1.E-6;
        break;
    }
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
    info.iterative_residual = 0.;
    info.status = QPSolverOutput::PROXQP_NOT_RUN;
    info.sparse_backend = SparseBackend::Automatic;
    info.minimal_H_eigenvalue_estimate = 0.;
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
    se.setZero();
    si.setZero();
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
    info.iterative_residual = 0.;
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
    info.minimal_H_eigenvalue_estimate = 0.;
    if (settings != nullopt) {
      info.rho = settings.value().default_rho;
      info.mu_eq = settings.value().default_mu_eq;
      info.mu_eq_inv = T(1) / info.mu_eq;
      info.mu_in = settings.value().default_mu_in;
      info.mu_in_inv = T(1) / info.mu_in;
      info.minimal_H_eigenvalue_estimate =
        settings.value().default_H_eigenvalue_estimate;
    }
    cleanup_statistics();
  }
  void cleanup_all_except_prox_parameters()
  {
    x.setZero();
    y.setZero();
    z.setZero();
    se.setZero();
    si.setZero();
    cleanup_statistics();
  }
};

template<typename T>
bool
operator==(const Info<T>& info1, const Info<T>& info2)
{
  bool value =
    info1.mu_eq == info2.mu_eq && info1.mu_eq_inv == info2.mu_eq_inv &&
    info1.mu_in == info2.mu_in && info1.mu_in_inv == info2.mu_in_inv &&
    info1.rho == info2.rho && info1.nu == info2.nu &&
    info1.iter == info2.iter && info1.iter_ext == info2.iter_ext &&
    info1.mu_updates == info2.mu_updates &&
    info1.rho_updates == info2.rho_updates && info1.status == info2.status &&
    info1.setup_time == info2.setup_time &&
    info1.solve_time == info2.solve_time && info1.run_time == info2.run_time &&
    info1.objValue == info2.objValue && info1.pri_res == info2.pri_res &&
    info1.dua_res == info2.dua_res && info1.duality_gap == info2.duality_gap &&
    info1.duality_gap == info2.duality_gap &&
    info1.minimal_H_eigenvalue_estimate == info2.minimal_H_eigenvalue_estimate;
  return value;
}

template<typename T>
bool
operator!=(const Info<T>& info1, const Info<T>& info2)
{
  return !(info1 == info2);
}

template<typename T>
bool
operator==(const Results<T>& results1, const Results<T>& results2)
{
  bool value = results1.x == results2.x && results1.y == results2.y &&
               results1.z == results2.z && results1.info == results2.info;
  return value;
}

template<typename T>
bool
operator!=(const Results<T>& results1, const Results<T>& results2)
{
  return !(results1 == results2);
}

} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_RESULTS_HPP */
