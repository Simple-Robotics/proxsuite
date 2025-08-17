//
// Copyright (c) 2022-2025 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_COMMON_DENSE_WRAPPER_HPP
#define PROXSUITE_COMMON_DENSE_WRAPPER_HPP

#include "proxsuite/common/status.hpp"
#include "proxsuite/common/settings.hpp"
#include "proxsuite/common/results.hpp"
#include "proxsuite/common/dense/model.hpp"
#include "proxsuite/common/dense/workspace.hpp"
#include <proxsuite/common/dense/helpers.hpp>
#include <proxsuite/common/dense/preconditioner/ruiz.hpp>
#include <chrono>

namespace proxsuite {
namespace common {
namespace dense {

///// Dense backend choice
template<typename T>
DenseBackend
dense_backend_choice(DenseBackend _dense_backend,
                     isize dim,
                     isize n_eq,
                     isize n_in,
                     bool box_constraints)
{
  if (_dense_backend == DenseBackend::Automatic) {
    isize n_constraints(n_in);
    if (box_constraints) {
      n_constraints += dim;
    }
    T threshold(1.5);
    T frequence(0.2);
    T PrimalDualLDLTCost =
      0.5 * std::pow(T(n_eq) / T(dim), 2) +
      0.17 * (std::pow(T(n_eq) / T(dim), 3) +
              std::pow(T(n_constraints) / T(dim), 3)) +
      frequence * std::pow(T(n_eq + n_constraints) / T(dim), 2) / T(dim);
    T PrimalLDLTCost =
      threshold *
      ((0.5 * T(n_eq) + T(n_constraints)) / T(dim) + frequence / T(dim));
    bool choice = PrimalDualLDLTCost > PrimalLDLTCost;
    if (choice) {
      return DenseBackend::PrimalLDLT;
    } else {
      return DenseBackend::PrimalDualLDLT;
    }
  } else {
    return _dense_backend;
  }
}

///
/// @brief This class defines the base API of the solvers with dense backend.
///
/*!
 * Base CRTP class for QP solvers with dense backend.
 */
template<typename Derived, typename T>
struct QPBase
{
private:
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

protected:
  DenseBackend dense_backend;
  bool box_constraints;
  HessianType hessian_type;

public:
  Results<T> results;
  Settings<T> settings;
  Model<T> model;
  Workspace<T> work;
  preconditioner::RuizEquilibration<T> ruiz;

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   * @param _dense_backend specify which factorization is used.
   */
  QPBase(isize _dim,
         isize _n_eq,
         isize _n_in,
         bool _box_constraints,
         HessianType _hessian_type,
         DenseBackend _dense_backend)
    : dense_backend(dense_backend_choice<T>(_dense_backend,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            _box_constraints))
    , box_constraints(_box_constraints)
    , hessian_type(_hessian_type)
    , results(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, _box_constraints)
    , work(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim,
                                                 _n_eq,
                                                 _n_in,
                                                 _box_constraints })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   * @param _dense_backend specify which factorization is used.
   */
  QPBase(isize _dim,
         isize _n_eq,
         isize _n_in,
         bool _box_constraints,
         DenseBackend _dense_backend,
         HessianType _hessian_type)
    : dense_backend(dense_backend_choice<T>(_dense_backend,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            _box_constraints))
    , box_constraints(_box_constraints)
    , hessian_type(_hessian_type)
    , results(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, _box_constraints)
    , work(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim,
                                                 _n_eq,
                                                 _n_in,
                                                 _box_constraints })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QPBase(isize _dim,
         isize _n_eq,
         isize _n_in,
         bool _box_constraints,
         HessianType _hessian_type)
    : dense_backend(dense_backend_choice<T>(DenseBackend::Automatic,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            _box_constraints))
    , box_constraints(_box_constraints)
    , hessian_type(_hessian_type)
    , results(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, _box_constraints)
    , work(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim,
                                                 _n_eq,
                                                 _n_in,
                                                 _box_constraints })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   * @param _dense_backend specify which factorization is used.
   */
  QPBase(isize _dim,
         isize _n_eq,
         isize _n_in,
         bool _box_constraints,
         DenseBackend _dense_backend)
    : dense_backend(dense_backend_choice<T>(_dense_backend,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            _box_constraints))
    , box_constraints(_box_constraints)
    , hessian_type(HessianType::Dense)
    , results(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, _box_constraints)
    , work(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim,
                                                 _n_eq,
                                                 _n_in,
                                                 _box_constraints })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QPBase(isize _dim, isize _n_eq, isize _n_in, bool _box_constraints)
    : dense_backend(dense_backend_choice<T>(DenseBackend::Automatic,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            _box_constraints))
    , box_constraints(_box_constraints)
    , hessian_type(HessianType::Dense)
    , results(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, _box_constraints)
    , work(_dim, _n_eq, _n_in, _box_constraints, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim,
                                                 _n_eq,
                                                 _n_in,
                                                 _box_constraints })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type specify that there are (or not) box constraints.
   */
  QPBase(isize _dim, isize _n_eq, isize _n_in, HessianType _hessian_type)
    : dense_backend(dense_backend_choice<T>(DenseBackend::Automatic,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            false))
    , box_constraints(false)
    , hessian_type(_hessian_type)
    , results(_dim, _n_eq, _n_in, false, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, false)
    , work(_dim, _n_eq, _n_in, false, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim, _n_eq, _n_in, false })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   */
  QPBase(isize _dim, isize _n_eq, isize _n_in)
    : dense_backend(dense_backend_choice<T>(DenseBackend::Automatic,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            false))
    , box_constraints(false)
    , hessian_type(HessianType::Dense)
    , results(_dim, _n_eq, _n_in, false, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, false)
    , work(_dim, _n_eq, _n_in, false, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim, _n_eq, _n_in, false })
  {
    work.timer.stop();
    derived().init_derived_settings();
    derived().init_derived_results();
  }

  // Accessors
  bool is_box_constrained() const { return box_constraints; };
  DenseBackend which_dense_backend() const { return dense_backend; };
  HessianType which_hessian_type() const { return hessian_type; };

  /*!
   * Setups the QP model (with dense matrix format) and equilibrates it if
   * specified by the user.
   * @param H quadratic cost input defining the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param compute_preconditioner boolean parameter for executing or not the
   * preconditioner.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   * @param manual_minimal_H_eigenvalue manual minimal eigenvalue proposed for H
   */
  void init(optional<MatRef<T>> H,
            optional<VecRef<T>> g,
            optional<MatRef<T>> A,
            optional<VecRef<T>> b,
            optional<MatRef<T>> C,
            optional<VecRef<T>> l,
            optional<VecRef<T>> u,
            bool compute_preconditioner = true,
            optional<T> rho = nullopt,
            optional<T> mu_eq = nullopt,
            optional<T> mu_in = nullopt,
            optional<T> manual_minimal_H_eigenvalue = nullopt)
  {
    PROXSUITE_THROW_PRETTY(
      box_constraints == true,
      std::invalid_argument,
      "wrong model setup: the QP object is designed with box "
      "constraints, but is initialized without lower or upper box "
      "inequalities.");
    // dense case
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    settings.compute_preconditioner = compute_preconditioner;
    // check the model is valid
    if (g != nullopt && g.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        g.value().size(),
        model.dim,
        "the dimension wrt the primal variable x variable for initializing g "
        "is not valid.");
    } else {
      g.reset();
    }
    if (b != nullopt && b.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        b.value().size(),
        model.n_eq,
        "the dimension wrt equality constrained variables for initializing b "
        "is not valid.");
    } else {
      b.reset();
    }
    if (u != nullopt && u.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        u.value().size(),
        model.n_in,
        "the dimension wrt inequality constrained variables for initializing u "
        "is not valid.");
    } else {
      u.reset();
    }
    if (l != nullopt && l.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        l.value().size(),
        model.n_in,
        "the dimension wrt inequality constrained variables for initializing l "
        "is not valid.");
    } else {
      l.reset();
    }
    if (H != nullopt && H.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.value().rows(),
        model.dim,
        "the row dimension for initializing H is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.value().cols(),
        model.dim,
        "the column dimension for initializing H is not valid.");
    } else {
      H.reset();
    }
    if (A != nullopt && A.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.value().rows(),
        model.n_eq,
        "the row dimension for initializing A is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.value().cols(),
        model.dim,
        "the column dimension for initializing A is not valid.");
    } else {
      A.reset();
    }
    if (C != nullopt && C.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.value().rows(),
        model.n_in,
        "the row dimension for initializing C is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.value().cols(),
        model.dim,
        "the column dimension for initializing C is not valid.");
    } else {
      C.reset();
    }
    if (settings.initial_guess ==
        InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT) {
      work.refactorize =
        true; // necessary for the first solve (then refactorize only if there
              // is an update of the matrices)
    } else {
      work.refactorize = false;
    }
    work.proximal_parameter_update = false;
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (compute_preconditioner) {
      preconditioner_status = PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = PreconditionerStatus::IDENTITY;
    }
    common::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    common::dense::update_default_rho_with_minimal_Hessian_eigen_value(
      manual_minimal_H_eigenvalue, results, settings);
    typedef optional<VecRef<T>> optional_VecRef;
    common::dense::setup(H,
                         g,
                         A,
                         b,
                         C,
                         l,
                         u,
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         settings,
                         model,
                         work,
                         results,
                         box_constraints,
                         ruiz,
                         preconditioner_status,
                         hessian_type);
    work.is_initialized = true;
    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };

  /*!
   * Setups the QP model (with dense matrix format) and equilibrates it if
   * specified by the user.
   * @param H quadratic cost input defining the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param l_box lower box inequality constraint vector input defining the QP
   * model.
   * @param u_box uppper box inequality constraint vector input defining the QP
   * model.
   * @param compute_preconditioner boolean parameter for executing or not the
   * preconditioner.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   * @param manual_minimal_H_eigenvalue manual minimal eigenvalue proposed for H
   */
  void init(optional<MatRef<T>> H,
            optional<VecRef<T>> g,
            optional<MatRef<T>> A,
            optional<VecRef<T>> b,
            optional<MatRef<T>> C,
            optional<VecRef<T>> l,
            optional<VecRef<T>> u,
            optional<VecRef<T>> l_box,
            optional<VecRef<T>> u_box,
            bool compute_preconditioner = true,
            optional<T> rho = nullopt,
            optional<T> mu_eq = nullopt,
            optional<T> mu_in = nullopt,
            optional<T> manual_minimal_H_eigenvalue = nullopt)
  {

    // dense case
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    settings.compute_preconditioner = compute_preconditioner;
    PROXSUITE_THROW_PRETTY(
      box_constraints == false && (l_box != nullopt || u_box != nullopt),
      std::invalid_argument,
      "wrong model setup: the QP object is designed without box "
      "constraints, but is initialized with lower or upper box inequalities.");
    if (l_box != nullopt && l_box.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(l_box.value().size(),
                                    model.dim,
                                    "the dimension wrt the primal variable x "
                                    "variable for initializing l_box "
                                    "is not valid.");
    } else {
      l_box.reset();
    }
    if (u_box != nullopt && u_box.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(u_box.value().size(),
                                    model.dim,
                                    "the dimension wrt the primal variable x "
                                    "variable for initializing u_box "
                                    "is not valid.");
    } else {
      l_box.reset();
    }
    // check the model is valid
    if (g != nullopt && g.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        g.value().size(),
        model.dim,
        "the dimension wrt the primal variable x variable for initializing g "
        "is not valid.");
    } else {
      g.reset();
    }
    if (b != nullopt && b.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        b.value().size(),
        model.n_eq,
        "the dimension wrt equality constrained variables for initializing b "
        "is not valid.");
    } else {
      b.reset();
    }
    if (u != nullopt && u.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        u.value().size(),
        model.n_in,
        "the dimension wrt inequality constrained variables for initializing u "
        "is not valid.");
    } else {
      u.reset();
    }
    if (u_box != nullopt && u_box.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        u_box.value().size(),
        model.dim,
        "the dimension wrt box inequality constrained variables for "
        "initializing u_box "
        "is not valid.");
    } else {
      u_box.reset();
    }
    if (l != nullopt && l.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        l.value().size(),
        model.n_in,
        "the dimension wrt inequality constrained variables for initializing l "
        "is not valid.");
    } else {
      l.reset();
    }
    if (l_box != nullopt && l_box.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        l_box.value().size(),
        model.dim,
        "the dimension wrt box inequality constrained variables for "
        "initializing l_box "
        "is not valid.");
    } else {
      l_box.reset();
    }
    if (H != nullopt && H.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.value().rows(),
        model.dim,
        "the row dimension for initializing H is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        H.value().cols(),
        model.dim,
        "the column dimension for initializing H is not valid.");
    } else {
      H.reset();
    }
    if (A != nullopt && A.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.value().rows(),
        model.n_eq,
        "the row dimension for initializing A is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        A.value().cols(),
        model.dim,
        "the column dimension for initializing A is not valid.");
    } else {
      A.reset();
    }
    if (C != nullopt && C.value().size() != 0) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.value().rows(),
        model.n_in,
        "the row dimension for initializing C is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        C.value().cols(),
        model.dim,
        "the column dimension for initializing C is not valid.");
    } else {
      C.reset();
    }
    if (settings.initial_guess ==
        InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT) {
      work.refactorize =
        true; // necessary for the first solve (then refactorize only if there
              // is an update of the matrices)
    } else {
      work.refactorize = false;
    }
    work.proximal_parameter_update = false;
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (compute_preconditioner) {
      preconditioner_status = PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = PreconditionerStatus::IDENTITY;
    }
    common::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    common::dense::update_default_rho_with_minimal_Hessian_eigen_value(
      manual_minimal_H_eigenvalue, results, settings);
    common::dense::setup(H,
                         g,
                         A,
                         b,
                         C,
                         l,
                         u,
                         l_box,
                         u_box,
                         settings,
                         model,
                         work,
                         results,
                         box_constraints,
                         ruiz,
                         preconditioner_status,
                         hessian_type);
    work.is_initialized = true;
    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };

  /*!
   * Updates the QP model (with dense matrix format) and re-equilibrates it if
   * specified by the user.
   * @param H quadratic cost input defining the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param update_preconditioner bool parameter for updating or not the
   * preconditioner and the associated scaled model.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   * @param manual_minimal_H_eigenvalue manual minimal eigenvalue proposed for H
   * @note The init method should be called before update. If it has not been
   * done before, init is called depending on the is_initialized flag.
   */
  void update(optional<MatRef<T>> H,
              optional<VecRef<T>> g,
              optional<MatRef<T>> A,
              optional<VecRef<T>> b,
              optional<MatRef<T>> C,
              optional<VecRef<T>> l,
              optional<VecRef<T>> u,
              bool update_preconditioner = false,
              optional<T> rho = nullopt,
              optional<T> mu_eq = nullopt,
              optional<T> mu_in = nullopt,
              optional<T> manual_minimal_H_eigenvalue = nullopt)
  {
    PROXSUITE_THROW_PRETTY(
      box_constraints == true,
      std::invalid_argument,
      "wrong model setup: the QP object is designed without box "
      "constraints, but the update does not include lower or upper box "
      "inequalities.");
    settings.update_preconditioner = update_preconditioner;
    if (!work.is_initialized) {
      init(H, g, A, b, C, l, u, update_preconditioner, rho, mu_eq, mu_in);
      return;
    }
    // dense case
    work.refactorize = false;
    work.proximal_parameter_update = false;
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (update_preconditioner) {
      preconditioner_status = PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = PreconditionerStatus::KEEP;
    }
    const bool matrix_update =
      !(H == nullopt && g == nullopt && A == nullopt && b == nullopt &&
        C == nullopt && u == nullopt && l == nullopt);
    if (matrix_update) {
      typedef optional<VecRef<T>> optional_VecRef;
      common::dense::update(H,
                            g,
                            A,
                            b,
                            C,
                            l,
                            u,
                            optional_VecRef(nullopt),
                            optional_VecRef(nullopt),
                            model,
                            work,
                            box_constraints);
    }
    common::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    common::dense::update_default_rho_with_minimal_Hessian_eigen_value(
      manual_minimal_H_eigenvalue, results, settings);
    typedef optional<MatRef<T>> optional_MatRef;
    typedef optional<VecRef<T>> optional_VecRef;
    common::dense::setup(/* avoid double assignation */
                         optional_MatRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_MatRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_MatRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         settings,
                         model,
                         work,
                         results,
                         box_constraints,
                         ruiz,
                         preconditioner_status,
                         hessian_type);

    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };

  /*!
   * Updates the QP model (with dense matrix format) and re-equilibrates it if
   * specified by the user.
   * @param H quadratic cost input defining the QP model.
   * @param g linear cost input defining the QP model.
   * @param A equality constraint matrix input defining the QP model.
   * @param b equality constraint vector input defining the QP model.
   * @param C inequality constraint matrix input defining the QP model.
   * @param l lower inequality constraint vector input defining the QP model.
   * @param u upper inequality constraint vector input defining the QP model.
   * @param l_box lower inequality constraint vector input defining the QP
   * model.
   * @param u_box upper inequality constraint vector input defining the QP
   * model.
   * @param update_preconditioner bool parameter for updating or not the
   * preconditioner and the associated scaled model.
   * @param rho proximal step size wrt primal variable.
   * @param mu_eq proximal step size wrt equality constrained multiplier.
   * @param mu_in proximal step size wrt inequality constrained multiplier.
   * @param manual_minimal_H_eigenvalue manual minimal eigenvalue proposed for H
   * @note The init method should be called before update. If it has not been
   * done before, init is called depending on the is_initialized flag.
   */
  void update(optional<MatRef<T>> H,
              optional<VecRef<T>> g,
              optional<MatRef<T>> A,
              optional<VecRef<T>> b,
              optional<MatRef<T>> C,
              optional<VecRef<T>> l,
              optional<VecRef<T>> u,
              optional<VecRef<T>> l_box,
              optional<VecRef<T>> u_box,
              bool update_preconditioner = false,
              optional<T> rho = nullopt,
              optional<T> mu_eq = nullopt,
              optional<T> mu_in = nullopt,
              optional<T> manual_minimal_H_eigenvalue = nullopt)
  {
    PROXSUITE_THROW_PRETTY(
      box_constraints == false && (l_box != nullopt || u_box != nullopt),
      std::invalid_argument,
      "wrong model setup: the QP object is designed without box "
      "constraints, but the update includes lower or upper box inequalities.");
    settings.update_preconditioner = update_preconditioner;
    if (!work.is_initialized) {
      init(H,
           g,
           A,
           b,
           C,
           l,
           u,
           l_box,
           u_box,
           update_preconditioner,
           rho,
           mu_eq,
           mu_in);
      return;
    }
    // dense case
    work.refactorize = false;
    work.proximal_parameter_update = false;
    if (settings.compute_timings) {
      work.timer.stop();
      work.timer.start();
    }
    PreconditionerStatus preconditioner_status;
    if (update_preconditioner) {
      preconditioner_status = PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = PreconditionerStatus::KEEP;
    }
    const bool matrix_update =
      !(H == nullopt && g == nullopt && A == nullopt && b == nullopt &&
        C == nullopt && u == nullopt && l == nullopt && u_box == nullopt &&
        l_box == nullopt);
    if (matrix_update) {
      common::dense::update(
        H, g, A, b, C, l, u, l_box, u_box, model, work, box_constraints);
    }
    common::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    common::dense::update_default_rho_with_minimal_Hessian_eigen_value(
      manual_minimal_H_eigenvalue, results, settings);
    typedef optional<MatRef<T>> optional_MatRef;
    typedef optional<VecRef<T>> optional_VecRef;
    common::dense::setup(/* avoid double assignation */
                         optional_MatRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_MatRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_MatRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         optional_VecRef(nullopt),
                         settings,
                         model,
                         work,
                         results,
                         box_constraints,
                         ruiz,
                         preconditioner_status,
                         hessian_type);

    if (settings.compute_timings) {
      results.info.setup_time = work.timer.elapsed().user; // in microseconds
    }
  };

  /*!
   * Solves the QP problem using Derived algorithm.
   */
  void solve() { derived().solve_implem(); };
  /*!
   * Solves the QP problem using solvers algorithm using a warm start.
   * @param x primal warm start.
   * @param y dual equality warm start.
   * @param z dual inequality warm start.
   */
  void solve(optional<VecRef<T>> x,
             optional<VecRef<T>> y,
             optional<VecRef<T>> z)
  {
    common::dense::warm_start(x, y, z, results, settings, model);
    derived().solve_implem();
  };
  /*!
   * Clean-ups solver's results and workspace.
   */
  void cleanup()
  {
    results.cleanup(settings);
    work.cleanup(box_constraints);
  }
};

/*!
 * Solves the QP problem using Derived algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to set
 * up some of the solver parameters (warm start, initial guess option, proximal
 * step sizes, absolute and relative accuracies, maximum number of iterations,
 * preconditioner execution). There are no box constraints in the model.
 * Templates QPDerived and ConfigDerived for solvers specialization.
 */
template<typename QPDerived, typename ConfigDerived, typename T>
Results<T>
solve_base(const ConfigDerived& config,
           optional<MatRef<T>> H,
           optional<VecRef<T>> g,
           optional<MatRef<T>> A,
           optional<VecRef<T>> b,
           optional<MatRef<T>> C,
           optional<VecRef<T>> l,
           optional<VecRef<T>> u,
           optional<VecRef<T>> x = nullopt,
           optional<VecRef<T>> y = nullopt,
           optional<VecRef<T>> z = nullopt)
{
  isize n(0);
  isize n_eq(0);
  isize n_in(0);
  if (H != nullopt) {
    n = H.value().rows();
  }
  if (A != nullopt) {
    n_eq = A.value().rows();
  }
  if (C != nullopt) {
    n_in = C.value().rows();
  }

  QPDerived Qp(n, n_eq, n_in, false, DenseBackend::PrimalDualLDLT);

  config.init_derived_settings(Qp.settings);
  config.init_qp(Qp, H, g, A, b, C, l, u);

  Qp.solve(x, y, z);

  return Qp.results;
}

/*!
 * Solves the QP problem using Derived algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to set
 * up some of the solver parameters (warm start, initial guess option, proximal
 * step sizes, absolute and relative accuracies, maximum number of iterations,
 * preconditioner execution).
 * Templates QPDerived and ConfigDerived for solvers specialization.
 */
template<typename QPDerived, typename ConfigDerived, typename T>
Results<T>
solve_base_box(const ConfigDerived& config,
               optional<MatRef<T>> H,
               optional<VecRef<T>> g,
               optional<MatRef<T>> A,
               optional<VecRef<T>> b,
               optional<MatRef<T>> C,
               optional<VecRef<T>> l,
               optional<VecRef<T>> u,
               optional<VecRef<T>> l_box,
               optional<VecRef<T>> u_box,
               optional<VecRef<T>> x = nullopt,
               optional<VecRef<T>> y = nullopt,
               optional<VecRef<T>> z = nullopt)
{
  isize n(0);
  isize n_eq(0);
  isize n_in(0);
  if (H != nullopt) {
    n = H.value().rows();
  }
  if (A != nullopt) {
    n_eq = A.value().rows();
  }
  if (C != nullopt) {
    n_in = C.value().rows();
  }

  QPDerived Qp(n, n_eq, n_in, true, DenseBackend::PrimalDualLDLT);

  config.init_derived_settings(Qp.settings);
  config.init_qp_box(Qp, H, g, A, b, C, l, u, l_box, u_box);

  Qp.solve(x, y, z);

  return Qp.results;
}

} // namespace dense
} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_COMMON_DENSE_WRAPPER_HPP */
