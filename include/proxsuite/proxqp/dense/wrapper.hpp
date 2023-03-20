//
// Copyright (c) 2022 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_PROXQP_DENSE_WRAPPER_HPP
#define PROXSUITE_PROXQP_DENSE_WRAPPER_HPP
#include <proxsuite/proxqp/dense/solver.hpp>
#include <proxsuite/proxqp/dense/helpers.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>
#include <chrono>

namespace proxsuite {
namespace proxqp {
namespace dense {
///
/// @brief This class defines the API of PROXQP solver with dense backend.
///
/*!
 * Wrapper class for using proxsuite API with dense backend
 * for solving linearly constrained convex QP problem using ProxQp algorithm.
 *
 * Example usage:
 * ```cpp
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <util.hpp>

using T = double;
auto main() -> int {

        // Generate a random QP problem with primal variable dimension of size
dim; n_eq equality constraints and n_in inequality constraints
        ::proxsuite::proxqp::test::rand::set_seed(1);
        proxqp::isize dim = 10;
        proxqp::isize n_eq(dim / 4);
        proxqp::isize n_in(dim / 4);
        T strong_convexity_factor(1.e-2);
        T sparsity_factor = 0.15; // controls the sparsity of each matrix of the
problem generated T eps_abs = T(1e-9); Qp<T> qp{
                        random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor};

        // Solve the problem
        proxqp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
        Qp.settings.eps_abs = eps_abs; // choose accuracy needed
        Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // setup the QP
object Qp.solve(); // solve the problem

        // Verify solution accuracy
        T pri_res = std::max(
                        (qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                        (helpers::positive_part(qp.C * Qp.results.x -
qp.u) + helpers::negative_part(qp.C * Qp.results.x - qp.l))
                                        .lpNorm<Eigen::Infinity>());
        T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() *
Qp.results.y + qp.C.transpose() * Qp.results.z) .lpNorm<Eigen::Infinity>();
        VEG_ASSERT(pri_res <= eps_abs);
        VEG_ASSERT(dua_res <= eps_abs);

        // Some solver statistics
        std::cout << "------solving qp with dim: " << dim
                                                << " neq: " << n_eq << " nin: "
<< n_in << std::endl; std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << Qp.results.info.iter
                                                << std::endl;
}
 * ```
 */
///// QP object
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
template<typename T>
struct QP
{
private:
  // structure of the problem
  // not supposed to change
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
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     proxsuite::proxqp::HessianType _hessian_type,
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
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     DenseBackend _dense_backend,
     proxsuite::proxqp::HessianType _hessian_type)
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
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type problem type (QP, LP, DIAGONAL)
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     bool _box_constraints,
     proxsuite::proxqp::HessianType _hessian_type)
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
  QP(isize _dim,
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
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in, bool _box_constraints)
    : dense_backend(dense_backend_choice<T>(DenseBackend::Automatic,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            _box_constraints))
    , box_constraints(_box_constraints)
    , hessian_type(proxsuite::proxqp::HessianType::Dense)
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
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type specify that there are (or not) box constraints.
   */
  QP(isize _dim,
     isize _n_eq,
     isize _n_in,
     proxsuite::proxqp::HessianType _hessian_type)
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
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in)
    : dense_backend(dense_backend_choice<T>(DenseBackend::Automatic,
                                            _dim,
                                            _n_eq,
                                            _n_in,
                                            false))
    , box_constraints(false)
    , hessian_type(proxsuite::proxqp::HessianType::Dense)
    , results(_dim, _n_eq, _n_in, false, dense_backend)
    , settings(dense_backend)
    , model(_dim, _n_eq, _n_in, false)
    , work(_dim, _n_eq, _n_in, false, dense_backend)
    , ruiz(preconditioner::RuizEquilibration<T>{ _dim, _n_eq, _n_in, false })
  {
    work.timer.stop();
  }
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
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::IDENTITY;
    }
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::
      update_default_rho_with_minimal_Hessian_eigen_value(
        manual_minimal_H_eigenvalue, results, settings);
    typedef optional<VecRef<T>> optional_VecRef;
    proxsuite::proxqp::dense::setup(H,
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
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::IDENTITY;
    }
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::
      update_default_rho_with_minimal_Hessian_eigen_value(
        manual_minimal_H_eigenvalue, results, settings);
    proxsuite::proxqp::dense::setup(H,
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
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::KEEP;
    }
    const bool matrix_update =
      !(H == nullopt && g == nullopt && A == nullopt && b == nullopt &&
        C == nullopt && u == nullopt && l == nullopt);
    if (matrix_update) {
      typedef optional<VecRef<T>> optional_VecRef;
      proxsuite::proxqp::dense::update(H,
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
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::
      update_default_rho_with_minimal_Hessian_eigen_value(
        manual_minimal_H_eigenvalue, results, settings);
    typedef optional<MatRef<T>> optional_MatRef;
    typedef optional<VecRef<T>> optional_VecRef;
    proxsuite::proxqp::dense::setup(/* avoid double assignation */
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
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
    } else {
      preconditioner_status = proxsuite::proxqp::PreconditionerStatus::KEEP;
    }
    const bool matrix_update =
      !(H == nullopt && g == nullopt && A == nullopt && b == nullopt &&
        C == nullopt && u == nullopt && l == nullopt && u_box == nullopt &&
        l_box == nullopt);
    if (matrix_update) {
      proxsuite::proxqp::dense::update(
        H, g, A, b, C, l, u, l_box, u_box, model, work, box_constraints);
    }
    proxsuite::proxqp::dense::update_proximal_parameters(
      settings, results, work, rho, mu_eq, mu_in);
    proxsuite::proxqp::dense::
      update_default_rho_with_minimal_Hessian_eigen_value(
        manual_minimal_H_eigenvalue, results, settings);
    typedef optional<MatRef<T>> optional_MatRef;
    typedef optional<VecRef<T>> optional_VecRef;
    proxsuite::proxqp::dense::setup(/* avoid double assignation */
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
   * Solves the QP problem using PRXOQP algorithm.
   */
  void solve()
  {
    qp_solve( //
      settings,
      model,
      results,
      work,
      box_constraints,
      dense_backend,
      hessian_type,
      ruiz);
  };
  /*!
   * Solves the QP problem using PROXQP algorithm using a warm start.
   * @param x primal warm start.
   * @param y dual equality warm start.
   * @param z dual inequality warm start.
   */
  void solve(optional<VecRef<T>> x,
             optional<VecRef<T>> y,
             optional<VecRef<T>> z)
  {
    proxsuite::proxqp::dense::warm_start(x, y, z, results, settings, model);
    qp_solve( //
      settings,
      model,
      results,
      work,
      box_constraints,
      dense_backend,
      hessian_type,
      ruiz);
  };
  /*!
   * Clean-ups solver's results and workspace.
   */
  void cleanup()
  {
    results.cleanup(settings);
    work.cleanup(box_constraints);
  }
  void compute_backward(VecRef<T> loss_derivative, T eps = 1.E-4)
  {
    bool check = results.info.status != QPSolverOutput::PROXQP_SOLVED &&
                 results.info.status !=
                   QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE &&
                 results.info.status != QPSolverOutput::PROXQP_MAX_ITER_REACHED;
    if (check) {
      // Add it as option warning without max iter reached
      // PROXSUITE_THROW_PRETTY(
      //   true,
      //   std::invalid_argument,
      //   "the QP problem has not been solved, so computing the derivatives "
      //   "is not valid.");
      PROXSUITE_THROW_PRETTY(
        true,
        std::invalid_argument,
        "the QP problem is not feasible, so computing the derivatives "
        "is not valid in this setting. Try enabling infeasible solving if "
        "the problem is only primally infeasible.");
    }
    model.backward_data.initialize(model.dim, model.n_eq, model.n_in);
    if (!check) {

      /// derive solution
      work.CTz.noalias() = model.C * results.x + results.z;
      work.active_set_up.array() = (work.CTz - model.u).array() >= 0.;
      work.active_set_low.array() = (work.CTz - model.l).array() <= 0.;
      work.active_inequalities = work.active_set_up || work.active_set_low;
      isize numactive_inequalities = work.active_inequalities.count();
      isize inner_pb_dim = model.dim + model.n_eq + numactive_inequalities;
      work.rhs.setZero();
      // work.dw_aug.setZero(); zeroed in active_set_change
      work.n_c = 0;
      linesearch::active_set_change(model, results, work);
      // a large amount of constraints might have changed
      // so to avoir too much refactorization later in the
      // iterative refinament, refactorization is directly
      // performed with mu changes as well
      T rho_new = 1.e-4;
      results.info.mu_eq = rho_new;
      results.info.mu_in = rho_new;
      work.constraints_changed = true;
      refactorize(model, results, work, rho_new);

      bool check_bis = numactive_inequalities == work.n_c;
      if (!check_bis) {
        std::cout << "work.n_c "
                  << " numactive_inequalities " << numactive_inequalities
                  << std::endl;
      }
      work.rhs.head(model.dim) =
        loss_derivative.head(model.dim); //-loss_derivative.head(model.dim);
      // std::cout << "work.active_set_up " << work.active_set_up << std::endl;
      // std::cout << "work.rhs " << work.rhs<< std::endl;

      ruiz.scale_dual_residual_in_place(
        VectorViewMut<T>{ from_eigen, work.rhs.head(model.dim) });
      if (!work.rhs.segment(model.dim, model.n_eq).isZero()) {
        work.rhs.segment(model.dim, model.n_eq) =
          // -loss_derivative.segment(model.dim, model.n_eq);
          loss_derivative.segment(model.dim, model.n_eq);
        ruiz.scale_primal_residual_in_place_eq(VectorViewMut<T>{
          from_eigen, work.rhs.segment(model.dim, model.n_eq) });
      }
      if (!work.rhs.tail(model.n_in).isZero()) {
        for (isize i = 0; i < model.n_in; i++) {
          isize j = work.current_bijection_map(i);
          if (j < work.n_c) {
            work.rhs(j + model.dim + model.n_eq) = loss_derivative(
              i + model.dim +
              model.n_eq); //-loss_derivative(i + model.dim + model.n_eq);
          }
          ruiz.scale_primal_residual_in_place_in(
            VectorViewMut<T>{ from_eigen, work.rhs.tail(model.n_in) });
        }
      }
      iterative_solve_with_permut_fact( //
        settings,
        model,
        results,
        work,
        eps,
        inner_pb_dim); // /!\ the full rhs is zeroed inside
      // use active_part_z as a temporary variable to derive unpermutted dz step
      work.active_part_z.setZero();
      for (isize j = 0; j < model.n_in; ++j) {
        isize i = work.current_bijection_map(j);
        if (i < work.n_c) {
          work.active_part_z(j) = work.dw_aug(model.dim + model.n_eq + i);
        }
        // we actually don't care of the part below for jacobian calculus
        // else {
        //   work.active_part_z(j) = -loss_derivative(model.dim + model.n_eq +
        //   j);
        // }
      }
      work.dw_aug.tail(model.n_in) = work.active_part_z;
      ruiz.unscale_primal_in_place(
        VectorViewMut<T>{ from_eigen, work.dw_aug.head(model.dim) });
      ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{
        from_eigen, work.dw_aug.segment(model.dim, model.n_eq) });
      ruiz.unscale_dual_in_place_in(
        VectorViewMut<T>{ from_eigen, work.dw_aug.tail(model.n_in) });

      /// compute bawkward derivatives
      model.backward_data.dL_dC.noalias() =
        work.dw_aug.tail(model.n_in) * results.x.transpose();
      model.backward_data.dL_dC.noalias() +=
        results.z * work.dw_aug.head(model.dim).transpose();

      model.backward_data.dL_du = -work.dw_aug.tail(model.n_in);
      // (work.active_set_up)
      //   .select(-work.dw_aug.tail(model.n_in),
      //           Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in));
      model.backward_data.dL_dl = -work.dw_aug.tail(model.n_in);
      // (work.active_set_low)
      //   .select(-work.dw_aug.tail(model.n_in),
      //           Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in));

      model.backward_data.dL_dA.noalias() =
        work.dw_aug.segment(model.dim, model.n_eq) * results.x.transpose();
      model.backward_data.dL_dA.noalias() +=
        results.y * work.dw_aug.head(model.dim).transpose();

      model.backward_data.dL_db = -work.dw_aug.segment(model.dim, model.n_eq);
      model.backward_data.dL_dH.noalias() =
        0.5 * (work.dw_aug.head(model.dim) * results.x.transpose() +
               results.x * work.dw_aug.head(model.dim).transpose());

      model.backward_data.dL_dg = work.dw_aug.head(model.dim);
    }
  }
};
/*!
 * Solves the QP problem using PROXQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to set
 * up some of the solver parameters (warm start, initial guess option, proximal
 * step sizes, absolute and relative accuracies, maximum number of iterations,
 * preconditioner execution). There are no box constraints in the model.
 * @param H quadratic cost input defining the QP model.
 * @param g linear cost input defining the QP model.
 * @param A equality constraint matrix input defining the QP model.
 * @param b equality constraint vector input defining the QP model.
 * @param C inequality constraint matrix input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param u upper inequality constraint vector input defining the QP model.
 * @param x primal warm start.
 * @param y dual equality constraint warm start.
 * @param z dual inequality constraint warm start.
 * @param verbose if set to true, the solver prints more information about each
 * iteration.
 * @param compute_preconditioner bool parameter for executing or not the
 * preconditioner.
 * @param compute_timings boolean parameter for computing the solver timings.
 * @param rho proximal step size wrt primal variable.
 * @param mu_eq proximal step size wrt equality constrained multiplier.
 * @param mu_in proximal step size wrt inequality constrained multiplier.
 * @param eps_abs absolute accuracy threshold.
 * @param eps_rel relative accuracy threshold.
 * @param max_iter maximum number of iteration.
 * @param initial_guess initial guess option for warm starting or not the
 * initial iterate values.
 * @param check_duality_gap If set to true, include the duality gap in absolute
 * and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 */
template<typename T>
proxqp::Results<T>
solve(
  optional<MatRef<T>> H,
  optional<VecRef<T>> g,
  optional<MatRef<T>> A,
  optional<VecRef<T>> b,
  optional<MatRef<T>> C,
  optional<VecRef<T>> l,
  optional<VecRef<T>> u,
  optional<VecRef<T>> x = nullopt,
  optional<VecRef<T>> y = nullopt,
  optional<VecRef<T>> z = nullopt,
  optional<T> eps_abs = nullopt,
  optional<T> eps_rel = nullopt,
  optional<T> rho = nullopt,
  optional<T> mu_eq = nullopt,
  optional<T> mu_in = nullopt,
  optional<bool> verbose = nullopt,
  bool compute_preconditioner = true,
  bool compute_timings = false,
  optional<isize> max_iter = nullopt,
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false,
  optional<T> manual_minimal_H_eigenvalue = nullopt)
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

  QP<T> Qp(n, n_eq, n_in, false, DenseBackend::PrimalDualLDLT);
  Qp.settings.initial_guess = initial_guess;
  Qp.settings.check_duality_gap = check_duality_gap;

  if (eps_abs != nullopt) {
    Qp.settings.eps_abs = eps_abs.value();
  }
  if (eps_rel != nullopt) {
    Qp.settings.eps_rel = eps_rel.value();
  }
  if (verbose != nullopt) {
    Qp.settings.verbose = verbose.value();
  }
  if (max_iter != nullopt) {
    Qp.settings.max_iter = max_iter.value();
  }
  if (eps_duality_gap_abs != nullopt) {
    Qp.settings.eps_duality_gap_abs = eps_duality_gap_abs.value();
  }
  if (eps_duality_gap_rel != nullopt) {
    Qp.settings.eps_duality_gap_rel = eps_duality_gap_rel.value();
  }
  Qp.settings.compute_timings = compute_timings;
  Qp.settings.primal_infeasibility_solving = primal_infeasibility_solving;
  if (manual_minimal_H_eigenvalue != nullopt) {
    Qp.init(H,
            g,
            A,
            b,
            C,
            l,
            u,
            compute_preconditioner,
            rho,
            mu_eq,
            mu_in,
            manual_minimal_H_eigenvalue.value());
  } else {
    Qp.init(
      H, g, A, b, C, l, u, compute_preconditioner, rho, mu_eq, mu_in, nullopt);
  }
  Qp.solve(x, y, z);

  return Qp.results;
}
/*!
 * Solves the QP problem using PROXQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to set
 * up some of the solver parameters (warm start, initial guess option, proximal
 * step sizes, absolute and relative accuracies, maximum number of iterations,
 * preconditioner execution).
 * @param H quadratic cost input defining the QP model.
 * @param g linear cost input defining the QP model.
 * @param A equality constraint matrix input defining the QP model.
 * @param b equality constraint vector input defining the QP model.
 * @param C inequality constraint matrix input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param u upper inequality constraint vector input defining the QP model.
 * @param l_box lower box inequality constraint vector input defining the QP
 * model.
 * @param u_box upper box inequality constraint vector input defining the QP
 * model.
 * @param x primal warm start.
 * @param y dual equality constraint warm start.
 * @param z dual inequality constraint warm start. The upper part must contain a
 * warm start for inequality constraints wrt C matrix, whereas the latter wrt
 * the box inequalities.
 * @param verbose if set to true, the solver prints more information about each
 * iteration.
 * @param compute_preconditioner bool parameter for executing or not the
 * preconditioner.
 * @param compute_timings boolean parameter for computing the solver timings.
 * @param rho proximal step size wrt primal variable.
 * @param mu_eq proximal step size wrt equality constrained multiplier.
 * @param mu_in proximal step size wrt inequality constrained multiplier.
 * @param eps_abs absolute accuracy threshold.
 * @param eps_rel relative accuracy threshold.
 * @param max_iter maximum number of iteration.
 * @param initial_guess initial guess option for warm starting or not the
 * initial iterate values.
 * @param check_duality_gap If set to true, include the duality gap in absolute
 * and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 */
template<typename T>
proxqp::Results<T>
solve(
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
  optional<VecRef<T>> z = nullopt,
  optional<T> eps_abs = nullopt,
  optional<T> eps_rel = nullopt,
  optional<T> rho = nullopt,
  optional<T> mu_eq = nullopt,
  optional<T> mu_in = nullopt,
  optional<bool> verbose = nullopt,
  bool compute_preconditioner = true,
  bool compute_timings = false,
  optional<isize> max_iter = nullopt,
  proxsuite::proxqp::InitialGuessStatus initial_guess =
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
  bool check_duality_gap = false,
  optional<T> eps_duality_gap_abs = nullopt,
  optional<T> eps_duality_gap_rel = nullopt,
  bool primal_infeasibility_solving = false,
  optional<T> manual_minimal_H_eigenvalue = nullopt)
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

  QP<T> Qp(n, n_eq, n_in, true, DenseBackend::PrimalDualLDLT);
  Qp.settings.initial_guess = initial_guess;
  Qp.settings.check_duality_gap = check_duality_gap;

  if (eps_abs != nullopt) {
    Qp.settings.eps_abs = eps_abs.value();
  }
  if (eps_rel != nullopt) {
    Qp.settings.eps_rel = eps_rel.value();
  }
  if (verbose != nullopt) {
    Qp.settings.verbose = verbose.value();
  }
  if (max_iter != nullopt) {
    Qp.settings.max_iter = max_iter.value();
  }
  if (eps_duality_gap_abs != nullopt) {
    Qp.settings.eps_duality_gap_abs = eps_duality_gap_abs.value();
  }
  if (eps_duality_gap_rel != nullopt) {
    Qp.settings.eps_duality_gap_rel = eps_duality_gap_rel.value();
  }
  Qp.settings.compute_timings = compute_timings;
  Qp.settings.primal_infeasibility_solving = primal_infeasibility_solving;
  if (manual_minimal_H_eigenvalue != nullopt) {
    Qp.init(H,
            g,
            A,
            b,
            C,
            l,
            u,
            l_box,
            u_box,
            compute_preconditioner,
            rho,
            mu_eq,
            mu_in,
            manual_minimal_H_eigenvalue.value());
  } else {
    Qp.init(H,
            g,
            A,
            b,
            C,
            l,
            u,
            l_box,
            u_box,
            compute_preconditioner,
            rho,
            mu_eq,
            mu_in,
            nullopt);
  }
  Qp.solve(x, y, z);

  return Qp.results;
}

template<typename T>
bool
operator==(const QP<T>& qp1, const QP<T>& qp2)
{
  bool value = qp1.model == qp2.model && qp1.settings == qp2.settings &&
               qp1.results == qp2.results &&
               qp1.is_box_constrained() == qp2.is_box_constrained();
  return value;
}

template<typename T>
bool
operator!=(const QP<T>& qp1, const QP<T>& qp2)
{
  return !(qp1 == qp2);
}

///// BatchQP object
template<typename T>
struct BatchQP
{
  /*!
   * A vector of QP aligned of size BatchSize
   * specified by the user.
   */
  std::vector<QP<T>> qp_vector;
  dense::isize m_size;

  explicit BatchQP(size_t batch_size)
  {
    if (qp_vector.max_size() != batch_size) {
      qp_vector.clear();
      qp_vector.reserve(batch_size);
    }
    m_size = 0;
  }

  /*!
   * Init a QP in place and return a reference to it
   */
  QP<T>& init_qp_in_place(dense::isize dim,
                          dense::isize n_eq,
                          dense::isize n_in)
  {
    qp_vector.emplace_back(dim, n_eq, n_in);
    auto& qp = qp_vector.back();
    m_size++;
    return qp;
  };

  /*!
   * Inserts a QP to the end of qp_vector
   */
  void insert(const QP<T>& qp) { qp_vector.emplace_back(qp); };

  /*!
   * Access QP at position i
   */
  QP<T>& get(isize i) { return qp_vector.at(size_t(i)); };

  /*!
   * Access QP at position i
   */
  const QP<T>& get(isize i) const { return qp_vector.at(size_t(i)); };

  /*!
   * Access QP at position i
   */
  QP<T>& operator[](isize i) { return get(i); };

  /*!
   * Access QP at position i
   */
  const QP<T>& operator[](isize i) const { return get(i); };

  dense::isize size() { return m_size; };
};

} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_WRAPPER_HPP */
