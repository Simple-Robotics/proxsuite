//
// Copyright (c) 2022-2025 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_PROXQP_DENSE_WRAPPER_HPP
#define PROXSUITE_PROXQP_DENSE_WRAPPER_HPP

#include <proxsuite/proxqp/dense/aliases.hpp>
#include <proxsuite/proxqp/dense/solver.hpp>
#include <proxsuite/common/dense/wrapper.hpp>

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
        isize dim = 10;
        isize n_eq(dim / 4);
        isize n_in(dim / 4);
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

template<typename T>
struct QP : public common::dense::QPBase<QP<T>, T>
{
private:
  using Base = common::dense::QPBase<QP<T>, T>;

public:
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
     HessianType _hessian_type,
     DenseBackend _dense_backend)
    : Base(_dim, _n_eq, _n_in, _box_constraints, _hessian_type, _dense_backend)
  {
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
     HessianType _hessian_type)
    : Base(_dim, _n_eq, _n_in, _box_constraints, _dense_backend, _hessian_type)
  {
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
     HessianType _hessian_type)
    : Base(_dim, _n_eq, _n_in, _box_constraints, _hessian_type)
  {
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
    : Base(_dim, _n_eq, _n_in, _box_constraints, _dense_backend)
  {
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _box_constraints specify that there are (or not) box constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in, bool _box_constraints)
    : Base(_dim, _n_eq, _n_in, _box_constraints)
  {
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   * @param _hessian_type specify that there are (or not) box constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in, HessianType _hessian_type)
    : Base(_dim, _n_eq, _n_in, _hessian_type)
  {
  }
  /*!
   * Default constructor using QP model dimensions.
   * @param _dim primal variable dimension.
   * @param _n_eq number of equality constraints.
   * @param _n_in number of inequality constraints.
   */
  QP(isize _dim, isize _n_eq, isize _n_in)
    : Base(_dim, _n_eq, _n_in)
  {
  }
  /*!
   * Initialize ProxQP-specific settings.
   */
  void init_derived_settings()
  {
    this->settings.default_mu_eq = 1.E-3;
    this->settings.default_mu_in = 1.E-1;

    this->settings.alpha_bcl = 0.1;
    this->settings.beta_bcl = 0.9;
    this->settings.refactor_dual_feasibility_threshold = 1E-2;
    this->settings.refactor_rho_threshold = 1E-7;

    this->settings.mu_min_eq = 1E-9;
    this->settings.mu_min_in = 1E-8;
    this->settings.mu_max_eq_inv = 1E9;
    this->settings.mu_max_in_inv = 1E8;

    this->settings.mu_update_factor = 0.1;
    this->settings.mu_update_inv_factor = 10;
    this->settings.cold_reset_mu_eq = 1. / 1.1;
    this->settings.cold_reset_mu_in = 1. / 1.1;
    this->settings.cold_reset_mu_eq_inv = 1.1;
    this->settings.cold_reset_mu_in_inv = 1.1;

    this->settings.eps_abs = 1.E-5;
    this->settings.eps_rel = 0;
    this->settings.max_iter = 10000;
    this->settings.max_iter_in = 1500;
    this->settings.safe_guard = 1.E4;
    this->settings.nb_iterative_refinement = 10;
    this->settings.eps_refact = 1.E-6;

    this->settings.verbose = false;
    this->settings.initial_guess =
      InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
    this->settings.update_preconditioner = false;
    this->settings.compute_preconditioner = true;
    this->settings.compute_timings = false;

    this->settings.check_duality_gap = false;
    this->settings.eps_duality_gap_abs = 1.E-4;
    this->settings.eps_duality_gap_rel = 0;

    this->settings.preconditioner_max_iter = 10;
    this->settings.preconditioner_accuracy = 1.E-3;
    this->settings.eps_primal_inf = 1.E-4;
    this->settings.eps_dual_inf = 1.E-4;
    this->settings.bcl_update = true;
    this->settings.merit_function_type = MeritFunctionType::GPDAL;
    this->settings.alpha_gpdal = 0.95;
    this->settings.sparse_backend = SparseBackend::Automatic;
    this->settings.primal_infeasibility_solving = false;
    this->settings.frequence_infeasibility_check = 1;
    this->settings.default_H_eigenvalue_estimate = 0.;

    this->settings.alpha = 1.6;
    this->settings.mu_max_eq = 1E3;
    this->settings.mu_max_in = 1E6;
    this->settings.mu_min_eq_inv = 1E-3;
    this->settings.mu_min_in_inv = 1E-6;
    this->settings.adaptive_mu = true;
    this->settings.adaptive_mu_interval = 50;
    this->settings.adaptive_mu_tolerance = 5.;
    this->settings.polishing = false;
    this->settings.delta = 1E-6;
    this->settings.polish_refine_iter = 3;
  }
  /*!
   * Initialize ProxQP-specific results.
   */
  void init_derived_results()
  {
    this->results.info.mu_eq = 1E-3;
    this->results.info.mu_in = 1E-1;
    this->results.info.mu_eq_inv = 1E3;
    this->results.info.mu_in_inv = 1E1;
  }
  /*!
   * ProxQP-specific solve implementation.
   * Calls the ProxQP algorithm.
   */
  void solve_implem()
  {
    qp_solve( //
      this->settings,
      this->model,
      this->results,
      this->work,
      this->is_box_constrained(),
      this->which_dense_backend(),
      this->which_hessian_type(),
      this->ruiz);
  }
};

///
/// @brief This class defines the ProxQP default parameter
/// configuration of the function proxqp::dense::solve<T>.
///
template<typename T>
struct ProxQPConfig
{
  optional<T> eps_abs;
  optional<T> eps_rel;
  optional<T> rho;
  optional<T> mu_eq;
  optional<T> mu_in;
  optional<bool> verbose;
  bool compute_preconditioner;
  bool compute_timings;
  optional<isize> max_iter;
  InitialGuessStatus initial_guess;
  bool check_duality_gap;
  optional<T> eps_duality_gap_abs;
  optional<T> eps_duality_gap_rel;
  bool primal_infeasibility_solving;
  optional<T> manual_minimal_H_eigenvalue;

  ProxQPConfig(optional<T> eps_abs = nullopt,
               optional<T> eps_rel = nullopt,
               optional<T> rho = nullopt,
               optional<T> mu_eq = nullopt,
               optional<T> mu_in = nullopt,
               optional<bool> verbose = nullopt,
               bool compute_preconditioner = true,
               bool compute_timings = false,
               optional<isize> max_iter = nullopt,
               InitialGuessStatus initial_guess =
                 InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
               bool check_duality_gap = false,
               optional<T> eps_duality_gap_abs = nullopt,
               optional<T> eps_duality_gap_rel = nullopt,
               bool primal_infeasibility_solving = false,
               optional<T> manual_minimal_H_eigenvalue = nullopt)
    : eps_abs(eps_abs)
    , eps_rel(eps_rel)
    , rho(rho)
    , mu_eq(mu_eq)
    , mu_in(mu_in)
    , verbose(verbose)
    , compute_preconditioner(compute_preconditioner)
    , compute_timings(compute_timings)
    , max_iter(max_iter)
    , initial_guess(initial_guess)
    , check_duality_gap(check_duality_gap)
    , eps_duality_gap_abs(eps_duality_gap_abs)
    , eps_duality_gap_rel(eps_duality_gap_rel)
    , primal_infeasibility_solving(primal_infeasibility_solving)
    , manual_minimal_H_eigenvalue(manual_minimal_H_eigenvalue)
  {
  }
  /*!
   * ProxQP settings initialization.
   */
  void init_derived_settings(Settings<T>& settings) const
  {
    settings.initial_guess = initial_guess;
    settings.check_duality_gap = check_duality_gap;
    settings.compute_timings = compute_timings;
    settings.primal_infeasibility_solving = primal_infeasibility_solving;

    if (eps_abs != nullopt) {
      settings.eps_abs = eps_abs.value();
    }
    if (eps_rel != nullopt) {
      settings.eps_rel = eps_rel.value();
    }
    if (verbose != nullopt) {
      settings.verbose = verbose.value();
    }
    if (max_iter != nullopt) {
      settings.max_iter = max_iter.value();
    }
    if (eps_duality_gap_abs != nullopt) {
      settings.eps_duality_gap_abs = eps_duality_gap_abs.value();
    }
    if (eps_duality_gap_rel != nullopt) {
      settings.eps_duality_gap_rel = eps_duality_gap_rel.value();
    }
  }
  /*!
   * Call to init() from QPBase without box constraints.
   */
  void init_qp(QP<T>& qp,
               optional<MatRef<T>> H,
               optional<VecRef<T>> g,
               optional<MatRef<T>> A,
               optional<VecRef<T>> b,
               optional<MatRef<T>> C,
               optional<VecRef<T>> l,
               optional<VecRef<T>> u) const
  {
    if (manual_minimal_H_eigenvalue != nullopt) {
      qp.init(H,
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
      qp.init(H,
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
              nullopt);
    }
  }
  /*!
   * Call to QPBase init() without box constraints.
   */
  void init_qp_box(QP<T>& qp,
                   optional<MatRef<T>> H,
                   optional<VecRef<T>> g,
                   optional<MatRef<T>> A,
                   optional<VecRef<T>> b,
                   optional<MatRef<T>> C,
                   optional<VecRef<T>> l,
                   optional<VecRef<T>> u,
                   optional<VecRef<T>> l_box,
                   optional<VecRef<T>> u_box) const
  {
    if (manual_minimal_H_eigenvalue != nullopt) {
      qp.init(H,
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
      qp.init(H,
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
  }
};

/*!
 * Solves the QP problem using PROXQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to
 * set up some of the solver parameters (warm start, initial guess option,
 * proximal step sizes, absolute and relative accuracies, maximum number of
 * iterations, preconditioner execution). There are no box constraints in the
 * model.
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
 * @param verbose if set to true, the solver prints more information about
 * each iteration.
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
 * @param check_duality_gap If set to true, include the duality gap in
 * absolute and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 */
template<typename T>
Results<T>
solve(optional<MatRef<T>> H,
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
      InitialGuessStatus initial_guess =
        InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      bool check_duality_gap = false,
      optional<T> eps_duality_gap_abs = nullopt,
      optional<T> eps_duality_gap_rel = nullopt,
      bool primal_infeasibility_solving = false,
      optional<T> manual_minimal_H_eigenvalue = nullopt)
{
  ProxQPConfig<T> config(eps_abs,
                         eps_rel,
                         rho,
                         mu_eq,
                         mu_in,
                         verbose,
                         compute_preconditioner,
                         compute_timings,
                         max_iter,
                         initial_guess,
                         check_duality_gap,
                         eps_duality_gap_abs,
                         eps_duality_gap_rel,
                         primal_infeasibility_solving,
                         manual_minimal_H_eigenvalue);

  return common::dense::solve_base<QP<T>, ProxQPConfig<T>, T>(
    config, H, g, A, b, C, l, u, x, y, z);
}
/*!
 * Solves the QP problem using PROXQP algorithm without the need to define a QP
 * object, with matrices defined by Dense Eigen matrices. It is possible to
 * set up some of the solver parameters (warm start, initial guess option,
 * proximal step sizes, absolute and relative accuracies, maximum number of
 * iterations, preconditioner execution).
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
 * @param z dual inequality constraint warm start. The upper part must contain
 * a warm start for inequality constraints wrt C matrix, whereas the latter
 * wrt the box inequalities.
 * @param verbose if set to true, the solver prints more information about
 * each iteration.
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
 * @param check_duality_gap If set to true, include the duality gap in
 * absolute and relative stopping criteria.
 * @param eps_duality_gap_abs absolute accuracy threshold for the duality-gap
 * criterion.
 * @param eps_duality_gap_rel relative accuracy threshold for the duality-gap
 * criterion.
 */
template<typename T>
Results<T>
solve(optional<MatRef<T>> H,
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
      InitialGuessStatus initial_guess =
        InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      bool check_duality_gap = false,
      optional<T> eps_duality_gap_abs = nullopt,
      optional<T> eps_duality_gap_rel = nullopt,
      bool primal_infeasibility_solving = false,
      optional<T> manual_minimal_H_eigenvalue = nullopt)
{
  ProxQPConfig<T> config(eps_abs,
                         eps_rel,
                         rho,
                         mu_eq,
                         mu_in,
                         verbose,
                         compute_preconditioner,
                         compute_timings,
                         max_iter,
                         initial_guess,
                         check_duality_gap,
                         eps_duality_gap_abs,
                         eps_duality_gap_rel,
                         primal_infeasibility_solving,
                         manual_minimal_H_eigenvalue);

  return common::dense::solve_base_box<QP<T>, ProxQPConfig<T>, T>(
    config, H, g, A, b, C, l, u, l_box, u_box, x, y, z);
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
