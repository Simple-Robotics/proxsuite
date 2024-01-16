## ProxQP API with examples

ProxQP solves convex quadratic programs, which minimize a convex quadratic cost under some linear constraints. It is mathematically described as:

$$\begin{equation}\label{eq:QP}\tag{QP}
\begin{aligned}
    \min_{x\in\mathbb{R}^{d}} & \quad \frac{1}{2}x^{T}Hx+g^{T}x \\\
    \text{s.t.}&\left\{
    \begin{array}{ll}
         Ax = b, \\\
        Cx \leq u. \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}\\\
\text{with } H\in\mathbb{R}^{d\times d}, A\in\mathbb{R}^{n_\text{eq}\times d}, C\in\mathbb{R}^{n_\text{in}\times d}, b\in\mathbb{R}^{n_\text{eq}}, u\in\mathbb{R}^{n_\text{in}}.
$$
H is a real symmetric positive semi-definite matrix. d is the problem dimension (i.e., the number of primal variables), while n_eq and n_in are the numbers of equality and inequality constraints, respectively.

For linearly constrained convex optimization problems such as \eqref{eq:QP}, strong duality holds, and the associated KKT conditions are necessary and sufficient for ensuring a primal-dual point (x,y,z) to be optimal (see, e.g.,[Section 5.2.3](https://web.stanford.edu/~boyd/cvxbook/)} and [Section 2, page 5](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf) for more details).
For \eqref{eq:QP}, the KKT system is given by the set of equations:

$$\begin{equation}\label{qp:kkt}\tag{KKT}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        Hx+g+A^Ty+C^Tz = 0, \\\
        Ax-b = 0, \\\
        Cx \leq u, \\\
        z\odot[Cx-u] = 0,\\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

where the last equation involves the Hadamard product (i.e., for two vectors, u and v, the Hadamard product is the vector whose ith entry is u_i v_i).

In practice, we look for a triplet (x,y,z) satisfying these optimality conditions \eqref{qp:kkt} up to a certain level of absolute accuracy (dependent of the application), leading us to the following absolute stopping criterion on the primal and dual residuals:

$$\begin{equation}\label{eq:approx_qp_sol}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|Hx+g+A^Ty+C^Tz\|_{\infty} \leq  \epsilon_{abs}, \\\
        \|Ax-b\|_{\infty} \leq \epsilon_{abs}, \\\
        \|[Cx-u]_+\|_{\infty}\leq \epsilon_{abs}. \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

The infinite norm is preferred to the L2 norm as it is independent of the problem dimensions. It is also common to consider relative convergence criteria for early stopping, as absolute targets might not be reached due to numerical issues. ProxQP provides it in a similar way as OSQP (for more details, see, e.g., OSQP's [convergence](https://osqp.org/docs/solver/index.html#convergence) criteria or [section 3.4](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf) in the corresponding paper). Hence more generally, the following stopping criterion can be used:

$$\begin{equation}\label{eq:approx_qp_sol_relative_criterion}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|Hx+g+A^Ty+C^Tz\|_{\infty} \leq  \epsilon_{\text{abs}} + \epsilon_{\text{rel}}\max(\|Hx\|_{\infty},\|A^Ty\|_{\infty},\|C^Tz\|_{\infty},\|g\|_{\infty}), \\\
        \|Ax-b\|_{\infty} \leq \epsilon_{\text{abs}} +\epsilon_{\text{rel}}\max(\|Ax\|_{\infty},\|b\|_{\infty}), \\\
        \|[Cx-u]_+\|_{\infty}\leq \epsilon_{\text{abs}} +\epsilon_{\text{rel}}\max(\|Cx\|_{\infty},\|u\|_{\infty}). \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

It is important to note that this stopping criterion on primal and dual residuals is not enough to guarantee that the returned solution satisfies all \eqref{qp:kkt} conditions. Indeed, as the problem has affine constraints and the objective is quadratic and convex, then as soon as the primal or the dual problem is feasible, strong duality holds (see, e.g., [Theorem 2](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture8.pdf) from L. El Ghaoui's lesson) and to satisfy all optimality conditions we need to add a third criterion on the *duality gap* \f$r_g\f$:

$$\begin{equation}\label{eq:approx_dg_sol}
\begin{aligned}
        r_g := | x^T H x + g^T x + b^T y + u^T [z]_+ + l^T [z]_- | \leq \epsilon^{\text{gap}}_{\text{abs}} + \epsilon^{\text{gap}}_{\text{rel}} \max(\|x^T H x\|, \|g^T x\|, \|b^T y\|, \|u^T [z]_+\|, \|l^T [z]_-\|), \\
\end{aligned}
\end{equation}$$

where \f$[z]_+\f$ and \f$[z]_-\f$ stand for the z projection onto the positive and negative orthant. ProxQP provides the ``check_duality_gap`` option to include this duality gap in the stopping criterion. Note that it is disabled by default, as other solvers don't check this criterion in general. Enable this option if you want a stronger guarantee that your solution is optimal. ProxQP will then check the same termination condition as SCS (for more details see, e.g., SCS's [optimality conditions checks](https://www.cvxgrp.org/scs/algorithm/index.html#optimality-conditions) as well as [section 7.2](https://doi.org/10.1137/20M1366307) in the corresponding paper). The absolute and relative thresholds \f$\epsilon^{\text{gap}}_{\text{abs}}, \epsilon^{\text{gap}}_{\text{rel}}\f$ for the duality gap can differ from those \f$\epsilon_{\text{abs}}, \epsilon_{\text{rel}}\f$ for residuals because, contrary to residuals which result from an infinite norm, the duality gap scales with the square root of the problem dimension (thus it is numerically harder to achieve a given duality gap for larger problems). A recommended choice is \f$\epsilon^{\text{gap}}_{\text{abs}} = \epsilon_{\text{abs}} \sqrt{\max(n, n_{\text{eq}}, n_{\text{ineq}})}\f$. Note finally that meeting all residual and duality-gap criteria can be difficult for ill-conditioned problems.

Finally, note that ProxQP has a specific feature for handling primal infeasibility. More precisely, if the problem appears to be primal infeasible, it will solve the closest primal feasible problem in \f$\ell_2\f$ sense, and (x,y,z) will satisfy.

$$\begin{equation}\label{eq:approx_closest_qp_sol_rel}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|Hx+g+A^Ty+C^Tz\|_{\infty} \leq  \epsilon_{\text{abs}} + \epsilon_{\text{rel}}\max(\|Hx\|_{\infty},\|A^Ty\|_{\infty},\|C^Tz\|_{\infty},\|g\|_{\infty}), \\\
        \|A^\top(Ax-b)+C^\top[Cx-u]_+\|_{\infty} \leq \|A^\top 1 + C^\top 1\|_{\infty}*\epsilon_{\text{abs}} +\epsilon_{\text{rel}}\max(\|Ax\|_{\infty},\|b\|_{\infty}), \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

You can find more details on these subjects (and how to activate this feature with ProxQP) in the subsections describing the Settings and Results classes. You can also find more technical references with [this work](https://hal.science/hal-01057577).

\section OverviewAPIstructure ProxQP unified API for dense and sparse backends

ProxQP algorithm is implemented in two versions specialized for dense and sparse matrices. One simple and unified API has been designed for loading dense and sparse backends. Concretely, it contains three methods:
* init : for initializing the QP model, along with some parameters,
* solve : for solving the QP problem,
* update : for updating some parameters of the QP model.

In what follows, we will make several examples to illustrate how to use this API in C++ and Python. Some subtle differences exist, nevertheless, between the dense and sparse backends, and we will point them out when needed. We will also present all solver's possible settings and show where the results are stored. We will then give some recommendations about which backend to use, considering your needs.

\subsection OverviewAPI The API structure

When creating a Qp object in C++ or Python, it automatically contains the following sub-classes:
* model: a class storing the QP problem model which we want to solve,
* results: a class storing the main solver's results,
* settings: a class with all solver's settings,
* work: a class (not exposed in Python), with auxiliary variables the solver uses for its subroutines.

For loading ProxQP with the dense backend, it is as simple as the following code below:

<table class="manual">
  <tr>
    <th>examples/cpp/loading_dense_qp.cpp</th>
    <th>examples/python/loading_dense_qp.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include loading_dense_qp.cpp
    </td>
    <td valign="top">
      \include loading_dense_qp.py
    </td>
  </tr>
</table>

The dimensions of the problem (i.e., n is the dimension of primal variable x, n_eq the number of equality constraints, and n_in the number of inequality constraints) are used for allocating the space needed for the Qp object. The dense Qp object is templated by the floating precision of the QP model (in the example above in C++ a double precision). Note that for the model to be valid, the primal dimension (i.e., n) must be strictly positive. If it is not the case, an assertion will be raised precising this issue.

The dense backend also has a specific feature for efficiently handling box inequality constraints. To benefit from it, constructors are overloaded as follows:

<table class="manual">
  <tr>
    <th>examples/cpp/loading_dense_qp_with_box_constraints.cpp</th>
    <th>examples/python/loading_dense_qp_with_box_constraints.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include loading_dense_qp_with_box_constraints.cpp
    </td>
    <td valign="top">
      \include loading_dense_qp_with_box_constraints.py
    </td>
  </tr>
</table>

Furthermore, the dense version of ProxQP has two different backends with different advantages:
* PrimalDualLDLT: it factorizes a regularized version of the KKT system and benefits from great accuracy and stability. Nevertheless, if the primal dimension (i.e., the one of x) is far smaller than the dimensions of the constraints, it will be slower than PrimalLDLT backend.
* PrimalLDLT: it factorizes at the beginning the matrix $$H+\rho I+\frac{1}{\mu_{eq}} A^\top A$$ and goes on then with rank one updates. It is less accurate than PrimalDualLDLT backend, but it will be far quicker if it happens that the primal dimension is much smaller than the ones of the constraints.

The QP constructor uses, by default, an automatic choice for deciding which backend suits a priori bests user's needs. It is based on a heuristic comparing a priori computational complexity of each backend. However, if you have more insights into your needs (e.g., accuracy specifications, primal dimension is known to be far larger than the one of the constraints etc.), we encourage you to specify directly in the constructor which backend to use. It is as simple as the following:

<table class="manual">
  <tr>
    <th>examples/cpp/loading_dense_qp_with_different_backend_choice.cpp</th>
    <th>examples/python/loading_dense_qp_with_different_backend_choice.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include loading_dense_qp_with_different_backend_choice.cpp
    </td>
    <td valign="top">
      \include loading_dense_qp_with_different_backend_choice.py
    </td>
  </tr>
</table>

For loading ProxQP with the sparse backend, they are two possibilities:
* one can use as before the dimensions of the QP problem (i.e., n, n_eq and n_in)
* or one can use the sparsity structure of the matrices defining the QP problem. More precisely, if H designs the quadratic cost of the model, A the equality constraint matrix, and C the inequality constraint matrix, one can pass in entry a boolean mask of these matrices (i.e., matrices with true value when one entry is non zero) for initializing the Qp object.

<table class="manual">
  <tr>
    <th>examples/cpp/loading_sparse_qp.cpp</th>
    <th>examples/python/loading_sparse_qp.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include loading_sparse_qp.cpp
    </td>
    <td valign="top">
      \include loading_sparse_qp.py
    </td>
  </tr>
</table>

The sparse Qp object is templated by the floating precision of the QP model (in the example above, a double precision) and the integer precision used for the different types of non-zero indices used (for the associated sparse matrix representation used).

\subsection explanationInitMethod The init method

Once you have defined a Qp object, the init method enables you to set up the QP problem to be solved (the example is given for the dense backend, it is similar for the sparse backend).

<table class="manual">
  <tr>
    <th>examples/cpp/init_dense_qp.cpp</th>
    <th>examples/python/init_dense_qp.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include init_dense_qp.cpp
    </td>
    <td valign="top">
      \include init_dense_qp.py
    </td>
  </tr>
</table>

If you use the specific feature of the dense backend for handling box constraints, the init method uses simply as follows:

<table class="manual">
  <tr>
    <th>examples/cpp/init_dense_qp_with_box.cpp</th>
    <th>examples/python/init_dense_qp_with_box.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include init_dense_qp_with_box.cpp
    </td>
    <td valign="top">
      \include init_dense_qp_with_box.py
    </td>
  </tr>
</table>

Note that with its dense backend, ProxQP solver manipulates matrices in dense representations (in the same spirit, the solver with sparse backend manipulates entries in sparse format). In the example above, the matrices are originally in sparse format and eventually converted into dense format. Note that if some elements of your QP model are not defined (for example, a QP without linear cost or inequality constraints), you can either pass a None argument or a matrix with zero shapes for specifying it. We provide an example below in cpp and Python (for the dense case, it is similar with sparse backend).

<table class="manual">
  <tr>
    <th>examples/cpp/initializing_with_none.cpp</th>
    <th>examples/python/initializing_with_none.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include initializing_with_none.cpp
    </td>
    <td valign="top">
      \include initializing_with_none.py
    </td>
  </tr>
</table>

With the init method, you can also setting-up on the same time some other parameters in the following order:
* compute_preconditioner: a boolean parameter for executing or not the preconditioner. The preconditioner is an algorithm used (for the moment, we use [Ruiz equilibrator](https://cds.cern.ch/record/585592/files/CM-P00040415.pdf)) for reducing the ill-conditioning of the QP problem, and hence speeding-up the solver and increasing its accuracy. It consists mostly of a heuristic involving linear scalings. Note that for very ill-conditioned QP problem, when one asks for a very accurate solution, the unscaling procedure can become less precise (we provide some remarks about this subject in section 6.D of the [following paper](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf)). By default, its value is set to true.
* rho: the proximal step size wrt primal variable. Reducing its value speed-ups convergence wrt the primal variable (but increases as well the ill-conditioning of sub-problems to solve). The minimal value it can take is 1.e-7. By default, its value is set to 1.e-6.
* mu_eq: the proximal step size wrt equality constrained multiplier. Reducing its value speed-ups convergence wrt equality-constrained variables (but increases as well the ill-conditioning of sub-problems to solve). The minimal value it can take is 1.e-9. By default, its value is set to 1.e-3.
* mu_in: the proximal step size wrt inequality constrained multiplier. Reducing its value speed-ups convergence wrt inequality-constrained variable (but increases as well the ill-conditioning of sub-problems to solve). The minimal value it can take is 1.e-9. By default, its value is set to 1.e-1.

We provide below one example in C++ and Python.

<table class="manual">
  <tr>
    <th>examples/cpp/init_dense_qp_with_other_options.cpp</th>
    <th>examples/python/init_dense_qp_with_other_options.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include init_dense_qp_with_other_options.cpp
    </td>
    <td valign="top">
      \include init_dense_qp_with_other_options.py
    </td>
  </tr>
</table>

Furthermore, some settings must be defined before the init method takes effect. For example, if you want the solver to compute the runtime properly (the sum of the setup time and the solving time), you must set this option before the init method (which is part of the setup time). We provide below an example.

<table class="manual">
  <tr>
    <th>examples/cpp/init_dense_qp_with_timings.cpp</th>
    <th>examples/python/init_dense_qp_with_timings.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include init_dense_qp_with_timings.cpp
    </td>
    <td valign="top">
      \include init_dense_qp_with_timings.py
    </td>
  </tr>
</table>

\subsection explanationSolveMethod The solve method

Once you have defined a Qp object and initialized it with a model, the solve method enables you solving the QP problem. The method is overloaded with two modes considering whether you provide or not a warm start to the method. We give below two examples (for the dense backend, with the sparse one it is similar).

<table class="manual">
  <tr>
    <th>examples/cpp/solve_dense_qp.cpp</th>
    <th>examples/python/solve_dense_qp.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include solve_dense_qp.cpp
    </td>
    <td valign="top">
      \include solve_dense_qp.py
    </td>
  </tr>
</table>

Before ending this section, we will talk about how to activate some other settings before launching the solve method. To do so, you only need to define your desired settings (for example, the stopping criterion accuracy threshold eps_abs, or the verbose option) after initializing the Qp object. They will then be taken into account only if there are set before the solve method (otherwise, they will be taken into account when the next solve or update method is called). The full description of all the settings is provided in a dedicated section below. Here we just give an example to illustrate the mentioned notion above.

<table class="manual">
  <tr>
    <th>examples/cpp/solve_dense_qp_with_setting.cpp</th>
    <th>examples/python/solve_dense_qp_with_setting.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include solve_dense_qp_with_setting.cpp
    </td>
    <td valign="top">
      \include solve_dense_qp_with_setting.py
    </td>
  </tr>
</table>

\subsection explanationUpdateMethod The update method

The update method is used to update the model or a parameter of the problem, as for the init method. We provide below an example for the dense case.

<table class="manual">
  <tr>
    <th>examples/cpp/update_dense_qp.cpp</th>
    <th>examples/python/update_dense_qp.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include update_dense_qp.cpp
    </td>
    <td valign="top">
      \include update_dense_qp.py
    </td>
  </tr>
</table>
Contrary to the init method, the compute_preconditioner boolean parameter becomes update_preconditioner, which enables you to keep the previous preconditioner (if set to false) to equilibrate the new updated problem, or to re-compute the preconditioner with the new values of the problem (if set to true). By default, the update_preconditioner parameter is set to false.

The major difference between the dense and sparse API is that in the sparse case only if you change the matrices of the model, the update will take effect only if the matrices have the same sparsity structure (i.e., the non-zero values are located at the same place). Hence, if the matrices have a different sparsity structure, you must create a new Qp object to solve the new problem. We provide an example below.

<table class="manual">
  <tr>
    <th>examples/cpp/update_sparse_qp.cpp</th>
    <th>examples/python/update_sparse_qp.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include update_sparse_qp.cpp
    </td>
    <td valign="top">
      \include update_sparse_qp.py
    </td>
  </tr>
</table>


Finally, if you want to change your initial guess option when updating the problem, you must change it in the setting before the update takes effect for the next solve (otherwise, it will keep the previous one set). It is important, especially for the WARM_START_WITH_PREVIOUS_RESULT initial guess option (set by default in the solver). Indeed, in this case, if no matrix is updated, the workspace keeps the previous factorization in the update method, which adds considerable speed up for the next solving. We provide below an example in the dense case.

<table class="manual">
  <tr>
    <th>examples/cpp/update_dense_qp_ws_previous_result.cpp</th>
    <th>examples/python/update_dense_qp_ws_previous_result.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include update_dense_qp_ws_previous_result.cpp
    </td>
    <td valign="top">
      \include update_dense_qp_ws_previous_result.py
    </td>
  </tr>
</table>

Note that one subsection is dedicated to the different initial guess options below.

\section OverviewSettings The settings subclass

In this section, you will find first the solver's settings and then a subsection detailing the different initial guess options.

\subsection OverviewAllSettings The solver's settings

In this table, you have the three columns from left to right: the name of the setting, its default value, and then a short description of it.

| Setting                             | Default value                      | Description
| ----------------------------------- | ---------------------------------- | -----------------------------------------
| eps_abs                             | 1.E-5                              | Asbolute stopping criterion of the solver.
| eps_rel                             | 0                                  | Relative stopping criterion of the solver.
| check_duality_gap                   | False                              | If set to true, include the duality gap in absolute and relative stopping criteria.
| eps_duality_gap_abs                 | 1.E-4                              | Asbolute duality-gap stopping criterion of the solver.
| eps_duality_gap_rel                 | 0                                  | Relative duality-gap stopping criterion of the solver.
| VERBOSE                             | False                              | If set to true, the solver prints information at each loop.
| default_rho                         | 1.E-6                              | Default rho parameter of result class (i.e., for each initial guess, except WARM_START_WITH_PREVIOUS_RESULT, after a new solve or update, the solver initializes rho to this value).
| default_mu_eq                       | 1.E-3                              | Default mu_eq parameter of result class (i.e., for each initial guess, except WARM_START_WITH_PREVIOUS_RESULT, after a new solve or update, the solver initializes mu_eq to this value).
| default_mu_in                       | 1.E-1                              | Default mu_in parameter of result class (i.e., for each initial guess, except WARM_START_WITH_PREVIOUS_RESULT, after a new solve or update, the solver initializes mu_in to this value).
| compute_timings                     | False                              | If set to true, timings will be computed by the solver (setup time, solving time, and run time = setup time + solving time).
| max_iter                            | 10.000                             | Maximal number of authorized outer iterations.
| max_iter_in                         | 1500                               | Maximal number of authorized inner iterations.
| initial_guess                       | EQUALITY_CONSTRAINED_INITIAL_GUESS | Sets the initial guess option for initilizing x, y and z.
| mu_min_eq                           | 1.E-9                              | Minimal authorized value for mu_eq.
| mu_min_in                           | 1.E-8                              | Minimal authorized value for mu_in.
| mu_update_factor                    | 0.1                                | Factor used for updating mu_eq and mu_in.
| eps_primal_inf                      | 1.E-4                              | Threshold under which primal infeasibility is detected.
| eps_dual_inf                        | 1.E-4                              | Threshold under which dual infeasibility is detected.
| update_preconditioner               | False                              | If set to true, the preconditioner will be re-derived with the update method, otherwise it uses previous one.
| compute_preconditioner              | True                               | If set to true, the preconditioner will be derived with the init method.
| alpha_bcl                           | 0.1                                | alpha parameter of the BCL algorithm.
| beta_bcl                            | 0.9                                | beta parameter of the BCL algorithm.
| refactor_dual_feasibility_threshold | 1.E-2                              | Threshold above which refactorization is performed to change rho parameter.
| refactor_rho_threshold              | 1.E-7                              | New rho parameter used if the refactor_dual_feasibility_threshold condition has been satisfied.
| cold_reset_mu_eq                    | 1./1.1                             | Value used for cold restarting mu_eq.
| cold_reset_mu_in                    | 1./1.1                             | Value used for cold restarting mu_in.
| nb_iterative_refinement             | 10                                 | Maximal number of iterative refinements.
| eps_refact                          | 1.E-6                              | Threshold value below which the Cholesky factorization is refactorized factorization in the iterative refinement loop.
| safe_guard                          | 1.E4                               | Safeguard parameter ensuring global convergence of the scheme. More precisely, if the total number of iterations is superior to safe_guard, the BCL scheme accepts always the multipliers (hence the scheme is a pure proximal point algorithm).
| preconditioner_max_iter             | 10                                 | Maximal number of authorized iterations for the preconditioner.
| preconditioner_accuracy             | 1.E-3                              | Accuracy level of the preconditioner.
| HessianType                         | Dense                              | Defines the type of problem solved (Dense, Zero, or Diagonal). In case the Zero or Diagonal option is used, the solver exploits the Hessian structure to evaluate the Cholesky factorization efficiently.
| primal_infeasibility_solving        | False                              | If set to true, it solves the closest primal feasible problem if primal infeasibility is detected.
| nb_power_iteration                  | 1000                               | Number of power iteration iteration used by default for estimating H lowest eigenvalue.
| power_iteration_accuracy            | 1.E-6                              | If set to true, it solves the closest primal feasible problem if primal infeasibility is detected.
| primal_infeasibility_solving        | False                              | Accuracy target of the power iteration algorithm for estimating the lowest eigenvalue of H.
| estimate_method_option           | NoRegularization                   | Option for estimating the minimal eigen value of H and regularizing default_rho  default_rho=rho_regularization_scaling*abs(default_H_eigenvalue_estimate). This option can be used for solving non convex QPs.
| default_H_eigenvalue_estimate       | 0.                                 | Default estimate of the minimal eigen value of H.
| rho_regularization_scaling          | 1.5                                | Scaling for regularizing default_rho according to the minimal eigen value of H.

\subsection OverviewInitialGuess The different initial guesses

The solver has five different possible initial guesses for warm starting or not the initial iterate values:
* NO_INITIAL_GUESS,
* EQUALITY_CONSTRAINED_INITIAL_GUESS,
* WARM_START_WITH_PREVIOUS_RESULT,
* WARM_START,
* COLD_START_WITH_PREVIOUS_RESULT.

The different options will be commented below in the introduced order above.

\subsubsection OverviewNoInitialGuess No initial guess

If set to this option, the solver will start with no initial guess, which means that the initial values of x, y, and z are the 0 vector.

\subsection OverviewEstimatingHminimalEigenValue The different options for estimating H minimal Eigenvalue

The solver environment provides an independent function for estimating the minimal eigenvalue of a dense or sparse symmetric matrix. It is named "estimate_minimal_eigen_value_of_symmetric_matrix". In the sparse case, it uses a power iteration algorithm (with two options: the maximal number of iterations and the accuracy target for the estimate). In the dense case, we provide two options within the struct EigenValueEstimateMethodOption:
* PowerIteration: a power iteration algorithm will be used for estimating H minimal eigenvalue,
* ExactMethod: in this case, an exact method from EigenSolver is used to provide an estimate.

Estimating minimal eigenvalue is particularly usefull for solving QP with non convex quadratics. Indeed, if default_rho is set to a value strictly higher than the minimal eigenvalue of H, then ProxQP is guaranteed for find a local minimum to the problem since it relies on a Proximal Method of Multipliers (for more detail for example this [work](https://arxiv.org/pdf/2010.02653.pdf) providing convergence proof of this property).

More precisely, ProxQP API enables the user to provide for the init or update methods estimate of the minimal eigenvalue of H (i.e., manual_minimal_H_eigenvalue). It the values are not empty, then the values of primal proximal step size rho will be updated according to: rho = rho + abs(manual_minimal_H_eigenvalue). It guarantees that the proximal step-size is larger than the minimal eigenvalue of H and hence to converging towards a local minimum of the QP. We provide below examples in C++ and python for using this feature appropriately with the dense backend (it is similar with the sparse one)

<table class="manual">
  <tr>
    <th>examples/cpp/estimate_nonconvex_eigenvalue.cpp</th>
    <th>examples/python/estimate_nonconvex_eigenvalue.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include estimate_nonconvex_eigenvalue.cpp
    </td>
    <td valign="top">
      \include estimate_nonconvex_eigenvalue.py
    </td>
  </tr>
</table>

\subsubsection OverviewNoInitialGuess No initial guess

If set to this option, the solver will start with no initial guess, which means that the initial values of x, y, and z are the 0 vector.

\subsubsection OverviewEqualityConstrainedInitialGuess Equality constrained initial guess

If set to this option, the solver will solve at the beginning the following system for warm starting x and y.
$$\begin{bmatrix}
H+\rho I & A^T \\\
A & -\mu_{eq} I
\end{bmatrix}
\begin{bmatrix}
x \\\ y
\end{bmatrix}
=
\begin{bmatrix}
-g \\\
b
\end{bmatrix}$$
z stays to 0. In general, this option warm starts well equality constrained QP.

\subsubsection OverviewWarmStartWithPreviousResult Warm start with the previous result

If set to this option, the solver will warm start x, y, and z with the values of the previous problem solved and it will keep all the last parameters of the solver (i.e., proximal step sizes, for example, and the full workspace with the Cholesky factorization etc.). Hence, if the new problem to solve is the same as the previous one, the problem is warm-started at the solution (and zero iteration will be executed).

This option was initially thought to be used in optimal control-like problems when the next problem to be solved is close to the previous one. Indeed, if the problem changes only slightly, it is reasonable to warm start the new problem with the value of the previous one for speeding the whole runtime.

Note, however, that if your update involves new matrices or you decide to change parameters involved in the Cholesky factorization (i.e., the proximal step sizes), then for consistency, the solver will automatically refactorize the Cholesky with these updates (and taking into account the last values of x, y, and z for the active set).

Finally, note that this option is set by default in the solver. At the first solve, as there is no previous results, x, y and z are warm started with the 0 vector value.

\subsubsection OverviewWarmStart Warm start

If set to this option, the solver expects then a warm start at the solve method.

Note, that it is unnecessary to set this option through the command below (for example, in C++) before the update or solve method call.
\code
Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
\endcode

It is sufficient to just add the warm start in the solve method, and the solver will automatically make the setting change internally.

\subsubsection OverviewColdStartWithPreviousResult Cold start with previous result

If set to this option, the solver will warm start x, y and z with the values of the previous problem solved. Contrary to the WARM_START_WITH_PREVIOUS_RESULT option, all other parameters of the solver (i.e., proximal step sizes for example, and the full workspace with the ldlt factorization etc.) are re-set to their default values (hence a factorization is reperformed taking into account of z warm start for the active set, but with default values of proximal step sizes).

This option has also been thought initially for being used in optimal control-like problems when the next problem to be solved is close to the previous one. Indeed, if the problem changes only slightly, it is reasonable to warm start the new problem with the value of the previous one for speeding the whole runtime.

Note finally that at the first solve, as there are no previous results, x, y, and z are warm started with the 0 vector value.

\section OverviewResults The results subclass

The result subclass is composed of the following:
* x: a primal solution,
* y: a Lagrange optimal multiplier for equality constraints,
* z: a Lagrange optimal multiplier for inequality constraints,
* se: the optimal shift in \f$\ell_2\f$ with respect to equality constraints,
* si: the optimal shift in \f$\ell_2\f$ with respect to inequality constraints,
* info: a subclass which containts some information about the solver's execution.

If the solver has solved the problem, the triplet (x,y,z) satisfies:

$$\begin{equation}\label{eq:approx_qp_sol_rel}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|Hx+g+A^Ty+C^Tz\|_{\infty} \leq  \epsilon_{\text{abs}} + \epsilon_{\text{rel}}\max(\|Hx\|_{\infty},\|A^Ty\|_{\infty},\|C^Tz\|_{\infty},\|g\|_{\infty}), \\\
        \|Ax-b\|_{\infty} \leq \epsilon_{\text{abs}} +\epsilon_{\text{rel}}\max(\|Ax\|_{\infty},\|b\|_{\infty}), \\\
        \|[Cx-u]_+\|_{\infty}\leq \epsilon_{\text{abs}} +\epsilon_{\text{rel}}\max(\|Cx\|_{\infty},\|u\|_{\infty}), \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$
accordingly with the parameters eps_abs and eps_rel chosen by the user.

If the problem is primal infeasible and you have enabled the solver to solve the closest feasible problem, then (x,y,z) will satisfy.
$$\begin{equation}\label{eq:approx_closest_qp_sol_rel_bis}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|Hx+g+A^Ty+C^Tz\|_{\infty} \leq  \epsilon_{\text{abs}} + \epsilon_{\text{rel}}\max(\|Hx\|_{\infty},\|A^Ty\|_{\infty},\|C^Tz\|_{\infty},\|g\|_{\infty}), \\\
        \|A^\top(Ax-b)+C^\top[Cx-u]_+\|_{\infty} \leq \|A^\top 1 + C^\top 1\|_{\infty}*\epsilon_{\text{abs}} +\epsilon_{\text{rel}}\max(\|Ax\|_{\infty},\|b\|_{\infty}), \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

(se, si) stands in this context for the optimal shifts in \f$\ell_2\f$ sense which enables recovering a primal feasible problem. More precisely, they are derived such that

\begin{equation}\label{eq:QP_primal_feasible}\tag{QP_feas}
\begin{aligned}
    \min_{x\in\mathbb{R}^{d}} & \quad \frac{1}{2}x^{T}Hx+g^{T}x \\\
    \text{s.t.}&\left\{
    \begin{array}{ll}
         Ax = b+se, \\\
        Cx \leq u+si, \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}\\\
defines a primal feasible problem.

Note that if you use the dense backend and its specific feature for handling box inequality constraints, then the first \f$n_{in}\f$ elements of z correspond to multipliers associated to the linear inequality formed with \f$C\f$ matrix, whereas the last \f$d\f$ elements correspond to multipliers associated to the box inequality constraints (see for example solve_dense_qp.cpp or solve_dense_qp.py).

\subsection OverviewInfoClass The info subclass

In this table you have on the three columns from left to right: the name of the info subclass item, its default value, and then a short description of it.

| Info item                           | Default value                  | Description
| ----------------------------------- | ------------------------------ | -----------------------------------------
| mu_eq                               | 1.E-3                          | Proximal step size wrt equality constraints multiplier.
| mu_in                               | 1.E-1                          | Proximal step size wrt inequality constraints multiplier.
| rho                                 | 1.E-6                          | Proximal step size wrt primal variable.
| iter                                | 0                              | Total number of iterations.
| iter_ext                            | 0                              | Total number of outer iterations.
| mu_updates                          | 0                              | Total number of mu updates.
| rho_updates                         | 0                              | Total number of rho updates.
| status                              | PROXQP_NOT_RUN                 | Status of the solver.
| setup_time                          | 0                              | Setup time (takes into account the equilibration procedure).
| solve_time                          | 0                              | Solve time (takes into account the first factorization).
| run_time                            | 0                              | the sum of the setup time and the solve time.
| objValue                            | 0                              | The objective value to minimize.
| pri_res                             | 0                              | The primal residual.
| dua_res                             | 0                              | The dual residual.


Note finally that when initializing a QP object, by default, the proximal step sizes (i.e., rho, mu_eq, and mu_in) are set up by the default values defined in the Setting class. Hence, when doing multiple solves, if not specified, their values are re-set respectively to default_rho, default_mu_eq, and default_mu_in. A small example is given below in C++ and Python.

<table class="manual">
  <tr>
    <th>examples/cpp/init_with_default_options.cpp</th>
    <th>examples/python/init_with_default_options.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include init_with_default_options.cpp
    </td>
    <td valign="top">
      \include init_with_default_options.py
    </td>
  </tr>
</table>


\subsection OverviewSolverStatus The solver's status

The solver has five status:
* PROXQP_SOLVED: the problem is solved.
* PROXQP_MAX_ITER_REACHED: the maximum number of iterations has been reached.
* PROXQP_PRIMAL_INFEASIBLE: the problem is primal infeasible.
* PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: the closest feasible problem in L2 sense is solved.
* PROXQP_DUAL_INFEASIBLE: the problem is dual infeasible.
* PROXQP_NOT_RUN: the solver has not been run yet.

Infeasibility is detected using the necessary conditions exposed in [section 3.4](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf). More precisely, primal infeasibility is assumed if the following conditions are matched for some non zeros dy and dz (according to the eps_prim_inf variable set by the user):

$$\begin{equation}\label{eq:approx_qp_sol_prim_inf}\tag{PrimalInfeas}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|A^Tdy\|_{\infty} \leq \epsilon_{\text{primal_inf}} \|dy\|_{\infty} , \\\
        b^T dy \leq -\epsilon_{\text{primal_inf}} \|dy\|_{\infty}, \\\
        \|C^Tdz\|_{\infty} \leq  \epsilon_{\text{primal_inf}} \|dz\|_{\infty}, \\\
        u^T [dz]_+ - l^T[-dz]_+\leq -\epsilon_{\text{primal_inf}} \|dz\|_{\infty}. \\\
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

Dual infeasibility is assumed if the following two conditions are matched for some non-zero dx (according to the eps_dual_inf variable set by the user):

$$\begin{equation}\label{eq:approx_qp_sol_dual_inf}\tag{DualInfeas}
\begin{aligned}
    &\left\{
    \begin{array}{ll}
        \|Hdx\|_{\infty} \leq \epsilon_{\text{dual_inf}} \|dx\|_{\infty} , \\\
        g^T dx \leq -\epsilon_{\text{dual_inf}} \|dx\|_{\infty}, \\\
        \|Adx\|_{\infty}\leq \epsilon_{\text{dual_inf}} \|dx\|_{\infty}, \\\
        (Cdz)_i \geq \epsilon_{\text{dual_inf}} \|dx\|_{\infty}, \mbox{ if } u_i = +\infty, \\\
        (Cdz)_i  \leq \epsilon_{\text{dual_inf}} \|dx\|_{\infty}, \mbox{ otherwise }.
    \end{array}
    \right.
\end{aligned}
\end{equation}$$

If the problem turns out to be primal or dual infeasible, then x, y, and z stored in the results class will be the certificate of primal or dual infeasibility. More precisely:
* if the problem is dual infeasible, Qp.results.x will be the certificate dx of dual infeasibility satisfying \eqref{eq:approx_qp_sol_dual_inf} at precisifion Qp.settings.eps_dual_inf specified by the user,
* if the problem is primal infeasible, Qp.results.y and Qp.results.z will be, respectively, the certificates dy and dz of primal infeasibility satisfying \eqref{eq:approx_qp_sol_prim_inf} at precision Qp.settings.eps_primal_inf specified by the user.


\section OverviewWhichBackend Which backend to use?

We have the following generic advice for choosing between the sparse and dense backend. If your problem is not:
* too large (less than some thousand variables),
* and too sparse (a sparsity ratio of your matrices greater than 0.1),

then we recommend using the solver with the dense backend.


The sparsity ratio of matrix A is defined as:

$$ \text{sparsity}(A) = \frac{\text{nnz}(A)}{\text{number}_{\text{row}}(A) * \text{number}_{\text{col}}(A)}, $$

which accounts for the percentage of non-zero elements in matrix A.

\section OverviewBenchmark Some important remarks when computing timings

We first provide some details about what is measured in the setup and solve time of ProxQP, which is of some importance when doing benchmarks with other solvers, as they can measure different things in a feature with a similar name.

Then we conclude this documentation section with some compilation options for ProxQP which can considerably speed up the solver, considering your OS architecture.

\subsection OverviewTimings What do the timings take into account?

An important remark about quadratic programming solver is that they all rely at some point on a factorization matrix algorithm, which constitutes the time bottleneck of the solver (as the factorization has a cubic order of complexity wrt dimension of the matrix to factorize).

Available solvers often have a similar API as the one we propose, with first an "init" method for initializing the model, and then a "solve" method for solving the QP problem. For not biasing the benchmarks, it is important to know where is done the first matrix factorization, as it constitutes a considerable cost. In our API, we have decided to make the first factorization in the solve method, as we consider that factorizing the problem is part of the solving part. Hence in terms of timing:
* results.info.setup_time: measures only the initialization of the model and also the preconditioning procedure (if it has been activated),
* results.info.solve_time: measures everything else (including the first factorization),
* results.info.run_time = results.info.setup_time + results.info.solve_time.

It is important to notice that some other solvers API have made different choices. For example, OSQP measures in the setup time the first factorization of the system (at the time [ProxQP algorithm](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf) was published). Hence we recommend that for benchmarking ProxQP against other solvers, you should compare ProxQP runtime against the other solvers' runtime (i.e., everything from what constitutes their setup to their solve method). Otherwise, the benchmarks won't take into comparable account timings.

\subsection OverviewArchitectureOptions Architecture options when compiling ProxSuite

We highly encourage you to enable the vectorization of the underlying linear algebra for the best performance. You just need to activate the cmake option `BUILD_WITH_SIMD_SUPPORT=ON`, like:

\code
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_WITH_SIMD_SUPPORT=ON
make
make install
\endcode

ProxQP can be compiled more precisely with two SIMD instructions options for x86 instruction set architectures: [AVX-2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) and [AVX-512](https://en.wikipedia.org/wiki/AVX-512). They can considerably enhance the speed of ProxQP, and we encourage you to use them if your OS is compatible with them. For your information, our benchmarks for [ProxQP algorithm](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf) have been realised with AVX-2 compilation option.
