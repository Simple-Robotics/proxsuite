## ProxQP solve function without API

ProxQP solves convex quadratic programs, which consists in minimizing a convex quadratic cost under some linear constraints. It is mathematically described as:

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
H is a real symmetric positive semi-definite matrix. d is the problem dimension (i.e., the number of primal variables), while n_eq and n_in are the numbers of equality and inequality constraints respectively.

For linearly constrained convex optimization problems such as \eqref{eq:QP}, strong duality holds and the associated KKT conditions are necessary and sufficient for ensuring a primal-dual point (x,y,z) to be optimal (see, e.g.,[Section 5.2.3](https://web.stanford.edu/~boyd/cvxbook/)} and [Section 2, page 5](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf) for more details).
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

where the last equation involves the Hadamard product (i.e., for two vectors u and v, the Hadamard product is the vector whose ith entry is u_i v_i).

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

The infite norm is preferred to the L2 norm as it is independent of the problem dimensions. It is also common to consider relative convergence criteria for early-stopping, as absolute targets might not bet reached due to numerical issues. ProxQP provides it in a similar way as OSQP (for more details see, e.g., OSQP's [convergence](https://osqp.org/docs/solver/index.html#convergence) criteria or [section 3.4](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf) in the corresponding paper). Hence more generally the following stopping criterion can be used:

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

It is important to note that this stopping criterion on primal and dual residuals is not enough to guarantee that the returned solution satisfies all \eqref{qp:kkt} conditions. Indeed, as the problem has affine constraints and the objective is quadratic and convex, then as soon as the primal or the dual problem is feasible, then strong duality holds (see e.g., [Theorem 2](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture8.pdf) from L. El Ghaoui's lesson) and to satisfy all optimality conditions we need to add a third criterion on the *duality gap* \f$r_g\f$:

$$\begin{equation}\label{eq:approx_dg_sol}
\begin{aligned}
        r_g := | x^T H x + g^T x + b^T y + u^T [z]_+ + l^T [z]_- | \leq \epsilon^{\text{gap}}_{\text{abs}} + \epsilon^{\text{gap}}_{\text{rel}} \max(\|x^T H x\|, \|g^T x\|, \|b^T y\|, \|u^T [z]_+\|, \|l^T [z]_-\|), \\
\end{aligned}
\end{equation}$$

where \f$[z]_+\f$ and \f$[z]_-\f$ stand for the projection of z onto the positive and negative orthant. ProxQP provides the ``check_duality_gap`` option to include this duality gap in the stopping criterion. Note that it is disabled by default, as other solvers don't check in general this criterion. Enable this option if you want a stronger guarantee that your solution is optimal. ProxQP will then check the same termination condition as SCS (for more details see, e.g., SCS's [optimality conditions checks](https://www.cvxgrp.org/scs/algorithm/index.html#optimality-conditions) as well as [section 7.2](https://doi.org/10.1137/20M1366307) in the corresponding paper). The absolute and relative thresholds \f$\epsilon^{\text{gap}}_{\text{abs}}, \epsilon^{\text{gap}}_{\text{rel}}\f$ for the duality gap can differ from those \f$\epsilon_{\text{abs}}, \epsilon_{\text{rel}}\f$ for residuals because, contrary to residuals which result from an infinite norm, the duality gap scales with the square root of the problem dimension (thus it is numerically harder to achieve a given duality gap for larger problems). A recommended choice is \f$\epsilon^{\text{gap}}_{\text{abs}} = \epsilon_{\text{abs}} \sqrt{\max(n, n_{\text{eq}}, n_{\text{ineq}})}\f$. Note finally that meeting all residual and duality-gap criteria can be difficult for ill-conditioned problems.

\section OverviewAsingleSolveFunction A single solve function for dense and sparse backends

If if you don't want to pass through [ProxQP API](2-ProxQP_api.md), it is also possible to use one single solve function. We will show how to do so with examples.

You just need to call a "solve" function with in entry the model of the convex QP you want to solve. We show you below examples in C++ and python for ProxQP sparse and dense backends. Note that the sparse and dense solvers take respectivaly entries in sparse and dense formats. Note finally that the dense backend benefits from a feature enabling it to handle more efficiently box inequality constraints. We provide an example below as well of how using it (you can find more details about it in [ProxQP API](2-ProxQP_api.md)).

<table class="manual">
  <tr>
    <th>examples/cpp/solve_without_api.cpp</th>
    <th>examples/python/solve_without_api.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include solve_without_api.cpp
    </td>
    <td valign="top">
      \include solve_without_api.py
    </td>
  </tr>
</table>

The results of the solve call are available in the result object, which structure is described in [ProxQP API with examples](./2-ProxQP_api.md) section (at "The results subclass" subsection).

Different options are available for the solve function. In the table below you have the list of the different parameters that can be specified in the solve function (without the model of the problem to solve). We precise from left to right their name, their default value, and a short description of it.

| Option                              | Default value                                   | Description
| ----------------------------------- | ------------------------------------------------| -----------------------------------------
| x                                   | Value of the EQUALITY_CONSTRAINED_INITIAL_GUESS | Warm start value for the primal variable.
| y                                   | Value of the EQUALITY_CONSTRAINED_INITIAL_GUESS | Warm start value for the dual Lagrange multiplier for equality constraints.
| z                                   | 0                                               | Warm start value for the dual Lagrange multiplier for inequality constraints.
| eps_abs                             | 1.E-5                                           | Asbolute stopping criterion of the solver.
| eps_rel                             | 0                                               | Relative stopping criterion of the solver.
| check_duality_gap                   | False                                           | If set to true, include the duality gap in absolute and relative stopping criteria.
| eps_duality_gap_abs                 | 1.E-4                                           | Asbolute duality-gap stopping criterion of the solver.
| eps_duality_gap_rel                 | 0                                               | Relative duality-gap stopping criterion of the solver.
| mu_eq                               | 1.E-3                                           | Proximal step size wrt equality constraints multiplier.
| mu_in                               | 1.E-1                                           | Proximal step size wrt inequality constraints multiplier.
| rho                                 | 1.E-6                                           | Proximal step size wrt primal variable.
| VERBOSE                             | False                                           | If set to true, the solver prints information at each loop.
| compute_preconditioner              | True                                            | If set to true, the preconditioner will be derived.
| compute_timings                     | False                                           | If set to true, timings in microseconds will be computed by the solver (setup time, solving time, and run time = setup time + solving time).
| max_iter                            | 10.000                                          | Maximal number of authorized outer iterations.
| initial_guess                       | EQUALITY_CONSTRAINED_INITIAL_GUESS              | Sets the initial guess option for initilizing x, y and z.

All other settings are set to their default values detailed in [ProxQP API with examples](2-ProxQP_api.md) differents subsections of "The settings subclass" section.

Note that contrary to ProxQP API, the default value of the initial guess is the "EQUALITY_CONSTRAINED_INITIAL_GUESS" option. Indeed, there is no meaning in using the "WARM_START_WITH_PREVIOUS_RESULT" option as after the solve call, it is not possible to warm start any Qp object with previous results. Hence, the only meaningful initial guess options are:
* EQUALITY_CONSTRAINED_INITIAL_GUESS,
* NO_INITIAL_GUESS,
* or a WARM_START explicitly provided by the user.

For the latter case, It is not necessary to specify the WARM_START initial guess in the solve function. It is sufficient to just add the warm start x, y and z in the solve function, and the solver will automatically make the setting change internally.

Finally, note that in C++, if you want to change one option, the order described above in the table matter. Any intermediary option not changed must be let to the std::nullopt value. In Python, you can just specify the name of the option you want to change in any order. We give an example below.


<table class="manual">
  <tr>
    <th>examples/cpp/solve_without_api_and_option.cpp</th>
    <th>examples/python/solve_without_api_and_option.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include solve_without_api_and_option.cpp
    </td>
    <td valign="top">
      \include solve_without_api_and_option.py
    </td>
  </tr>
</table>

Note that if some elements of your QP model are not defined (for example a QP without linear cost or inequality constraints), you can either pass a None argument, or a matrix with zero shape for specifying it. We provide an example below in cpp and python (for the dense case, it is similar with sparse backend).

<table class="manual">
  <tr>
    <th>examples/cpp/initializing_with_none_without_api.cpp</th>
    <th>examples/python/initializing_with_none_without_api.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include initializing_with_none_without_api.cpp
    </td>
    <td valign="top">
      \include initializing_with_none_without_api.py
    </td>
  </tr>
</table>

Finally, note that you can also you ProxQP for solving QP with non convex quadratic. For doing so, you just need to provide to the solve function an estimate of the smallest eigenvalue of the quadratic cost H. The solver environment provides an independent function for estimating the minimal eigenvalue of a dense or sparse symmetric matrix. It is named "estimate_minimal_eigen_value_of_symmetric_matrix". You can find more details in [ProxQP API with examples](2-ProxQP_api.md) about the different other settings that can be used for setting other related parameters (e.g., for using a Power Iteration algorithm).
