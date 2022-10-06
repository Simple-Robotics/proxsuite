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

In practice, we look for a triplet (x,y,z) satisfying these optimality conditions \eqref{qp:kkt} up to a certain level of absolute accuracy (dependent of the application), leading us to the following natural absolute stopping criterion:

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

The infite norm is preferred to the L2 norm as it is independent of the problem dimensions. It is also common to consider relative convergence criteria for early-stopping, as absolute targets might not bet reached due to numerical issues. ProxQP provides it in a similar way as OSQP (for more details see, e.g., [section 3.4](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf)). Hence more generally the following stopping criterion can be used.

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

\section OverviewAsingleSolveFunction A single solve function for dense and sparse backends

If if you don't want to pass through [ProxQP API](2-ProxQP_api.md), it is also possible to use one single solve function. We will show how to do so with examples.

You just need to call a "solve" function with in entry the model of the convex QP you want to solve. We show you below examples in C++ and python for ProxQP sparse and dense backends. Note that the sparse and dense solvers take respectivaly entries in sparse and dense formats.

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
| eps_abs                             | 1.E-3                                           | Asbolute stopping criterion of the solver.
| eps_rel                             | 0                                               | Relative stopping criterion of the solver.
| mu_eq                               | 1.E-3                                           | Proximal step size wrt equality constraints multiplier.
| mu_in                               | 1.E-1                                           | Proximal step size wrt inequality constraints multiplier.
| rho                                 | 1.E-6                                           | Proximal step size wrt primal variable.
| VERBOSE                             | False                                           | If set to true, the solver prints information at each loop.
| compute_preconditioner              | True                                            | If set to true, the preconditioner will be derived.
| compute_timings                     | True                                            | If set to true, timings will be computed by the solver (setup time, solving time, and run time = setup time + solving time).
| max_iter                            | 1.E4                                            | Maximal number of authorized outer iterations.
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