## ProxQP API with examples

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

\section OverviewAPIstructure ProxQP unified API for dense and sparse backends

ProxQP algorithm is implemented in two versions specialized for dense and sparse matrices. One simple and unified API has been designed for loading the dense and sparse backends. Concretely, it contains three methods:
* init : for initializing the QP model, along with some parameters,
* solve : for solving the QP problem,
* update : for updating some parameters of the QP model.

In what follows, we will make several examples for illustrating how to use this API in C++ and python. Some subttle differences exist nevertheless between the dense and sparse backends, and we will point them out when needed. We will also present all solver's possible settings and show where are stored the results. We will then give some recommandations about which backend to use considering your needs.

\subsection OverviewAPI The API structure

When creating a Qp object in C++ or python, it contains automatically the following sub-classes:
* model: a class storing the QP problem model which we want to solve,
* results: a class storing the main solver's results,
* settings: a class with all solver's settings,
* work: a class (not exposed in python), with auxiliary variables used by the solver for its subroutines.

For loading ProxQP with dense backend it is as simple as the following code below:

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

The dimensions of the problem (i.e., n is the dimension of primal variable x, n_eq the number of equality constraints, and n_in the number of inequality constraints) are used for allocating the space needed for the Qp object. The dense Qp object is templated by the floatting precision of the QP model (in the example above in C++ a double precision). Note that for model to be valid, the primal dimension (i.e., n) must be strictly positive. If it is not the case an assertion will be raised precising this issue.

For loading ProxQP with sparse backend they are two possibilities:
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

The sparse Qp object is templated by the floatting precision of the QP model (in the example above a double precision), and the integer precision used for the different types of non zero indices used (for the associated sparse matrix representation used).

\subsection explanationInitMethod The init method

Once you have defined a Qp object, the init method enables you setting up the QP problem to be solved (the example is given for the dense backend, it is similar for sparse backend).

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

Note that with its dense backend, ProxQP solver manipulates matrices in dense representations (in the same spirit, the solver with sparse backend manipulates entries in sparse format). In the example above the matrices are originally in sparse format, and eventually converted into dense format. Note that if some elements of your QP model are not defined (for example a QP without linear cost or inequality constraints), you can either pass a None argument, or a matrix with zero shape for specifying it. We provide an example below in cpp and python (for the dense case, it is similar with sparse backend).

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
* compute_preconditioner: a boolean parameter for executing or not the preconditioner. The preconditioner is an algorithm used (for the moment we use [Ruiz equilibrator](https://cds.cern.ch/record/585592/files/CM-P00040415.pdf)) for reducing the ill-conditioning of the QP problem, and hence speeding-up the solver and increasing its accuracy. It consists mostly of an heuristic involving linear scalings. Note that for very ill-conditioned QP problem, when one asks for a very accurate solution, the unscaling procedure can become less precise (we provide some remarks about this subject in section 6.D of the [following paper](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf)). By default its value is set to true.
* rho: the proximal step size wrt primal variable. Reducing its value speed-ups convergence wrt primal variable (but increases as well ill-conditioning of sub-problems to solve). The minimal value it can take is 1.e-7. By default its value is set to 1.e-6.
* mu_eq: the proximal step size wrt equality constrained multiplier. Reducing its value speed-ups convergence wrt equality constrained variable (but increases as well ill-conditioning of sub-problems to solve). The minimal value it can take is 1.e-9. By default its value is set to 1.e-3.
* mu_in: the proximal step size wrt inequality constrained multiplier. Reducing its value speed-ups convergence wrt inequality constrained variable (but increases as well ill-conditioning of sub-problems to solve). The minimal value it can take is 1.e-9. By default its value is set to 1.e-1.

We provide below one example in C++ and python.

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

Furthermore, some settings must be defined before the init method to take effect. For example, if you want the solver to compute the runtime properly (the sum of the setup time and the solving time), you must set this option before the init method (which is part of the setup time). We provide below an example.

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

Before ending this section, we will talk about how to activate some other settings before launching the solve method. To do so, you only need to define your desired settings (for example, the stopping criterion accuracy threshold eps_abs, or the verbose option) after initializing the Qp object. They will be then taken into account only if there are set before the solve method (otherwise, they will be taken into account when a next solve or update method is called). The full description of all the settings is provided at a dedicated section below. Here we just give an example to illustrate the mentioned notion above.

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
Contrary to the init method, the compute_preconditioner boolean parameter becomes update_preconditioner, which enables you keeping the previous preconditioner (if set to false) to equilibrate the new updated problem, or to re-compute the preconditioner with the new values of the problem (if set to true). By default the update_preconditioner parameter is set to true.

The major difference between the dense and sparse API is that in the sparse case only, if you change matrices of the model, the update will take effect only if the matrices have the same sparsity structure (i.e., the non zero values are located at the same place). Hence, if the matrices have a different sparsity structure, you must create a new Qp object to solve the new problem. We provide an example below.

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


Finally, if you want to change your initial guess option when updating the problem, you must change it in the setting before the update to take effect for the next solve (otherwise it will keep the previous one set). It is important especially for the WARM_START_WITH_PREVIOUS_RESULT initial guess option (set by default in the solver). Indeed, in this case, if no matrix is updated, the workspace keeps the previous factorization in the update method, which adds considerable speed-up for the next solve. We provide below an example in the dense case.

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

In this section you will find first the solver's settings, and then a subsection detailing the different initial guess options.

\subsection OverviewAllSettings The solver's settings

In this table you have on the three columns from left to right: the name of the setting, its default value and then a short description of it.

| Setting                             | Default value                  | Description
| ----------------------------------- | ------------------------------ | -----------------------------------------
| eps_abs                             | 1.E-3                          | Asbolute stopping criterion of the solver.
| eps_rel                             | 0                              | Relative stopping criterion of the solver.
| VERBOSE                             | False                          | If set to true, the solver prints information at each loop.
| default_rho                         | 1.E-6                          | Default rho parameter of result class (i.e., for each initial guess, except WARM_START_WITH_PREVIOUS_RESULT, after a new solve or update, the solver initializes rho to this value).
| default_mu_eq                       | 1.E-3                          | Default mu_eq parameter of result class (i.e., for each initial guess, except WARM_START_WITH_PREVIOUS_RESULT, after a new solve or update, the solver initializes mu_eq to this value).
| default_mu_in                       | 1.E-1                          | Default mu_in parameter of result class (i.e., for each initial guess, except WARM_START_WITH_PREVIOUS_RESULT, after a new solve or update, the solver initializes mu_in to this value).
| compute_timings                     | True                           | If set to true, timings will be computed by the solver (setup time, solving time, and run time = setup time + solving time).
| max_iter                            | 1.E4                           | Maximal number of authorized outer iterations.
| max_iter_in                         | 1500                           | Maximal number of authorized inner iterations.
| initial_guess                       | WARM_START_WITH_PREVIOUS_RESULT| Sets the initial guess option for initilizing x, y and z.
| mu_min_eq                           | 1.E-9                          | Minimal authorized value for mu_eq.
| mu_min_in                           | 1.E-8                          | Minimal authorized value for mu_in.
| mu_update_factor                    | 0.1                            | Factor used for updating mu_eq and mu_in.
| eps_primal_inf                      | 1.E-4                          | Threshold under which primal infeasibility is detected.
| eps_dual_inf                        | 1.E-4                          | Threshold under which dual infeasibility is detected.
| update_preconditioner               | True                           | If set to true, the preconditioner will be re-derived with the update method, otherwise it uses previous one.
| compute_preconditioner              | True                           | If set to true, the preconditioner will be derived with the init method.
| alpha_bcl                           | 0.1                            | alpha parameter of the BCL algorithm.
| beta_bcl                            | 0.9                            | beta parameter of the BCL algorithm.
| refactor_dual_feasibility_threshold | 1.E-2                          | Threshold above which refactorization is performed to change rho parameter.
| refactor_rho_threshold              | 1.E-7                          | New rho parameter used if the refactor_dual_feasibility_threshold condition has been satisfied.
| cold_reset_mu_eq                    | 1./1.1                         | Value used for cold restarting mu_eq.
| cold_reset_mu_in                    | 1./1.1                         | Value used for cold restarting mu_in.
| nb_iterative_refinement             | 10                             | Maximal number of iterative refinements.
| eps_refact                          | 1.E-6                          | Threshold value below which the ldlt is refactorized factorization in the iterative refinement loop.
| safe_guard                          | 1.E4                           | Safeguard parameter ensuring global convergence of the scheme. More precisely, if the total number of iteration is superior to safe_guard, the BCL scheme accept always the multipliers (hence the scheme is a pure proximal point algorithm).
| preconditioner_max_iter             | 10                             | Maximal number of authorized iterations for the preconditioner.
| preconditioner_accuracy             | 1.E-3                          | Accuracy level of the preconditioner.

\subsection OverviewInitialGuess The different initial guesses

The solver has five different possible initial guesses for warm starting or not the initial iterate values:
* NO_INITIAL_GUESS,
* EQUALITY_CONSTRAINED_INITIAL_GUESS,
* WARM_START_WITH_PREVIOUS_RESULT,
* WARM_START,
* COLD_START_WITH_PREVIOUS_RESULT.

The different options will be commented below in the introduced order above.

\subsubsection OverviewNoInitialGuess No initial guess

If set to this option, the solver will start with no initial guess, which means that the initial values of x, y and z are the 0 vector.

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
z stays to 0. In general this option warm starts well equality constrained QP.

\subsubsection OverviewWarmStartWithPreviousResult Warm start with previous result

If set to this option, the solver will warm start x, y and z with the values of the previous problem solved and it will keep all the last parameters of the solver (i.e., proximal step sizes for example, and the full workspace with the ldlt factorization etc.). Hence, if the new problem to solve is the same as the previous one, the problem is warm started at the solution (and zero iteration will be executed).

This option has been thought initially for being used in optimal control like problems, when the next problem to be solved is closed to the previous one. Indeed, if the problem changes only slightly, it is reasonable to warm start the new problem with the value of the previous one for speeding the whole runtime.

Note however, that if your update involves new matrices or that you decide to change parameters involved in the ldlt factorization (i.e., the proximal step sizes), then for consistency, the solver will automatically refactorize the ldlt with these updates (and taking into account the last values of x, y and z for the active set).

Finally, note that this option is set by default in the solver. At the first solve, as there is no previous results, x, y and z are warm started with the 0 vector value.

\subsubsection OverviewWarmStart Warm start

If set to this option, the solver expects then a warm start at the solve method.

Note, that it is not necessary to set this option through the command below (for example in C++) before the update or solve method call.
\code
Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
\endcode

It is sufficient to just add the warm start in the solve method, and the solver will automatically make the setting change internally.

\subsubsection OverviewColdStartWithPreviousResult Cold start with previous result

If set to this option, the solver will warm start x, y and z with the values of the previous problem solved. Contrary to the WARM_START_WITH_PREVIOUS_RESULT option, all other parameters of the solver (i.e., proximal step sizes for example, and the full workspace with the ldlt factorization etc.) are re-set to their default values (hence a factorization is reperformed taking into account of z warm start for the active set, but with default values of proximal step sizes).

This option has also been thought initially for being used in optimal control like problems, when the next problem to be solved is closed to the previous one. Indeed, if the problem changes only slightly, it is reasonable to warm start the new problem with the value of the previous one for speeding the whole runtime.

Note finally that at the first solve, as there is no previous results, x, y and z are warm started with the 0 vector value.

\section OverviewResults The results subclass

The result subclass is composed of:
* x: a primal solution,
* y: a Lagrange optimal multiplier for equality constraints,
* z: a Lagrange optimal multiplier for inequality constraints,
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

\subsection OverviewInfoClass The info subclass

In this table you have on the three columns from left to right: the name of the info subclass item, its default value and then a short description of it.

| Info item                           | Default value                  | Description
| ----------------------------------- | ------------------------------ | -----------------------------------------
| mu_eq                               | 1.E-3                          | Proximal step size wrt equality constraints multiplier.
| mu_in                               | 1.E-1                          | Proximal step size wrt inequality constraints multiplier.
| rho                                 | 1.E-6                          | Proximal step size wrt primal variable.
| iter                                | 0                              | Total number of iterations.
| iter_ext                            | 0                              | Total number of outer iterations.
| mu_updates                          | 0                              | Total number of mu updates.
| rho_updates                         | 0                              | Total number of rho updates.
| status                              | PROXQP_MAX_ITER_REACHED        | Status of the solver.
| setup_time                          | 0                              | Setup time (takes into account the equlibration procedure).
| solve_time                          | 0                              | Solve time (takes into account the first factorization).
| run_time                            | 0                              | the sum of the setupe time and the solve time.
| objValue                            | 0                              | The objective value to minimize.
| pri_res                             | 0                              | The primal residual.
| dua_res                             | 0                              | The dual residual.


Note finally that when initializing a QP object, by default the proximal step sizes (i.e., rho, mu_eq and mu_in) are set up by the default values defined in the Setting class. Hence, when doing multiple solves, if not specified, their values are re-set respectively to default_rho, default_mu_eq and default_mu_in. A small example is given below in c++ and python.

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

The solver has four status:
* PROXQP_SOLVED: the problem is solved.
* PROXQP_MAX_ITER_REACHED: the maximum number of iterations has been reached.
* PROXQP_PRIMAL_INFEASIBLE: the problem is primal infeasible.
* PROXQP_DUAL_INFEASIBLE: the problem is dual infeasible.

Infeasibility is detected using the necessary conditions exposed in [section 3.4](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf). More precisely, primal infeasibility is assumed if the following conditions are matched for some non zeros dy and dz (according the eps_prim_inf variable set by the user):

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

Dual infeasibility is assumed if the following two conditions are matched for some non zero dx (according the eps_dual_inf variable set by the user):

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

If the problem turns out to be primal or dual infeasible, then x, y and z stored in the results class will be the certificate of primal or dual infeasibility. More precisely:
* if the problem is dual infeasible, Qp.results.x will be the certificate dx of dual infeasibility satisfying \eqref{eq:approx_qp_sol_dual_inf} at precisifion Qp.settings.eps_dual_inf specified by the user,
* if the problem is primal infeasible, Qp.results.y and and Qp.results.z will be respectively the certificates dy and dz of primal infeasibility satisfying \eqref{eq:approx_qp_sol_prim_inf} at precisifion Qp.settings.eps_primal_inf specified by the user.


\section OverviewWhichBackend Which backend to use?

We have the following generic advices for choosing between the sparse and dense backend. If your problem is not:
* too large (less than some thousands variables),
* and too sparse (a sparsity ratio of your matrices greater than 0.1),

then we recommand using the solver with dense backend.


The sparsity ratio of matrix A is defined as:

$$ \text{sparsity}(A) = \frac{\text{nnz}(A)}{\text{number}_{\text{row}}(A) * \text{number}_{\text{col}}(A)}, $$

which accounts for the percentage of non zero elements in matrix A.

\section OverviewBenchmark Some important remarks when computing timings

We provide first some details about what is measured in the setup and solve time of ProxQP, which is of some importance when doing benchmarks with other solvers, as they can measure different things in a feature with a similar name.

Then we conclude this documentation section with some compilation options for ProxQP which can considerably speed up the solver, considering your OS architecture.

\subsection OverviewTimings What do the timings take into account?

An important remark about quadratic programming solver is that they all rely at some point on a factorization matrix algorithm, which constitutes the time bottle neck of the solver (as the factorization has a cubic order of complexity wrt dimension of the matrix to factorize).

Available solvers have often a similar API as the one we propose, with first an "init" method for initializing the model, and then a "solve" method for solving the QP problem. For not biasing the benchmarks, it is important to know where is done the first matrix factorization, as it constitutes a considerable cost. In our API, we have decided to make the first factorization in the solve method, as we consider that factorizing the problem is part of the solving part. Hence in terms of timing:
* results.info.setup_time: measures only the initialization of the model and also the preconditioning procedure (if it has been activated),
* results.info.solve_time: measures everything else (including the first factorization),
* results.info.run_time = results.info.setup_time + results.info.solve_time.

It is important to notice that some other solvers API have made different choices. For example, OSQP measures in the setup time the first factorization of the system (at the time [ProxQP algorithm](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf) was published). Hence our recommandation is that for benchmarking ProxQP against other solvers you should compare ProxQP runtime against the other solvers' runtime (i.e., everything from what constitutes their setup to their solve method). Otherwise, the benchmarks won't take into account timings that are comparable.

\subsection OverviewArchitectureOptions Architecture options when compiling ProxSuite

We highly encourage you to enable the vectorization of the underlying linear algebra for the best performances. You just need to activate the cmake option `BUILD_WITH_SIMD_SUPPORT=ON`, like:

\code
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_WITH_SIMD_SUPPORT=ON
make
make install
\endcode

ProxQP can be compiled more precisely with two SIMD instructions options for x86 instruction set architectures: [AVX-2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) and [AVX-512](https://en.wikipedia.org/wiki/AVX-512). They can considerably enhance the speed of ProxQP, and we encourage you to use them if your OS is compatible with them. For your information, our benchmarks for [ProxQP algorithm](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf) have been realised with AVX-2 compilation option.
