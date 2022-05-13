#include <qp/dense/wrapper.hpp>
#include <qp/sparse/wrapper.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace proxsuite {
namespace qp {
using veg::isize;

namespace python {

template <typename T>
void solveDenseQp(pybind11::module_ m) {

    m.def("solve_dense_qp", &qp::dense::solve<T>,"function for solving a dense QP problem. It is possible to setting up solver parameters or warm start it.",
        pybind11::arg_v("H", tl::nullopt, "quadratic cost"),
        pybind11::arg_v("g", tl::nullopt, "linear cost"),
        pybind11::arg_v("A", tl::nullopt, "equality constraint matrix"),
        pybind11::arg_v("b", tl::nullopt, "equality constraint vector"),
        pybind11::arg_v("C", tl::nullopt, "inequality constraint matrix"),
        pybind11::arg_v("u", tl::nullopt, "upper inequality constraint vector"),
        pybind11::arg_v("l", tl::nullopt, "lower inequality constraint vector"),
        pybind11::arg_v("eps_abs", tl::nullopt, "absolute accuracy level used for the solver stopping criterion."),
        pybind11::arg_v("eps_rel", tl::nullopt, "relative accuracy level used for the solver stopping criterion. Deactivated in standard settings."),
        pybind11::arg_v("rho",tl::nullopt,"primal proximal parameter"),
        pybind11::arg_v("mu_eq",tl::nullopt,"dual equality constraint proximal parameter"),
        pybind11::arg_v("mu_in",tl::nullopt,"dual inequality constraint proximal parameter"),
        pybind11::arg_v("x", tl::nullopt, "primal warm start"),
        pybind11::arg_v("y", tl::nullopt, "dual equality warm start"),
        pybind11::arg_v("z", tl::nullopt, "dual inequality warm start"),
        pybind11::arg_v("verbose", tl::nullopt, "verbose option to print information at each iteration."),
        pybind11::arg_v("max_iter", tl::nullopt, "maximum number of iteration."),
        pybind11::arg_v("alpha_bcl", tl::nullopt, "bcl alpha parameter"),
        pybind11::arg_v("beta_bcl", tl::nullopt, "bcl beta parameter"),
        pybind11::arg_v("refactor_dual_feasibility_threshold",tl::nullopt, "threshold used to change the reduce rho proximal parameter with a refactorization."),
        pybind11::arg_v("refactor_rho_threshold", tl::nullopt, "threshold used to refactorize the KKT matrix."),
        pybind11::arg_v("mu_max_eq", tl::nullopt, "threshold for the equality constraint proximal parameter."),
        pybind11::arg_v("mu_max_in", tl::nullopt, "thresold for the inequality constraint proximal parameter."),
        pybind11::arg_v("mu_update_factor", tl::nullopt, "factor used for updating dual proximal parameters mu_eq and mu_in."),
        pybind11::arg_v("cold_reset_mu_eq", tl::nullopt, "value used to cold re-start equality constraint proximal parameter."),
        pybind11::arg_v("cold_reset_mu_in", tl::nullopt, "value used to cold re-start inequality constraint proximal parameter."),
        pybind11::arg_v("max_iter_in", tl::nullopt, "maximum number of inner loop iteration."),
        pybind11::arg_v("eps_refact",tl::nullopt,"safeguard threshold for refactoring the KKT matrix"),
        pybind11::arg_v("nb_iterative_refinement", tl::nullopt, "maximal number of iterative refinement steps used for solving linear problems involved in Newton steps."),
        pybind11::arg_v("eps_primal_inf", tl::nullopt, "primal accuracy level used for certifiate the problem is primal infeasible."),
        pybind11::arg_v("eps_dual_inf", tl::nullopt, "dual accuracy level used for certifiate the problem is dual infeasible.")
        );
}

template <typename T,typename I>
void solveSparseQp(pybind11::module_ m) {

    m.def("solve_sparse_qp", &qp::sparse::solve<T,I>,"function for solving a sparse QP problem. It is possible to setting up solver parameters or warm start it.",
        pybind11::arg_v("H", tl::nullopt, "quadratic cost"),
        pybind11::arg_v("g", tl::nullopt, "linear cost"),
        pybind11::arg_v("A", tl::nullopt, "equality constraint matrix"),
        pybind11::arg_v("b", tl::nullopt, "equality constraint vector"),
        pybind11::arg_v("C", tl::nullopt, "inequality constraint matrix"),
        pybind11::arg_v("u", tl::nullopt, "upper inequality constraint vector"),
        pybind11::arg_v("l", tl::nullopt, "lower inequality constraint vector"),
        pybind11::arg_v("eps_abs", tl::nullopt, "absolute accuracy level used for the solver stopping criterion."),
        pybind11::arg_v("eps_rel", tl::nullopt, "relative accuracy level used for the solver stopping criterion. Deactivated in standard settings."),
        pybind11::arg_v("rho",tl::nullopt,"primal proximal parameter"),
        pybind11::arg_v("mu_eq",tl::nullopt,"dual equality constraint proximal parameter"),
        pybind11::arg_v("mu_in",tl::nullopt,"dual inequality constraint proximal parameter"),
        pybind11::arg_v("x", tl::nullopt, "primal warm start"),
        pybind11::arg_v("y", tl::nullopt, "dual equality warm start"),
        pybind11::arg_v("z", tl::nullopt, "dual inequality warm start"),
        pybind11::arg_v("verbose", tl::nullopt, "verbose option to print information at each iteration."),
        pybind11::arg_v("max_iter", tl::nullopt, "maximum number of iteration."),
        pybind11::arg_v("alpha_bcl", tl::nullopt, "bcl alpha parameter"),
        pybind11::arg_v("beta_bcl", tl::nullopt, "bcl beta parameter"),
        pybind11::arg_v("refactor_dual_feasibility_threshold",tl::nullopt, "threshold used to change the reduce rho proximal parameter with a refactorization."),
        pybind11::arg_v("refactor_rho_threshold", tl::nullopt, "threshold used to refactorize the KKT matrix."),
        pybind11::arg_v("mu_max_eq", tl::nullopt, "threshold for the equality constraint proximal parameter."),
        pybind11::arg_v("mu_max_in", tl::nullopt, "thresold for the inequality constraint proximal parameter."),
        pybind11::arg_v("mu_update_factor", tl::nullopt, "factor used for updating dual proximal parameters mu_eq and mu_in."),
        pybind11::arg_v("cold_reset_mu_eq", tl::nullopt, "value used to cold re-start equality constraint proximal parameter."),
        pybind11::arg_v("cold_reset_mu_in", tl::nullopt, "value used to cold re-start inequality constraint proximal parameter."),
        pybind11::arg_v("max_iter_in", tl::nullopt, "maximum number of inner loop iteration."),
        pybind11::arg_v("eps_refact",tl::nullopt,"safeguard threshold for refactoring the KKT matrix"),
        pybind11::arg_v("nb_iterative_refinement", tl::nullopt, "maximal number of iterative refinement steps used for solving linear problems involved in Newton steps."),
        pybind11::arg_v("eps_primal_inf", tl::nullopt, "primal accuracy level used for certifiate the problem is primal infeasible."),
        pybind11::arg_v("eps_dual_inf", tl::nullopt, "dual accuracy level used for certifiate the problem is dual infeasible.")
        );
}

} //namespace python
} // namespace qp
} // namespace proxsuite
