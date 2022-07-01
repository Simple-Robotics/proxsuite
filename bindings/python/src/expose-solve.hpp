//
// Copyright (c) 2022, INRIA
//
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

    m.def("solve_dense_qp", &dense::solve<T>,"function for solving a QP problem using dense backend. It is possible to setting up solver parameters or warm start it.",
        pybind11::arg_v("H_dense", std::nullopt, "quadratic cost with dense format."),
        pybind11::arg_v("H_sparse", std::nullopt, "quadratic cost in sparse format"),
        pybind11::arg_v("g", std::nullopt, "linear cost"),
        pybind11::arg_v("A_dense", std::nullopt, "equality constraint matrix with dense format."),
        pybind11::arg_v("A_sparse", std::nullopt, "equality constraint matrix in sparse format"),
        pybind11::arg_v("b", std::nullopt, "equality constraint vector"),
        pybind11::arg_v("C_dense", std::nullopt, "inequality constraint matrix with dense format."),
        pybind11::arg_v("C_sparse", std::nullopt, "inequality constraint matrix in sparse format"),
        pybind11::arg_v("u", std::nullopt, "upper inequality constraint vector"),
        pybind11::arg_v("l", std::nullopt, "lower inequality constraint vector"),
        pybind11::arg_v("eps_abs", std::nullopt, "absolute accuracy level used for the solver stopping criterion."),
        pybind11::arg_v("eps_rel", std::nullopt, "relative accuracy level used for the solver stopping criterion. Deactivated in standard settings."),
        pybind11::arg_v("rho",std::nullopt,"primal proximal parameter"),
        pybind11::arg_v("mu_eq",std::nullopt,"dual equality constraint proximal parameter"),
        pybind11::arg_v("mu_in",std::nullopt,"dual inequality constraint proximal parameter"),
        pybind11::arg_v("x", std::nullopt, "primal warm start"),
        pybind11::arg_v("y", std::nullopt, "dual equality warm start"),
        pybind11::arg_v("z", std::nullopt, "dual inequality warm start"),
        pybind11::arg_v("verbose", std::nullopt, "verbose option to print information at each iteration."),
        pybind11::arg_v("max_iter", std::nullopt, "maximum number of iteration."),
        pybind11::arg_v("alpha_bcl", std::nullopt, "bcl alpha parameter"),
        pybind11::arg_v("beta_bcl", std::nullopt, "bcl beta parameter"),
        pybind11::arg_v("refactor_dual_feasibility_threshold",std::nullopt, "threshold used to change the reduce rho proximal parameter with a refactorization."),
        pybind11::arg_v("refactor_rho_threshold", std::nullopt, "threshold used to refactorize the KKT matrix."),
        pybind11::arg_v("mu_max_eq", std::nullopt, "threshold for the equality constraint proximal parameter."),
        pybind11::arg_v("mu_max_in", std::nullopt, "thresold for the inequality constraint proximal parameter."),
        pybind11::arg_v("mu_update_factor", std::nullopt, "factor used for updating dual proximal parameters mu_eq and mu_in."),
        pybind11::arg_v("cold_reset_mu_eq", std::nullopt, "value used to cold re-start equality constraint proximal parameter."),
        pybind11::arg_v("cold_reset_mu_in", std::nullopt, "value used to cold re-start inequality constraint proximal parameter."),
        pybind11::arg_v("max_iter_in", std::nullopt, "maximum number of inner loop iteration."),
        pybind11::arg_v("eps_refact",std::nullopt,"safeguard threshold for refactoring the KKT matrix"),
        pybind11::arg_v("nb_iterative_refinement", std::nullopt, "maximal number of iterative refinement steps used for solving linear problems involved in Newton steps."),
        pybind11::arg_v("eps_primal_inf", std::nullopt, "primal accuracy level used for certifiate the problem is primal infeasible."),
        pybind11::arg_v("eps_dual_inf", std::nullopt, "dual accuracy level used for certifiate the problem is dual infeasible.")
        );
}

template <typename T,typename I>
void solveSparseQp(pybind11::module_ m) {

    m.def("solve_sparse_qp", &sparse::solve<T,I>,"function for solving a QP problem with sparse backend. It is possible to setting up solver parameters or warm start it.",
        pybind11::arg_v("H", std::nullopt, "quadratic cost"),
        pybind11::arg_v("g", std::nullopt, "linear cost"),
        pybind11::arg_v("A", std::nullopt, "equality constraint matrix"),
        pybind11::arg_v("b", std::nullopt, "equality constraint vector"),
        pybind11::arg_v("C", std::nullopt, "inequality constraint matrix"),
        pybind11::arg_v("u", std::nullopt, "upper inequality constraint vector"),
        pybind11::arg_v("l", std::nullopt, "lower inequality constraint vector"),
        pybind11::arg_v("eps_abs", std::nullopt, "absolute accuracy level used for the solver stopping criterion."),
        pybind11::arg_v("eps_rel", std::nullopt, "relative accuracy level used for the solver stopping criterion. Deactivated in standard settings."),
        pybind11::arg_v("rho",std::nullopt,"primal proximal parameter"),
        pybind11::arg_v("mu_eq",std::nullopt,"dual equality constraint proximal parameter"),
        pybind11::arg_v("mu_in",std::nullopt,"dual inequality constraint proximal parameter"),
        pybind11::arg_v("x", std::nullopt, "primal warm start"),
        pybind11::arg_v("y", std::nullopt, "dual equality warm start"),
        pybind11::arg_v("z", std::nullopt, "dual inequality warm start"),
        pybind11::arg_v("verbose", std::nullopt, "verbose option to print information at each iteration."),
        pybind11::arg_v("max_iter", std::nullopt, "maximum number of iteration."),
        pybind11::arg_v("alpha_bcl", std::nullopt, "bcl alpha parameter"),
        pybind11::arg_v("beta_bcl", std::nullopt, "bcl beta parameter"),
        pybind11::arg_v("refactor_dual_feasibility_threshold",std::nullopt, "threshold used to change the reduce rho proximal parameter with a refactorization."),
        pybind11::arg_v("refactor_rho_threshold", std::nullopt, "threshold used to refactorize the KKT matrix."),
        pybind11::arg_v("mu_max_eq", std::nullopt, "threshold for the equality constraint proximal parameter."),
        pybind11::arg_v("mu_max_in", std::nullopt, "thresold for the inequality constraint proximal parameter."),
        pybind11::arg_v("mu_update_factor", std::nullopt, "factor used for updating dual proximal parameters mu_eq and mu_in."),
        pybind11::arg_v("cold_reset_mu_eq", std::nullopt, "value used to cold re-start equality constraint proximal parameter."),
        pybind11::arg_v("cold_reset_mu_in", std::nullopt, "value used to cold re-start inequality constraint proximal parameter."),
        pybind11::arg_v("max_iter_in", std::nullopt, "maximum number of inner loop iteration."),
        pybind11::arg_v("eps_refact",std::nullopt,"safeguard threshold for refactoring the KKT matrix"),
        pybind11::arg_v("nb_iterative_refinement", std::nullopt, "maximal number of iterative refinement steps used for solving linear problems involved in Newton steps."),
        pybind11::arg_v("eps_primal_inf", std::nullopt, "primal accuracy level used for certifiate the problem is primal infeasible."),
        pybind11::arg_v("eps_dual_inf", std::nullopt, "dual accuracy level used for certifiate the problem is dual infeasible.")
        );
}

} //namespace python
} // namespace qp
} // namespace proxsuite
