#include <qp/dense/wrapper.hpp>
#include <qp/sparse/wrapper.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<tl::optional<T>> : optional_caster<tl::optional<T>> {};
template <>
struct type_caster<tl::nullopt_t> : public void_caster<tl::nullopt_t> {};

} // namespace detail
} // namespace pybind11

namespace proxsuite {
namespace qp {
using veg::isize;

namespace python {

template <typename T>
void exposeQpObjectDense(pybind11::module_ m) {

	::pybind11::class_<proxsuite::qp::dense::QP<T>>(m, "QP_dense")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readwrite(
					"results",
					&qp::dense::QP<T>::results,
					"class containing the solution or certificate of infeasibility, "
					"and "
					"information statistics in an info subclass.")
			.def_readwrite(
					"settings",
					&qp::dense::QP<T>::settings,
					"class with settings option of the solver.")
			.def_readwrite(
					"data", &qp::dense::QP<T>::data, "class containing the QP model")
			.def(
					"setup_dense_matrices",
					&qp::dense::QP<T>::setup_dense_matrices,
					"function for setting up the solver when passing dense matrices in "
					"entry.",
					pybind11::arg_v("H", tl::nullopt, "quadratic cost"),
					pybind11::arg_v("g", tl::nullopt, "linear cost"),
					pybind11::arg_v("A", tl::nullopt, "equality constraint matrix"),
					pybind11::arg_v("b", tl::nullopt, "equality constraint vector"),
					pybind11::arg_v("C", tl::nullopt, "inequality constraint matrix"),
					pybind11::arg_v(
							"l", tl::nullopt, "lower inequality constraint vector"),
					pybind11::arg_v(
							"u", tl::nullopt, "upper inequality constraint vector"))
			.def(
					"setup_sparse_matrices",
					&qp::dense::QP<T>::setup_sparse_matrices,
					"function for setting up the solver when passing sparse matrices "
					"in "
					"entry.",
					pybind11::arg_v("H", tl::nullopt, "quadratic cost"),
					pybind11::arg_v("g", tl::nullopt, "linear cost"),
					pybind11::arg_v("A", tl::nullopt, "equality constraint matrix"),
					pybind11::arg_v("b", tl::nullopt, "equality constraint vector"),
					pybind11::arg_v("C", tl::nullopt, "inequality constraint matrix"),
					pybind11::arg_v(
							"l", tl::nullopt, "lower inequality constraint vector"),
					pybind11::arg_v(
							"u", tl::nullopt, "upper inequality constraint vector"))
			.def(
					"solve",
					&qp::dense::QP<T>::solve,
					"function used for solving the QP problem.")
			.def(
					"update",
					&qp::dense::QP<T>::update,
					"function used for updating matrix or vector entry of the model.",
					pybind11::arg_v("H", tl::nullopt, "quadratic cost"),
					pybind11::arg_v("g", tl::nullopt, "linear cost"),
					pybind11::arg_v("A", tl::nullopt, "equality constraint matrix"),
					pybind11::arg_v("b", tl::nullopt, "equality constraint vector"),
					pybind11::arg_v("C", tl::nullopt, "inequality constraint matrix"),
					pybind11::arg_v(
							"l", tl::nullopt, "lower inequality constraint vector"),
					pybind11::arg_v(
							"u", tl::nullopt, "upper inequality constraint vector"))
			.def(
					"update_prox_parameter",
					&qp::dense::QP<T>::update_prox_parameter,
					"function used for updating proximal parameters of the solver. The "
					"user must settup back the solver before using solve method.",
					pybind11::arg_v("rho", tl::nullopt, "primal proximal parameter"),
					pybind11::arg_v(
							"mu_eq",
							tl::nullopt,
							"dual equatlity constraint proximal parameter"),
					pybind11::arg_v(
							"mu_in",
							tl::nullopt,
							"dual inequatlity constraint proximal parameter"))
			.def(
					"warm_start",
					&qp::dense::QP<T>::warm_start,
					"function used for warm starting the solver with some vectors. The "
					"user must settup back the solver before using solve method.",
					pybind11::arg_v("x", tl::nullopt, "primal warm start"),
					pybind11::arg_v("y", tl::nullopt, "dual equality warm start"),
					pybind11::arg_v("z", tl::nullopt, "dual inequality warm start"))
			.def(
					"cleanup",
					&qp::dense::QP<T>::cleanup,
					"function used for cleaning the workspace and result "
					"classes.");
}

template <typename T,typename I>
void exposeQpObjectSparse(pybind11::module_ m) {

	::pybind11::class_<proxsuite::qp::sparse::QP<T,I>>(m, "QP_sparse")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readwrite(
					"results",
					&qp::sparse::QP<T,I>::results,
					"class containing the solution or certificate of infeasibility, "
					"and "
					"information statistics in an info subclass.")
			.def_readwrite(
					"settings",
					&qp::sparse::QP<T,I>::settings,
					"class with settings option of the solver.")
			.def(
					"setup_sparse_matrices",
					&qp::sparse::QP<T,I>::setup_sparse_matrices,
					"function for setting up the solver when passing sparse matrices in "
					"entry.",
					pybind11::arg_v("H", tl::nullopt, "quadratic cost"),
					pybind11::arg_v("g", tl::nullopt, "linear cost"),
					pybind11::arg_v("A", tl::nullopt, "equality constraint matrix"),
					pybind11::arg_v("b", tl::nullopt, "equality constraint vector"),
					pybind11::arg_v("C", tl::nullopt, "inequality constraint matrix"),
					pybind11::arg_v(
							"l", tl::nullopt, "lower inequality constraint vector"),
					pybind11::arg_v(
							"u", tl::nullopt, "upper inequality constraint vector"))
			.def(
					"solve",
					&qp::sparse::QP<T,I>::solve,
					"function used for solving the QP problem.")
			.def(
					"update_prox_parameter",
					&qp::sparse::QP<T,I>::update_prox_parameter,
					"function used for updating proximal parameters of the solver. The "
					"user must settup back the solver before using solve method.",
					pybind11::arg_v("rho", tl::nullopt, "primal proximal parameter"),
					pybind11::arg_v(
							"mu_eq",
							tl::nullopt,
							"dual equatlity constraint proximal parameter"),
					pybind11::arg_v(
							"mu_in",
							tl::nullopt,
							"dual inequatlity constraint proximal parameter"))
			.def(
					"warm_sart",
					&qp::sparse::QP<T,I>::warm_start,
					"function used for warm starting the solver with some vectors. The "
					"user must settup back the solver before using solve method.",
					pybind11::arg_v("x", tl::nullopt, "primal warm start"),
					pybind11::arg_v("y", tl::nullopt, "dual equality warm start"),
					pybind11::arg_v("z", tl::nullopt, "dual inequality warm start"))
			.def(
					"cleanup",
					&qp::sparse::QP<T,I>::cleanup,
					"function used for cleaning the result "
					"class.");
}

} //namespace python

} // namespace qp
} // namespace proxsuite
