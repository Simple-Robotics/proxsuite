#include <qp/results.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace proxsuite {
namespace qp {
namespace python {
template <typename T>
void exposeResults(pybind11::module_ m) {

	::pybind11::class_<Info<T>>(m, "info",pybind11::module_local())
			.def(::pybind11::init())
			.def_readwrite("n_c", &Info<T>::n_c)
			.def_readwrite("mu_eq", &Info<T>::mu_eq)
			.def_readwrite("mu_in", &Info<T>::mu_in)
			.def_readwrite("rho", &Info<T>::rho)
			.def_readwrite("iter", &Info<T>::iter)
			.def_readwrite("iter_ext", &Info<T>::iter_ext)
			.def_readwrite("run_time", &Info<T>::run_time)
			.def_readwrite("setup_time", &Info<T>::setup_time)
			.def_readwrite("solve_time", &Info<T>::solve_time)
			.def_readwrite("pri_res", &Info<T>::pri_res)
			.def_readwrite("dua_res", &Info<T>::dua_res)
			.def_readwrite("objValue", &Info<T>::objValue)
			.def_readwrite("status", &Info<T>::status)
			.def_readwrite("rho_updates", &Info<T>::rho_updates)
			.def_readwrite("mu_updates", &Info<T>::mu_updates);

	::pybind11::class_<Results<T>>(m, "Results",pybind11::module_local())
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readwrite("x", &Results<T>::x) 
			.def_readwrite("y", &Results<T>::y)
			.def_readwrite("z", &Results<T>::z)
			.def_readwrite("info", &Results<T>::info);
}
} //namespace python
} // namespace qp
} // namespace proxsuite
