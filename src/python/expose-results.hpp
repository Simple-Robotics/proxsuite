#include <qp/Results.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace qp {
namespace python{
template<typename T>
void exposeResults(pybind11::module_ m){

	::pybind11::class_<qp::Info<T>>(m, "info")
			.def(::pybind11::init()) 
			.def_readwrite("n_c", &qp::Info<T>::n_c)
			.def_readwrite("mu_eq", &qp::Info<T>::mu_eq)
			.def_readwrite("mu_in", &qp::Info<T>::mu_in)
			.def_readwrite("rho", &qp::Info<T>::rho)
			.def_readwrite("iter", &qp::Info<T>::iter)
			.def_readwrite("iter_ext", &qp::Info<T>::iter_ext)
			.def_readwrite("run_time", &qp::Info<T>::run_time)
			.def_readwrite("setup_time", &qp::Info<T>::setup_time)
			.def_readwrite("solve_time", &qp::Info<T>::solve_time)
			.def_readwrite("pri_res", &qp::Info<T>::pri_res)
			.def_readwrite("dua_res", &qp::Info<T>::dua_res)
			.def_readwrite("objValue", &qp::Info<T>::objValue)
			.def_readwrite("status", &qp::Info<T>::status)
			.def_readwrite("rho_updates", &qp::Info<T>::rho_updates)
			.def_readwrite("mu_updates", &qp::Info<T>::mu_updates);

	::pybind11::class_<qp::Results<T>>(m, "Results")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readwrite("x", &qp::Results<T>::x)
			.def_readwrite("y", &qp::Results<T>::y)
			.def_readwrite("z", &qp::Results<T>::z)
			.def_readwrite("info", &qp::Results<T>::info);
}
} //namespace python
} // namespace qp
