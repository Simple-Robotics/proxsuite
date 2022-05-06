#include <qp/dense/wrapper.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace qp {
using veg::isize;


namespace python{
template<typename T>
void exposeQpObject(pybind11::module_ m){

	::pybind11::class_<qp::dense::QP<T>>(m, "QP")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readwrite("results", &qp::dense::QP<T>::results)
			.def_readwrite("settings", &qp::dense::QP<T>::settings)
			.def_readwrite("data", &qp::dense::QP<T>::data)
			.def("setup_dense_matrices",&qp::dense::QP<T>::setup_dense_matrices)
			.def("setup_sparse_matrices",&qp::dense::QP<T>::setup_sparse_matrices)
			.def("solve", &qp::dense::QP<T>::solve)
			.def("update", &qp::dense::QP<T>::update)
			.def("update_prox_parameter", &qp::dense::QP<T>::update_prox_parameter)
			.def("warm_sart", &qp::dense::QP<T>::warm_sart)
			.def("cleanup", &qp::dense::QP<T>::cleanup);
}
} //namespace python

} // namespace qp
