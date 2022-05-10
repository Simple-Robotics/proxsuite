#include <qp/dense/Data.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace proxsuite {
namespace qp {
namespace dense {
namespace python {
template <typename T>
void exposeData(pybind11::module_ m) {
	::pybind11::class_<qp::dense::Data<T>>(m, "Data")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readonly("H", &qp::dense::Data<T>::H)
			.def_readonly("g", &qp::dense::Data<T>::g)
			.def_readonly("A", &qp::dense::Data<T>::A)
			.def_readonly("b", &qp::dense::Data<T>::b)
			.def_readonly("C", &qp::dense::Data<T>::C)
			.def_readonly("u", &qp::dense::Data<T>::u)
			.def_readonly("l", &qp::dense::Data<T>::l)
			.def_readonly("dim", &qp::dense::Data<T>::dim)
			.def_readonly("n_eq", &qp::dense::Data<T>::n_eq)
			.def_readonly("n_in", &qp::dense::Data<T>::n_in)
			.def_readonly("n_total", &qp::dense::Data<T>::n_total);
}
} // namespace python
} // namespace dense
} // namespace qp
} // namespace proxsuite
