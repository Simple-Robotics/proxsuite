#include <qp/dense/model.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace proxsuite {
namespace qp {
namespace dense {
namespace python {
template <typename T>
void exposeModel(pybind11::module_ m) {
	::pybind11::class_<qp::dense::Model<T>>(m, "Model")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readonly("H", &qp::dense::Model<T>::H)
			.def_readonly("g", &qp::dense::Model<T>::g)
			.def_readonly("A", &qp::dense::Model<T>::A)
			.def_readonly("b", &qp::dense::Model<T>::b)
			.def_readonly("C", &qp::dense::Model<T>::C)
			.def_readonly("u", &qp::dense::Model<T>::u)
			.def_readonly("l", &qp::dense::Model<T>::l)
			.def_readonly("dim", &qp::dense::Model<T>::dim)
			.def_readonly("n_eq", &qp::dense::Model<T>::n_eq)
			.def_readonly("n_in", &qp::dense::Model<T>::n_in)
			.def_readonly("n_total", &qp::dense::Model<T>::n_total);
}
} // namespace python
} // namespace dense
} // namespace qp
} // namespace proxsuite
