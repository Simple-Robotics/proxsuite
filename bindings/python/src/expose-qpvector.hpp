//
// Copyright (c) 2022 INRIA
//

#include <proxsuite/proxqp/dense/wrapper.hpp>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
exposeQpVectorDense(pybind11::module_ m)
{

  ::pybind11::class_<dense::VectorQP<T>>(m, "VectorQP")
    .def(
      ::pybind11::init<i64>(),
      pybind11::arg_v("batch_size", 0, "number of QPs to be stored."),
      "Default constructor using the BatchSize of qp models to store.") // constructor
    .def("init_qp_in_place",
         &dense::VectorQP<T>::init_qp_in_place,
         pybind11::return_value_policy::reference,
         "init a dense QP in place and return a reference to it.")
    .def("insert",
         &dense::VectorQP<T>::insert,
         "inserts a qp at the end of the vector of qps.")
    .def("get",
         &dense::VectorQP<T>::get,
         pybind11::return_value_policy::reference,
         "get the qp.");
}
} // namespace python
} // namespace dense

namespace sparse {
namespace python {

template<typename T, typename I>
void
exposeQpVectorSparse(pybind11::module_ m)
{

  ::pybind11::class_<sparse::VectorQP<T, I>>(m, "VectorQP")
    .def(
      ::pybind11::init<i64>(),
      pybind11::arg_v("batch_size", 0, "number of QPs to be stored."),
      "Default constructor using the BatchSize of qp models to store.") // constructor
    .def("init_qp_in_place",
         &sparse::VectorQP<T, I>::init_qp_in_place,
         pybind11::return_value_policy::reference,
         "init a sparse QP in place and return a reference to it.")
    .def("insert",
         &sparse::VectorQP<T, I>::insert,
         "inserts a qp at the end of the vector of qps.")
    .def("get",
         &sparse::VectorQP<T, I>::get,
         pybind11::return_value_policy::reference,
         "get the qp.");
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
