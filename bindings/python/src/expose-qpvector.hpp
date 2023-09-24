//
// Copyright (c) 2023 INRIA
//

#include <proxsuite/proxqp/dense/wrapper.hpp>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
exposeQPVectorDense(pybind11::module_ m)
{

  ::pybind11::class_<dense::BatchQP<T>>(m, "BatchQP")
    .def(
      ::pybind11::init<i64>(),
      pybind11::arg_v("batch_size", 0, "number of QPs to be stored."),
      "Default constructor using the BatchSize of qp models to store.") // constructor
    .def("init_qp_in_place",
         &dense::BatchQP<T>::init_qp_in_place,
         pybind11::return_value_policy::reference,
         "init a dense QP in place and return a reference to it.")
    .def("insert",
         &dense::BatchQP<T>::insert,
         "inserts a qp at the end of the vector of qps.")
    .def("size", &dense::BatchQP<T>::size)
    .def("get",
         (dense::QP<T> & (dense::BatchQP<T>::*)(isize)) &
           dense::BatchQP<T>::get,
         pybind11::return_value_policy::reference,
         "get the qp.");
}
} // namespace python
} // namespace dense

namespace sparse {
namespace python {

template<typename T, typename I>
void
exposeQPVectorSparse(pybind11::module_ m)
{

  ::pybind11::class_<sparse::BatchQP<T, I>>(m, "BatchQP")
    .def(
      ::pybind11::init<i64>(),
      pybind11::arg_v("batch_size", 0, "number of QPs to be stored."),
      "Default constructor using the BatchSize of qp models to store.") // constructor
    .def("init_qp_in_place",
         &sparse::BatchQP<T, I>::init_qp_in_place,
         pybind11::return_value_policy::reference,
         "init a sparse QP in place and return a reference to it.")
    .def("size", &sparse::BatchQP<T, I>::size)
    .def("get",
         (sparse::QP<T, I> & (sparse::BatchQP<T, I>::*)(isize)) &
           sparse::BatchQP<T, I>::get,
         pybind11::return_value_policy::reference,
         "get the qp.");
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
