//
// Copyright (c) 2023 INRIA
//

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>

#include <nanobind/nanobind.h>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
exposeQPVectorDense(nanobind::module_ m)
{

  ::nanobind::class_<dense::BatchQP<T>>(m, "BatchQP")
    .def(
      ::nanobind::init<size_t>(),
      nanobind::arg("batch_size") = 0,
      "Default constructor using the BatchSize of qp models to store.") // constructor
    .def("init_qp_in_place",
         &dense::BatchQP<T>::init_qp_in_place,
         nanobind::rv_policy::reference,
         "init a dense QP in place and return a reference to it.")
    .def("insert",
         &dense::BatchQP<T>::insert,
         "inserts a qp at the end of the vector of qps.")
    .def("size", &dense::BatchQP<T>::size)
    .def("get",
         (dense::QP<T> & (dense::BatchQP<T>::*)(isize)) &
           dense::BatchQP<T>::get,
         nanobind::rv_policy::reference,
         "get the qp.");
}
} // namespace python
} // namespace dense

namespace sparse {
namespace python {

template<typename T, typename I>
void
exposeQPVectorSparse(nanobind::module_ m)
{

  ::nanobind::class_<sparse::BatchQP<T, I>>(m, "BatchQP")
    .def(
      ::nanobind::init<long unsigned int>(),
      nanobind::arg("batch_size") = 0,
      "Default constructor using the BatchSize of qp models to store.") // constructor
    .def("init_qp_in_place",
         &sparse::BatchQP<T, I>::init_qp_in_place,
         nanobind::rv_policy::reference,
         "init a sparse QP in place and return a reference to it.")
    .def("size", &sparse::BatchQP<T, I>::size)
    .def("get",
         (sparse::QP<T, I> & (sparse::BatchQP<T, I>::*)(isize)) &
           sparse::BatchQP<T, I>::get,
         nanobind::rv_policy::reference,
         "get the qp.");
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
