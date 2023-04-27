//
// Copyright (c) 2023 INRIA
//

#ifndef PROXSUITE_PROXQP_PARALLEL_OMP_HPP
#define PROXSUITE_PROXQP_PARALLEL_OMP_HPP

#include <omp.h>

namespace proxsuite {
inline void
set_default_omp_options(const size_t num_threads = (size_t)
                          omp_get_max_threads())
{
  omp_set_num_threads((int)num_threads);
  omp_set_dynamic(0);
}
}

#endif // ifndef PROXSUITE_PROXQP_PARALLEL_OMP_HPP
