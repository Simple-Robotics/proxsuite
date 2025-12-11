function(proxsuite_add_benchmark name)
  add_executable(proxsuite-benchmark-${name} ${name}.cpp)
  target_link_libraries(proxsuite-benchmark-${name} PRIVATE proxsuite)

  if(BUILD_WITH_VECTORIZATION_SUPPORT)
    add_executable(proxsuite-vectorized-benchmark-${name} ${name}.cpp)
    target_link_libraries(proxsuite-vectorized-benchmark-${name} PRIVATE proxsuite-vectorized)
  endif()
endfunction()

proxsuite_add_benchmark(timings-box-constraints)
proxsuite_add_benchmark(timings-dense-backend)
proxsuite_add_benchmark(timings-diagonal-hessian)
proxsuite_add_benchmark(timings-lp)

if(BUILD_WITH_OPENMP_SUPPORT)
  proxsuite_add_benchmark(timings-parallel)
endif()
