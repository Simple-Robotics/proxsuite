function(proxsuite_add_benchmark name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs LINK_LIBRARIES)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(proxsuite-benchmark-${name} ${name}.cpp)
  target_link_libraries(proxsuite-benchmark-${name} PRIVATE proxsuite ${arg_LINK_LIBRARIES})
  proxsuite_set_default_target_options(proxsuite-benchmark-${name})

  if(BUILD_WITH_VECTORIZATION_SUPPORT)
    add_executable(proxsuite-vectorized-benchmark-${name} ${name}.cpp)
    target_link_libraries(
      proxsuite-vectorized-benchmark-${name}
      PRIVATE proxsuite-vectorized ${arg_LINK_LIBRARIES}
    )
    proxsuite_set_default_target_options(proxsuite-vectorized-benchmark-${name})
  endif()
endfunction()

proxsuite_add_benchmark(timings-box-constraints)
proxsuite_add_benchmark(timings-dense-backend)
proxsuite_add_benchmark(timings-diagonal-hessian)
proxsuite_add_benchmark(timings-lp)

if(BUILD_WITH_OPENMP_SUPPORT)
  proxsuite_add_benchmark(timings-parallel LINK_LIBRARIES proxsuite-parallel)
endif()
