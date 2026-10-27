# ROS humble is mainly on ubuntu 22.04, where nanobind is not available
if("$ENV{ROS_DISTRO}" STREQUAL "humble")
  message(STATUS "ROS humble detected. Don't build python interface by default")
  set(BUILD_PYTHON_INTERFACE_DEFAULT OFF)
endif()
