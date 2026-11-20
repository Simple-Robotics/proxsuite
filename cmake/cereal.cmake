# Detect and workaround ubuntu 22.04 https://bugs.launchpad.net/ubuntu/+source/libcereal/+bug/2174758
get_target_property(_cereal_INCLUDE_DIRECTORIES cereal::cereal INTERFACE_INCLUDE_DIRECTORIES)
if(_cereal_INCLUDE_DIRECTORIES STREQUAL "/include")
  get_filename_component(_cereal_PREFIX "${cereal_DIR}" PATH)
  get_filename_component(_cereal_PREFIX "${_cereal_PREFIX}" PATH)
  get_filename_component(_cereal_PREFIX "${_cereal_PREFIX}" PATH)
  set_target_properties(
    cereal::cereal
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_cereal_PREFIX}/include"
  )
endif()
