find_path(
  MAROS_MESZAROS_DIR
  NAMES AUG2D.mat QPCBOEI1.mat
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/data/maros_meszaros_data
  REQUIRED
)

add_subdirectory(cpp)

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
