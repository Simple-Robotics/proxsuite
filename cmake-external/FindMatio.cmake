# =============================================================================
# SPDX-FileCopyrightText: 2021 Stefan Gerlach <stefan.gerlach@uni.kn>
#
# SPDX-License-Identifier: BSD-3-Clause
# =============================================================================

find_library(MATIO_LIBRARIES NAMES matio libmatio)

find_path(MATIO_INCLUDE_DIR matio.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Matio
  FOUND_VAR MATIO_FOUND
  REQUIRED_VARS MATIO_LIBRARIES MATIO_INCLUDE_DIR
)

if(MATIO_FOUND)
  add_library(matio UNKNOWN IMPORTED)
  set_target_properties(
    matio
    PROPERTIES
      IMPORTED_LOCATION "${MATIO_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}"
  )
else()
  set(MATIO_LIBRARIES "")
endif()

mark_as_advanced(MATIO_LIBRARIES MATIO_INCLUDE_DIR)

include(FeatureSummary)
set_package_properties(
  Matio
  PROPERTIES
    DESCRIPTION "Reading and writing binary MATLAB MAT files"
    URL "https://github.com/tbeu/matio"
)
