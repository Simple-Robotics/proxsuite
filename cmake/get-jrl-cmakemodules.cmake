# Get jrl-cmakemodules package

# Upstream (https://github.com/jrl-umi3218/jrl-cmakemodules), the new v2 version is located in a subfolder,
# We need to set this variable to bypass the v1 and load the v2.
set(
  JRL_CMAKEMODULES_USE_V2
  ON
  CACHE BOOL
  "Use jrl-cmakemodules v2 on https://github.com/jrl-umi3218/jrl-cmakemodules"
)

# Option 1: pass -DJRL_CMAKEMODULES_SOURCE_DIR=... to cmake command line
if(JRL_CMAKEMODULES_SOURCE_DIR)
  message(
    STATUS
    "JRL_CMAKEMODULES_SOURCE_DIR variable set, adding jrl-cmakemodules from source directory: ${JRL_CMAKEMODULES_SOURCE_DIR}"
  )
  add_subdirectory(${JRL_CMAKEMODULES_SOURCE_DIR} jrl-cmakemodules)
  return()
endif()

# Option 2: use JRL_CMAKEMODULES_SOURCE_DIR environment variable (pixi might unset it, prefer option 1)
if(ENV{JRL_CMAKEMODULES_SOURCE_DIR})
  message(
    STATUS
    "JRL_CMAKEMODULES_SOURCE_DIR environement variable set, adding jrl-cmakemodules from source directory: $ENV{JRL_CMAKEMODULES_SOURCE_DIR}"
  )
  add_subdirectory($ENV{JRL_CMAKEMODULES_SOURCE_DIR} jrl-cmakemodules)
  return()
endif()

# Option 3: Try to look for the installed package
message(STATUS "Looking for jrl-cmakemodules (version: >=1.1.2) package...")
find_package(jrl-cmakemodules 1.1.2 CONFIG QUIET)

# Verify that the v2 directory exists (might not have been released yet)
if(jrl-cmakemodules_FOUND)
  if(NOT EXISTS ${jrl-cmakemodules_DIR}/../../jrl-cmakemodules/v2)
    message(
      WARNING
      "jrl-cmakemodules found (version: ${jrl-cmakemodules_VERSION}) at '${jrl-cmakemodules_DIR}', but v2 directory is missing. Ignoring this installation."
    )
    unset(jrl-cmakemodules_FOUND)
  endif()
endif()

# If we have the package, we are done here.
if(jrl-cmakemodules_FOUND)
  message(STATUS "Found jrl-cmakemodules (version: ${jrl-cmakemodules_VERSION}) package.")
  return()
endif()

# Option 4: Fallback to FetchContent
message(STATUS "Fetching jrl-cmakemodules using FetchContent...")
include(FetchContent)
FetchContent_Declare(
  jrl-cmakemodules
  GIT_REPOSITORY https://github.com/ahoarau/jrl-cmakemodules
  GIT_TAG jrl-next
)
FetchContent_MakeAvailable(jrl-cmakemodules)
