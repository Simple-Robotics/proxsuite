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
    DEBUG
    "JRL_CMAKEMODULES_SOURCE_DIR variable set, adding jrl-cmakemodules from source directory: ${JRL_CMAKEMODULES_SOURCE_DIR}"
  )
  add_subdirectory(${JRL_CMAKEMODULES_SOURCE_DIR} jrl-cmakemodules)
  return()
endif()

# Option 2: use JRL_CMAKEMODULES_SOURCE_DIR environment variable (pixi might unset it, prefer option 1)
if(ENV{JRL_CMAKEMODULES_SOURCE_DIR})
  message(
    DEBUG
    "JRL_CMAKEMODULES_SOURCE_DIR environement variable set, adding jrl-cmakemodules from source directory: ${JRL_CMAKEMODULES_SOURCE_DIR}"
  )
  add_subdirectory(${JRL_CMAKEMODULES_SOURCE_DIR} jrl-cmakemodules)
  return()
endif()

# Try to look for the installed package
message(DEBUG "Looking for jrl-cmakemodules package...")
find_package(jrl-cmakemodules CONFIG QUIET)

# If we have the package, we are done.
if(jrl-cmakemodules_FOUND)
  message(DEBUG "Found jrl-cmakemodules package.")
  return()
endif()

# Fallback to FetchContent if not found
message(DEBUG "Fetching jrl-cmakemodules using FetchContent...")
include(FetchContent)
FetchContent_Declare(
  jrl-cmakemodules
  GIT_REPOSITORY https://github.com/ahoarau/jrl-cmakemodules-v2
  GIT_TAG main
)
FetchContent_MakeAvailable(jrl-cmakemodules)
