option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
if(ENABLE_CPPCHECK)
  find_program(CPPCHECK cppcheck)
  if(CPPCHECK)
    set(CMAKE_CXX_CPPCHECK
        ${CPPCHECK}
        --suppress=missingInclude
        --enable=all
        --inconclusive
        -i
        ${CMAKE_SOURCE_DIR}/imgui/lib
    )
  else()
    message(SEND_ERROR "cppcheck requested but executable not found")
  endif()
endif()

if(ENABLE_CLANG_TIDY AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  find_program(CLANGTIDY clang-tidy)
  find_program(UNBUFFER unbuffer)
  if(CLANGTIDY)
    if(UNBUFFER)
      set(CMAKE_CXX_CLANG_TIDY unbuffer ${CLANGTIDY})
    else()
      message(WARNING "unbuffer not found")
      set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
    endif()
  else()
    message(SEND_ERROR "clang-tidy requested but executable not found")
  endif()
endif()
