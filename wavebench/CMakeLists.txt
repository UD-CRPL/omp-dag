#==============================================================================

cmake_minimum_required(VERSION 2.8)

project("wavebench")

SET(INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/src/main
  ${CMAKE_SOURCE_DIR}/src/algorithms
  )

SET(SOURCES
  src/main/arguments.c
  src/main/dimensions.c
  src/algorithms/radiation_transport.cpp
  src/algorithms/local_sequence_alignment.cpp
  src/algorithms/gauss_seidel.cpp
  )

INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
ADD_LIBRARY(wavebench_lib STATIC ${SOURCES})
ADD_EXECUTABLE(wavebench src/main/wavebench.cpp)
TARGET_LINK_LIBRARIES(wavebench wavebench_lib)

install(TARGETS wavebench DESTINATION bin)

SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

MESSAGE("Compiler ID: ${CMAKE_C_COMPILER_ID}")
