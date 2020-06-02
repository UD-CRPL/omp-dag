#==============================================================================

cmake_minimum_required(VERSION 2.8)

project("wavebench")

SET(INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/src/main
  ${CMAKE_SOURCE_DIR}/src/algorithms
  ${CMAKE_SOURCE_DIR}/include
  )

SET(SOURCES
  src/main/arguments.c
  src/main/dimensions.c
  src/algorithms/radiation_transport.cpp
  src/algorithms/local_sequence_alignment.cpp
  src/algorithms/gauss_seidel.cpp
  include/wavefront/smith-waterman/dependency-graph.cpp
  include/linear-algebra/cholesky/super-nodes.cpp
  include/core/util/cuda-util.cpp
  include/core/debug/timing.cpp
  include/core/io/std-io.cpp
  include/omp_dag_timing.cpp
  )

set(CMAKE_CXX_COMPILER g++-9)

INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
ADD_LIBRARY(wavebench_lib STATIC ${SOURCES})
ADD_EXECUTABLE(wavebench src/main/wavebench.cpp)
TARGET_LINK_LIBRARIES(wavebench gomp pthread wavebench_lib)

install(TARGETS wavebench DESTINATION bin)

SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

MESSAGE("Compiler ID: ${CMAKE_C_COMPILER_ID}")
