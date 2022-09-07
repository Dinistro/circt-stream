cmake_minimum_required(VERSION 3.13.4)
project(circt-stream LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)


set(PROJECT_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

#-------------------------------------------------------------------------------
# Icarus Verilog Configuration
#-------------------------------------------------------------------------------

# If Icarus Verilog hasn't been explicitly disabled, find it.
option(IVERILOG_DISABLE "Disable the Icarus Verilog tests.")
if (IVERILOG_DISABLE)
  message(STATUS "Disabling Icarus Verilog tests.")
else()
  find_program(IVERILOG_PATH "iverilog")
  if(EXISTS ${IVERILOG_PATH})
    # Find iverilog version.
    execute_process(COMMAND ${IVERILOG_PATH} -V
      OUTPUT_VARIABLE IVERILOG_VERSION)

    string(REGEX MATCH "Icarus Verilog version (([0-9]+)\.([0-9]+)) \.*"
      MATCH ${IVERILOG_VERSION})

    if (${CMAKE_MATCH_1} LESS 11.0)
      message(FATAL_ERROR "CIRCT only supports Icarus Verilog version 11.0 and up. \
                           Found version: ${CMAKE_MATCH_1}. You can disable \
                           the Icarus Verilog tests with '-DIVERILOG_DISABLE=ON'.")
      set(IVERILOG_PATH "")
    endif()
    message(STATUS "Found iverilog at ${IVERILOG_PATH}.")
  else()
    set(IVERILOG_PATH "")
    message(STATUS "Did not find iverilog.")
  endif()
endif()


find_package(Python3)
if(Python3_FOUND)
  message(STATUS "Found python at ${Python3_EXECUTABLE}")
endif()


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})



include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(integration_test)
add_subdirectory(tools)
