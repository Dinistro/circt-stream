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
if (EXISTS ${VIVADO_PATH})
  get_filename_component(VIVADO_PATH ${VIVADO_PATH} DIRECTORY)
  message(STATUS "Setting vivado path to ${VIVADO_PATH}.")
else()
  # Search for vivado's `vivado` command.
  find_program(VIVADO_PATH "vivado")
  if(EXISTS ${VIVADO_PATH})
    # Then strip the filename.
    get_filename_component(VIVADO_PATH ${VIVADO_PATH} DIRECTORY)
    message(STATUS "Found vivado at ${VIVADO_PATH}.")
  else()
    set(VIVADO_PATH "")
    message(STATUS "Did not find vivado.")
  endif()
endif()

if (NOT DEFINED VERILATOR_PATH)
  find_program(VERILATOR_PATH "verilator" PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/bin" NO_DEFAULT_PATH)
  find_program(VERILATOR_PATH "verilator")
endif()

if(EXISTS ${VERILATOR_PATH})
  message(STATUS "Found Verilator at ${VERILATOR_PATH}.")

  # Find Verilator version.
  execute_process(COMMAND ${VERILATOR_PATH} --version
    OUTPUT_VARIABLE VERILATOR_VERSION)
  string(REGEX MATCH "Verilator (([0-9]+)\.([0-9]+)) \.*"
    MATCH ${VERILATOR_VERSION})
  # It's gotta be at least v4.110.
  if (${CMAKE_MATCH_1} LESS 4.110)
    message(FATAL_ERROR "CIRCT only supports Verilator version 4.110 and up. \
                         Found version: ${CMAKE_MATCH_1}. You can disable \
                         the Verilator tests with '-DVERILATOR_DISABLE=ON'.")
    set(VERILATOR_PATH "")
  endif()
else()
  set(VERILATOR_PATH "")
  message(STATUS "Did not find Verilator.")
endif()

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
