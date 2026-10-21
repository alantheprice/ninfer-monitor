cmake_minimum_required(VERSION 3.28)

# Architecture gate. Blackwell (sm_120a) is the reference/default target; the
# RTX-4090 (sm_89) and Ampere (sm_86) targets are supported by re-applying the
# Ada/Ampere kernel layer (see the sm_89 port). Validate the architecture before
# CUDA compiler detection so CMake never injects a generic host-toolchain default
# or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|86|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=120a, 86 or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()
# NVFP4 (Blackwell FP4 tensor cores + TMA/setmaxnreg warp-specialized kernels) is
# only available on sm_120a. On sm_86/89 the groupwise-int path (q4/q5/q6/q8
# row-split) is used and the NVFP4 kernel archives are excluded from the build.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a)$")
  set(NINFER_ENABLE_NVFP4 ON)
  add_compile_definitions(NINFER_SM120A=1)
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^86$")
  set(NINFER_ENABLE_NVFP4 OFF)
  add_compile_definitions(NINFER_SM86=1)
else()
  set(NINFER_ENABLE_NVFP4 OFF)
  add_compile_definitions(NINFER_SM89=1)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
