# Internal headers are private to each compile owner. External headers come from
# the dependency targets (or a component-local include for bundled C sources).
function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/generated)
endfunction()

function(ninfer_cuda_library target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# CUDA 12.x ptxas -O3 promotes large register arrays to static shared memory past
# the 48KB per-kernel cap on sm_100a (nvlink rejects the result). Cap the device
# optimizer at O2 for 12.x; -O2/-O3 differ only in allocation strategy here, not
# in generated math. 13.x keeps its default O3.
if(CUDAToolkit_VERSION VERSION_LESS 12.9)
  # 12.x toolchain guards:
  # - ptxas -O3 promotes large register arrays past the 48KB static shared cap
  #   on sm_100a (nvlink rejects). -O2 keeps them under the cap.
  # - nvcc 12.8's bundled gcc-11 host backend ICEs on the context/reclaim
  #   templates at any optimization level. g++-12 handles the same TUs at -O3.
  #   Set NVCC_HOST_COMPILER=g++-12 (or CMAKE_CUDA_HOST_COMPILER) when 12.x is
  #   used; on Ubuntu 22.04: apt install g++-12.
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O2>)
endif()

function(ninfer_cuda_non_rdc_archive target)
  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()
