cmake_minimum_required(VERSION 3.28)

option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running" ON)

project(infini_train VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Third-party deps
# ------------------------------------------------------------------------------

# gflags
add_subdirectory(third_party/gflags)
include_directories(${gflags_SOURCE_DIR}/include)

# glog
set(WITH_GFLAGS OFF CACHE BOOL "Disable glog finding system gflags" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Disable glog finding system gtest" FORCE)
add_subdirectory(third_party/glog)
include_directories(${glog_SOURCE_DIR}/src)

# eigen
if(USE_OMP)
  find_package(OpenMP REQUIRED)
endif()
add_subdirectory(third_party/eigen)
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)

# Common project include
include_directories(${PROJECT_SOURCE_DIR})

# ------------------------------------------------------------------------------
# PyTorch discovery for FlashAttention vendor kernels
# ------------------------------------------------------------------------------
# Dao-AILab FlashAttention csrc/*.cu is not pure CUDA-only code. It includes
# ATen/c10 headers from PyTorch, so when USE_CUDA=ON we must discover Torch.

if(USE_CUDA)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # Allow manual override:
  #   -DTorch_DIR=/path/to/torch/share/cmake/Torch
  # or
  #   -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake
  if(NOT Torch_DIR)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_FROM_PYTHON
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE TORCH_PREFIX_QUERY_RESULT
    )

    if(NOT TORCH_PREFIX_QUERY_RESULT EQUAL 0 OR TORCH_CMAKE_PREFIX_FROM_PYTHON STREQUAL "")
      message(FATAL_ERROR
        "Failed to query torch.utils.cmake_prefix_path from Python.\n"
        "Please activate the Python environment that contains torch, or pass:\n"
        "  -DTorch_DIR=/path/to/torch/share/cmake/Torch\n"
        "or\n"
        "  -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake")
    endif()

    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_FROM_PYTHON}")
    set(Torch_DIR "${TORCH_CMAKE_PREFIX_FROM_PYTHON}/Torch"
        CACHE PATH "Path to TorchConfig.cmake" FORCE)
  endif()

  find_package(Torch REQUIRED)

  # Torch often exports ABI-related flags such as -D_GLIBCXX_USE_CXX11_ABI=...
  set(TORCH_CXX_FLAGS_LIST "")
  if(DEFINED TORCH_CXX_FLAGS AND NOT TORCH_CXX_FLAGS STREQUAL "")
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  endif()

  message(STATUS "Torch_DIR = ${Torch_DIR}")
  message(STATUS "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")
  message(STATUS "TORCH_CXX_FLAGS = ${TORCH_CXX_FLAGS}")
endif()

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

# Framework core sources (*.cc), excluding cpu kernels (they are built separately)
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
list(FILTER SRC EXCLUDE REGEX ".*kernels/cpu/.*")

if(NOT USE_NCCL)
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/ccl/cuda/.*")
endif()

# CPU kernels (*.cc)
file(GLOB_RECURSE CPU_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cpu/*.cc)

# CUDA sources split into:
#   1) project CUDA kernels
#   2) FlashAttention vendor kernels + project dispatch wrapper

if(USE_CUDA)
  file(GLOB_RECURSE PROJECT_CUDA_KERNELS
    ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu
  )

  # This file is the bridge/dispatch wrapper for vendor FlashAttention templates.
  # Build it together with vendor kernels, not with the general project CUDA kernels.
  set(FLASHATTN_DISPATCH_CU
    ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cuda/flash_attention_dispatch.cu
  )
  list(REMOVE_ITEM PROJECT_CUDA_KERNELS "${FLASHATTN_DISPATCH_CU}")

  # Dao-AILab FlashAttention-2 kernel instantiations.
  # Do NOT add flash_api.cpp here; we are not using the PyTorch C++ wrapper path.
  file(GLOB FLASHATTN_VENDOR_KERNELS
    ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/flash_attn/src/*.cu
  )
endif()

# ------------------------------------------------------------------------------
# CPU kernels library
# ------------------------------------------------------------------------------

add_library(infini_train_cpu_kernels STATIC ${CPU_KERNELS})

target_link_libraries(infini_train_cpu_kernels
  PUBLIC
    glog
    Eigen3::Eigen
)

if(USE_OMP)
  target_compile_definitions(infini_train_cpu_kernels PUBLIC USE_OMP=1)
  target_link_libraries(infini_train_cpu_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

if(PROFILE_MODE)
  target_compile_definitions(infini_train_cpu_kernels PUBLIC PROFILE_MODE=1)
endif()

# ------------------------------------------------------------------------------
# CUDA kernels library (project kernels only)
# ------------------------------------------------------------------------------

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # Project kernels target: NO vendor-only fast-math flags here.
  add_library(infini_train_cuda_kernels STATIC ${PROJECT_CUDA_KERNELS})

  set_target_properties(infini_train_cuda_kernels PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )

  target_compile_definitions(infini_train_cuda_kernels PUBLIC USE_CUDA=1)
  if(PROFILE_MODE)
    target_compile_definitions(infini_train_cuda_kernels PUBLIC PROFILE_MODE=1)
  endif()

  target_include_directories(infini_train_cuda_kernels
    PUBLIC
      ${PROJECT_SOURCE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${TORCH_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/flash_attn
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/flash_attn/src
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/cutlass/include
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/cutlass/tools/util/include
  )

  # Keep ABI flags for units that include Torch/c10 headers, but do NOT use vendor-only
  # fast-math / half-unmask flags here.
  if(TORCH_CXX_FLAGS_LIST)
    target_compile_options(infini_train_cuda_kernels PRIVATE ${TORCH_CXX_FLAGS_LIST})
  endif()

  target_compile_options(infini_train_cuda_kernels
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
      >
  )

  target_link_libraries(infini_train_cuda_kernels
    PUBLIC
      glog
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
  )

  # --------------------------------------------------------------------------
  # FlashAttention vendor library (vendor kernels + project dispatch wrapper)
  # --------------------------------------------------------------------------

  add_library(flashattn_vendor_kernels STATIC
    ${FLASHATTN_VENDOR_KERNELS}
    ${FLASHATTN_DISPATCH_CU}
  )

  set_target_properties(flashattn_vendor_kernels PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )

  target_compile_definitions(flashattn_vendor_kernels PUBLIC USE_CUDA=1)
  if(PROFILE_MODE)
    target_compile_definitions(flashattn_vendor_kernels PUBLIC PROFILE_MODE=1)
  endif()

  target_include_directories(flashattn_vendor_kernels
    PUBLIC
      ${PROJECT_SOURCE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${TORCH_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/flash_attn
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/flash_attn/src
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/cutlass/include
      ${PROJECT_SOURCE_DIR}/third_party/Dao-AILab-flash-attention/csrc/cutlass/tools/util/include
  )

  target_compile_options(flashattn_vendor_kernels
    PRIVATE
      ${TORCH_CXX_FLAGS_LIST}
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_BFLOAT16_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__
        -U__CUDA_NO_BFLOAT16_OPERATORS__
      >
  )

  target_link_libraries(flashattn_vendor_kernels
    PUBLIC
      glog
      ${TORCH_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
  )

  # Project CUDA kernels depend on the vendor FlashAttention symbols.
  target_link_libraries(infini_train_cuda_kernels
    PUBLIC
      flashattn_vendor_kernels
  )

  if(USE_NCCL)
    message(STATUS "Add USE_NCCL, use NCCL with CUDA")
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(NCCL REQUIRED)

    target_compile_definitions(infini_train_cuda_kernels PUBLIC USE_NCCL=1)
    target_compile_definitions(flashattn_vendor_kernels PUBLIC USE_NCCL=1)

    target_link_libraries(infini_train_cuda_kernels PUBLIC nccl)
    target_link_libraries(flashattn_vendor_kernels PUBLIC nccl)
  endif()
endif()

# ------------------------------------------------------------------------------
# Main framework library
# ------------------------------------------------------------------------------

add_library(infini_train STATIC ${SRC})

target_link_libraries(infini_train
  PUBLIC
    glog
    gflags
    infini_train_cpu_kernels
)

if(PROFILE_MODE)
  target_compile_definitions(infini_train PUBLIC PROFILE_MODE=1)
endif()

if(USE_CUDA)
  target_compile_definitions(infini_train PUBLIC USE_CUDA=1)
  target_include_directories(infini_train PUBLIC ${TORCH_INCLUDE_DIRS})

  if(TORCH_CXX_FLAGS_LIST)
    target_compile_options(infini_train PRIVATE ${TORCH_CXX_FLAGS_LIST})
  endif()

  target_link_libraries(infini_train
    PUBLIC
      infini_train_cuda_kernels
      flashattn_vendor_kernels
      ${TORCH_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
  )

  if(USE_NCCL)
    target_link_libraries(infini_train PUBLIC nccl)
    target_compile_definitions(infini_train PUBLIC USE_NCCL=1)
  endif()
endif()

# ------------------------------------------------------------------------------
# Helper: link libraries in a group to fix static lib one-pass resolution
# ------------------------------------------------------------------------------

function(link_infini_train_exe target_name)
  if(USE_CUDA)
    target_link_libraries(${target_name} PRIVATE
      "-Wl,--start-group"
      "-Wl,--whole-archive"
        infini_train
        infini_train_cpu_kernels
        infini_train_cuda_kernels
        flashattn_vendor_kernels
      "-Wl,--no-whole-archive"
      "-Wl,--end-group"
      ${TORCH_LIBRARIES}
    )
  else()
    target_link_libraries(${target_name} PRIVATE
      "-Wl,--start-group"
      "-Wl,--whole-archive"
        infini_train
        infini_train_cpu_kernels
      "-Wl,--no-whole-archive"
      "-Wl,--end-group"
    )
  endif()
endfunction()

# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------

add_executable(mnist
  example/mnist/main.cc
  example/mnist/dataset.cc
  example/mnist/net.cc
)
link_infini_train_exe(mnist)

add_executable(gpt2
  example/gpt2/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/gpt2/net.cc
  example/common/tokenizer.cc
)
link_infini_train_exe(gpt2)

add_executable(llama3
  example/llama3/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/llama3/net.cc
  example/common/tokenizer.cc
)
link_infini_train_exe(llama3)

# Tools
add_subdirectory(tools/infini_run)
set_target_properties(infini_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Tests
add_executable(test_hook test/hook/test_hook.cc)
link_infini_train_exe(test_hook)

add_executable(test_precision_check test/hook/test_precision_check.cc)
link_infini_train_exe(test_precision_check)
