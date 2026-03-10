#pragma once

#include <cuda_runtime.h>

#include "third_party/Dao-AILab-flash-attention/csrc/flash_attn/src/namespace_config.h"
#include "third_party/Dao-AILab-flash-attention/csrc/flash_attn/src/flash.h"

namespace FLASH_NAMESPACE {

void run_mha_fwd(Flash_fwd_params &params,
                 cudaStream_t stream,
                 bool force_split_kernel = false);

void run_mha_bwd(Flash_bwd_params &params,
                 cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
