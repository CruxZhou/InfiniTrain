#include "infini_train/src/nn/flash_attention_bridge.h"

#include <math.h>
#include <string.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

namespace fa = FLASH_NAMESPACE;
using fa::Flash_bwd_params;
using fa::Flash_fwd_params;


namespace {

constexpr int64_t kFlashMaxHeadDim = 256;
constexpr int kThreadsPerBlock = 256;

static cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

template <typename T>
__device__ float ToFloat(T value);

template <>
__device__ float ToFloat<half>(half value) {
    return __half2float(value);
}

template <>
__device__ float ToFloat<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T FromFloat(float value);

template <>
__device__ half FromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__device__ nv_bfloat16 FromFloat<nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
__global__ void ReduceGroupedHeadsKernel(const T *expanded,
                                         T *reduced,
                                         int64_t batch_size,
                                         int64_t seqlen_k,
                                         int64_t num_heads_q,
                                         int64_t num_heads_kv,
                                         int64_t head_dim) {
    const int64_t group_size = num_heads_q / num_heads_kv;
    const int64_t reduced_numel = batch_size * seqlen_k * num_heads_kv * head_dim;

    for (int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear_idx < reduced_numel;
         linear_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t d = linear_idx % head_dim;
        const int64_t h_kv = (linear_idx / head_dim) % num_heads_kv;
        const int64_t t = (linear_idx / head_dim / num_heads_kv) % seqlen_k;
        const int64_t b = linear_idx / head_dim / num_heads_kv / seqlen_k;

        const int64_t head_base = h_kv * group_size;
        float sum = 0.0f;
        for (int64_t g = 0; g < group_size; ++g) {
            const int64_t h_q = head_base + g;
            const int64_t expanded_idx =
                (((b * seqlen_k + t) * num_heads_q + h_q) * head_dim) + d;
            sum += ToFloat<T>(expanded[expanded_idx]);
        }
        reduced[linear_idx] = FromFloat<T>(sum);
    }
}

void LaunchReduceGroupedHeads(const std::shared_ptr<Tensor> &expanded,
                              const std::shared_ptr<Tensor> &reduced,
                              int64_t batch_size,
                              int64_t seqlen_k,
                              int64_t num_heads_q,
                              int64_t num_heads_kv,
                              int64_t head_dim,
                              DataType dtype,
                              cudaStream_t stream) {
    const int64_t numel = batch_size * seqlen_k * num_heads_kv * head_dim;
    const int blocks = static_cast<int>((numel + kThreadsPerBlock - 1) / kThreadsPerBlock);

    if (dtype == DataType::kFLOAT16) {
        ReduceGroupedHeadsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const half *>(expanded->DataPtr()),
            static_cast<half *>(reduced->DataPtr()),
            batch_size,
            seqlen_k,
            num_heads_q,
            num_heads_kv,
            head_dim);
    } else if (dtype == DataType::kBFLOAT16) {
        ReduceGroupedHeadsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const nv_bfloat16 *>(expanded->DataPtr()),
            static_cast<nv_bfloat16 *>(reduced->DataPtr()),
            batch_size,
            seqlen_k,
            num_heads_q,
            num_heads_kv,
            head_dim);
    } else {
        LOG(FATAL) << "Unsupported dtype for GQA reduction";
    }
}

void CheckCommonInputs(const std::shared_ptr<Tensor> &q,
                       const std::shared_ptr<Tensor> &k,
                       const std::shared_ptr<Tensor> &v,
                       bool enable_gqa) {
    CHECK(q && k && v) << "q, k, v must be non-null";

    // Input layout:
    //   q: (B, T_q, H_q, D)
    //   k: (B, T_k, H_kv, D)
    //   v: (B, T_k, H_kv, D)
    CHECK_EQ(q->Dims().size(), 4) << "Expected q shape (B, T_q, H_q, D)";
    CHECK_EQ(k->Dims().size(), 4) << "Expected k shape (B, T_k, H_kv, D)";
    CHECK_EQ(v->Dims().size(), 4) << "Expected v shape (B, T_k, H_kv, D)";

    CHECK_EQ(q->Dims()[0], k->Dims()[0]) << "Batch size mismatch between q and k";
    CHECK_EQ(q->Dims()[0], v->Dims()[0]) << "Batch size mismatch between q and v";
    CHECK_EQ(k->Dims()[0], v->Dims()[0]) << "Batch size mismatch between k and v";

    CHECK_EQ(k->Dims()[1], v->Dims()[1]) << "k/v sequence length mismatch";
    CHECK_EQ(k->Dims()[2], v->Dims()[2]) << "k/v head count mismatch";
    CHECK_EQ(q->Dims()[3], k->Dims()[3]) << "q/k head_dim mismatch";
    CHECK_EQ(q->Dims()[3], v->Dims()[3]) << "q/v head_dim mismatch";

    const int64_t head_dim = q->Dims()[3];
    const int64_t num_heads_q = q->Dims()[2];
    const int64_t num_heads_kv = k->Dims()[2];

    CHECK(head_dim <= kFlashMaxHeadDim)
        << "FlashAttention only supports head_dim <= 256, got " << head_dim;
    CHECK_EQ(head_dim % 8, 0)
        << "FlashAttention requires head_dim % 8 == 0, got " << head_dim;

    if (enable_gqa) {
        CHECK_GT(num_heads_kv, 0) << "num_heads_kv must be positive";
        CHECK_EQ(num_heads_q % num_heads_kv, 0)
            << "enable_gqa=true requires q_heads % kv_heads == 0";
    } else {
        CHECK_EQ(num_heads_q, num_heads_kv)
            << "enable_gqa=false requires q_heads == kv_heads";
    }

    const auto dtype = q->Dtype();
    CHECK(dtype == DataType::kFLOAT16 || dtype == DataType::kBFLOAT16)
        << "FlashAttention only supports fp16 and bf16";

    CHECK(static_cast<int>(dtype) == static_cast<int>(k->Dtype()))
        << "q/k dtype mismatch, q dtype=" << static_cast<int>(dtype)
        << ", k dtype=" << static_cast<int>(k->Dtype());

    CHECK(static_cast<int>(dtype) == static_cast<int>(v->Dtype()))
        << "q/v dtype mismatch, q dtype=" << static_cast<int>(dtype)
        << ", v dtype=" << static_cast<int>(v->Dtype());

}

void FillFwdStrides(Flash_fwd_params *params,
                    int64_t seqlen_q,
                    int64_t seqlen_k,
                    int64_t num_heads_q,
                    int64_t num_heads_kv,
                    int64_t head_dim) {
    params->q_batch_stride = seqlen_q * num_heads_q * head_dim;
    params->q_row_stride = num_heads_q * head_dim;
    params->q_head_stride = head_dim;

    params->k_batch_stride = seqlen_k * num_heads_kv * head_dim;
    params->k_row_stride = num_heads_kv * head_dim;
    params->k_head_stride = head_dim;

    params->v_batch_stride = seqlen_k * num_heads_kv * head_dim;
    params->v_row_stride = num_heads_kv * head_dim;
    params->v_head_stride = head_dim;

    params->o_batch_stride = seqlen_q * num_heads_q * head_dim;
    params->o_row_stride = num_heads_q * head_dim;
    params->o_head_stride = head_dim;
}

void FillBwdCommonStrides(Flash_bwd_params *params,
                          int64_t seqlen_q,
                          int64_t seqlen_k,
                          int64_t num_heads_q,
                          int64_t num_heads_kv,
                          int64_t head_dim) {
    params->q_batch_stride = seqlen_q * num_heads_q * head_dim;
    params->q_row_stride = num_heads_q * head_dim;
    params->q_head_stride = head_dim;

    params->k_batch_stride = seqlen_k * num_heads_kv * head_dim;
    params->k_row_stride = num_heads_kv * head_dim;
    params->k_head_stride = head_dim;

    params->v_batch_stride = seqlen_k * num_heads_kv * head_dim;
    params->v_row_stride = num_heads_kv * head_dim;
    params->v_head_stride = head_dim;

    params->o_batch_stride = seqlen_q * num_heads_q * head_dim;
    params->o_row_stride = num_heads_q * head_dim;
    params->o_head_stride = head_dim;

    params->do_batch_stride = seqlen_q * num_heads_q * head_dim;
    params->do_row_stride = num_heads_q * head_dim;
    params->do_head_stride = head_dim;

    params->dq_batch_stride = seqlen_q * num_heads_q * head_dim;
    params->dq_row_stride = num_heads_q * head_dim;
    params->dq_head_stride = head_dim;
}

}  // namespace

// ---------------------------------------------------------------------------
// FlashAttention Forward
// ---------------------------------------------------------------------------
//
// Input layout:
//   q : (B, T_q, H_q, D)
//   k : (B, T_k, H_kv, D)
//   v : (B, T_k, H_kv, D)
//
// Returns:
//   {output (B, T_q, H_q, D), softmax_lse (B, H_q, T_q)}
//
std::vector<std::shared_ptr<Tensor>> FlashAttentionForward(
    const std::shared_ptr<Tensor> &q,
    const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v,
    const std::shared_ptr<Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    double scale,
    bool enable_gqa) {
    CheckCommonInputs(q, k, v, enable_gqa);

    CHECK(attn_mask == nullptr)
        << "FlashAttention CUDA path currently does not support arbitrary attn_mask";
    CHECK_EQ(dropout_p, 0.0)
        << "FlashAttention CUDA path currently requires dropout_p == 0";

    const auto &q_dims = q->Dims();
    const auto &k_dims = k->Dims();

    const int64_t batch_size = q_dims[0];
    const int64_t seqlen_q = q_dims[1];
    const int64_t num_heads_q = q_dims[2];
    const int64_t head_dim = q_dims[3];

    const int64_t seqlen_k = k_dims[1];
    const int64_t num_heads_kv = k_dims[2];

    const auto device = q->GetDevice();
    const auto dtype = q->Dtype();
    const cudaStream_t stream = GetCudaStream(device);

    const int64_t seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;
    const int64_t seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;
    const int64_t d_rounded = ((head_dim + 31) / 32) * 32;

    auto output = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, seqlen_q, num_heads_q, head_dim},
        dtype,
        device);

    auto softmax_lse = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, num_heads_q, seqlen_q_rounded},
        DataType::kFLOAT32,
        device);

    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = q->DataPtr();
    params.k_ptr = k->DataPtr();
    params.v_ptr = v->DataPtr();
    params.o_ptr = output->DataPtr();
    params.softmax_lse_ptr = softmax_lse->DataPtr();

    params.b = batch_size;
    params.h = num_heads_q;
    params.h_k = num_heads_kv;
    params.h_h_k_ratio = num_heads_q / num_heads_kv;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = head_dim;
    params.d_rounded = d_rounded;

    params.scale_softmax = static_cast<float>(scale);
    params.scale_softmax_log2 = static_cast<float>(scale * M_LOG2E);
    params.is_causal = is_causal;
    params.is_bf16 = (dtype == DataType::kBFLOAT16);

    FillFwdStrides(&params, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim);

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    if (is_causal) {
        params.window_size_left = -1;
        params.window_size_right = 0;
    } else {
        params.window_size_left = -1;
        params.window_size_right = -1;
    }

    params.is_seqlens_k_cumulative = true;

    fa::run_mha_fwd(params, stream);

    return {output, softmax_lse};
}

// ---------------------------------------------------------------------------
// FlashAttention Backward
// ---------------------------------------------------------------------------
//
// Inputs:
//   grad_output : (B, T_q, H_q, D)
//   q           : (B, T_q, H_q, D)
//   k           : (B, T_k, H_kv, D)
//   v           : (B, T_k, H_kv, D)
//   output      : (B, T_q, H_q, D)
//   softmax_lse : (B, H_q, T_q)
//
// Returns:
//   {dq (B, T_q, H_q, D), dk (B, T_k, H_kv, D), dv (B, T_k, H_kv, D)}
//
std::vector<std::shared_ptr<Tensor>> FlashAttentionBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::shared_ptr<Tensor> &q,
    const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v,
    const std::shared_ptr<Tensor> &output,
    const std::shared_ptr<Tensor> &softmax_lse,
    double dropout_p,
    bool is_causal,
    double scale,
    bool enable_gqa) {
    CheckCommonInputs(q, k, v, enable_gqa);

    CHECK(grad_output) << "grad_output must be non-null";
    CHECK(output) << "output must be non-null";
    CHECK(softmax_lse) << "softmax_lse must be non-null";

    CHECK_EQ(dropout_p, 0.0)
        << "FlashAttention CUDA path currently requires dropout_p == 0";

    const auto &q_dims = q->Dims();
    const auto &k_dims = k->Dims();

    const int64_t batch_size = q_dims[0];
    const int64_t seqlen_q = q_dims[1];
    const int64_t num_heads_q = q_dims[2];
    const int64_t head_dim = q_dims[3];

    const int64_t seqlen_k = k_dims[1];
    const int64_t num_heads_kv = k_dims[2];

    const auto device = q->GetDevice();
    const auto dtype = q->Dtype();
    const cudaStream_t stream = GetCudaStream(device);

    CHECK_EQ(grad_output->Dims().size(), 4) << "grad_output must be rank-4";
    CHECK_EQ(output->Dims().size(), 4) << "output must be rank-4";
    CHECK_EQ(softmax_lse->Dims().size(), 3) << "softmax_lse must be rank-3";

    CHECK_EQ(grad_output->Dims()[0], batch_size);
    CHECK_EQ(grad_output->Dims()[1], seqlen_q);
    CHECK_EQ(grad_output->Dims()[2], num_heads_q);
    CHECK_EQ(grad_output->Dims()[3], head_dim);

    CHECK_EQ(output->Dims()[0], batch_size);
    CHECK_EQ(output->Dims()[1], seqlen_q);
    CHECK_EQ(output->Dims()[2], num_heads_q);
    CHECK_EQ(output->Dims()[3], head_dim);

    CHECK_EQ(softmax_lse->Dims()[0], batch_size);
    CHECK_EQ(softmax_lse->Dims()[1], num_heads_q);
    

    auto dq = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, seqlen_q, num_heads_q, head_dim},
        dtype,
        device);

    auto dk = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, seqlen_k, num_heads_kv, head_dim},
        dtype,
        device);

    auto dv = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, seqlen_k, num_heads_kv, head_dim},
        dtype,
        device);

    std::shared_ptr<Tensor> dk_expanded = dk;
    std::shared_ptr<Tensor> dv_expanded = dv;
    const bool use_gqa_reduce = (num_heads_q != num_heads_kv);

    if (use_gqa_reduce) {
        dk_expanded = std::make_shared<Tensor>(
            std::vector<int64_t>{batch_size, seqlen_k, num_heads_q, head_dim},
            dtype,
            device);
        dv_expanded = std::make_shared<Tensor>(
            std::vector<int64_t>{batch_size, seqlen_k, num_heads_q, head_dim},
            dtype,
            device);
    }

    const int64_t seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;
    const int64_t seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;
    const int64_t d_rounded = ((head_dim + 31) / 32) * 32;

    CHECK_EQ(softmax_lse->Dims()[2], seqlen_q_rounded);


    auto dsoftmax = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, num_heads_q, seqlen_q_rounded},
        DataType::kFLOAT32,
        device);

    auto dq_accum = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, seqlen_q_rounded, num_heads_q, d_rounded},
        DataType::kFLOAT32,
        device);

    cudaMemsetAsync(
        dq_accum->DataPtr(),
        0,
        batch_size * seqlen_q_rounded * num_heads_q * d_rounded * sizeof(float),
        stream);
    cudaMemsetAsync(
        dsoftmax->DataPtr(),
        0,
        batch_size * num_heads_q * seqlen_q_rounded * sizeof(float),
        stream);

    Flash_bwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = q->DataPtr();
    params.k_ptr = k->DataPtr();
    params.v_ptr = v->DataPtr();
    params.o_ptr = output->DataPtr();
    params.do_ptr = grad_output->DataPtr();

    params.dq_ptr = dq->DataPtr();
    params.dk_ptr = dk_expanded->DataPtr();
    params.dv_ptr = dv_expanded->DataPtr();

    params.softmax_lse_ptr = softmax_lse->DataPtr();
    params.dsoftmax_sum = dsoftmax->DataPtr();
    params.dq_accum_ptr = dq_accum->DataPtr();

    params.b = batch_size;
    params.h = num_heads_q;
    params.h_k = num_heads_kv;
    params.h_h_k_ratio = num_heads_q / num_heads_kv;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = head_dim;
    params.d_rounded = d_rounded;

    params.scale_softmax = static_cast<float>(scale);
    params.scale_softmax_log2 = static_cast<float>(scale * M_LOG2E);
    params.is_causal = is_causal;
    params.is_bf16 = (dtype == DataType::kBFLOAT16);

    FillBwdCommonStrides(&params, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim);

    const int64_t dk_dv_num_heads = use_gqa_reduce ? num_heads_q : num_heads_kv;
    params.dk_batch_stride = seqlen_k * dk_dv_num_heads * head_dim;
    params.dk_row_stride = dk_dv_num_heads * head_dim;
    params.dk_head_stride = head_dim;

    params.dv_batch_stride = seqlen_k * dk_dv_num_heads * head_dim;
    params.dv_row_stride = dk_dv_num_heads * head_dim;
    params.dv_head_stride = head_dim;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    if (is_causal) {
        params.window_size_left = -1;
        params.window_size_right = 0;
    } else {
        params.window_size_left = -1;
        params.window_size_right = -1;
    }

    params.is_seqlens_k_cumulative = true;
    params.deterministic = false;
    params.dq_accum_split_stride = 0;

    size_t elem_size = (dtype == DataType::kBFLOAT16) ? sizeof(nv_bfloat16) : sizeof(half);

    cudaMemsetAsync(
        dq->DataPtr(),
        0,
        batch_size * seqlen_q * num_heads_q * head_dim * elem_size,
        stream);

    cudaMemsetAsync(
        dk_expanded->DataPtr(),
        0,
        batch_size * seqlen_k * dk_dv_num_heads * head_dim * elem_size,
        stream);

    cudaMemsetAsync(
        dv_expanded->DataPtr(),
        0,
        batch_size * seqlen_k * dk_dv_num_heads * head_dim * elem_size,
        stream);


    cudaMemsetAsync(
        dk->DataPtr(),
        0,
        batch_size * seqlen_k * dk_dv_num_heads * head_dim * elem_size,
        stream);

    cudaMemsetAsync(
        dv->DataPtr(),
        0,
        batch_size * seqlen_k * dk_dv_num_heads * head_dim * elem_size,
        stream);

    fa::run_mha_bwd(params, stream);

    if (use_gqa_reduce) {
        LaunchReduceGroupedHeads(
            dk_expanded,
            dk,
            batch_size,
            seqlen_k,
            num_heads_q,
            num_heads_kv,
            head_dim,
            dtype,
            stream);

        LaunchReduceGroupedHeads(
            dv_expanded,
            dv,
            batch_size,
            seqlen_k,
            num_heads_q,
            num_heads_kv,
            head_dim,
            dtype,
            stream);
    }

    return {dq, dk, dv};
}

}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FLASH_ATTENTION_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_FLASH_ATTENTION_KERNEL