#include <cmath>
#include <cstddef>

#include <cub/block/block_reduce.cuh>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/cub_compat.cuh"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::kernels::cuda {

// q,k,v,y : (B, T, H, D) contiguous in row-major
// index = (((b*T + t)*H + h)*D + d)
__device__ __forceinline__ int64_t Idx4(int64_t b, int64_t t, int64_t h, int64_t d,
                                        int64_t T, int64_t H, int64_t D) {
    return (((b * T + t) * H + h) * D + d);
}

template <typename T>
__device__ __forceinline__ float LoadAsFloat(const T* ptr) {
    return common::cuda::Cast<float>(*ptr);
}

template <typename T>
__device__ __forceinline__ T StoreFromFloat(float x) {
    return common::cuda::Cast<T>(x);
}

// --------------------------- Forward ---------------------------
// One thread computes one output element y[b,t,h,d].
// It recomputes softmax over s by scanning s twice (max, sum).
template <typename T>
__global__ void SDPAForwardKernel(T* y,
                                 const T* q,
                                 const T* k,
                                 const T* v,
                                 int64_t B, int64_t Tlen, int64_t H, int64_t D,
                                 float scale,
                                 bool is_causal) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * Tlen * H * D;
    if (linear >= total) return;

    int64_t d = linear % D;
    int64_t tmp = linear / D;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t t = tmp % Tlen;
    int64_t b = tmp / Tlen;

    // max score over s
    float max_score = -INFINITY;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;

        float dot = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            const float qv = LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]);
            const float kv = LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
            dot += qv * kv;
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    // sum exp
    float sum_exp = 0.0f;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;

        float dot = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            const float qv = LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]);
            const float kv = LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
            dot += qv * kv;
        }
        float score = dot * scale;
        sum_exp += expf(score - max_score);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    // output
    float out = 0.0f;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;

        float dot = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            const float qv = LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]);
            const float kv = LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
            dot += qv * kv;
        }
        float score = dot * scale;
        float p = expf(score - max_score) * inv_sum;

        const float vv = LoadAsFloat(&v[Idx4(b, s, h, d, Tlen, H, D)]);
        out += p * vv;
    }

    y[Idx4(b, t, h, d, Tlen, H, D)] = StoreFromFloat<T>(out);
}

// --------------------------- Backward helpers ---------------------------
// For a fixed (b,t,h), we need:
// P_s = softmax(score_s)
// dP_s = sum_d dy_d * v_s_d
// sum_j dP_j * P_j
// dS_s = P_s * (dP_s - sum_j dP_j*P_j)
// Then:
// dq[b,t,h,d] = sum_s dS_s * scale * k[b,s,h,d]
// dk[b,s,h,d] = sum_t dS[b,t,h,s] * scale * q[b,t,h,d]
// dv[b,s,h,d] = sum_t P[b,t,h,s] * dy[b,t,h,d]
//
// To avoid atomics, we compute dq, dk, dv in separate kernels with different output indexing.
// This is slow but simple & deterministic.

// Compute dq: parallel over (b,t,h,d)
template <typename T>
__global__ void SDPABwdDQKernel(T* dq,
                               const T* dy,
                               const T* q,
                               const T* k,
                               const T* v,
                               int64_t B, int64_t Tlen, int64_t H, int64_t D,
                               float scale,
                               bool is_causal) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * Tlen * H * D;
    if (linear >= total) return;

    int64_t d = linear % D;
    int64_t tmp = linear / D;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t t = tmp % Tlen;
    int64_t b = tmp / Tlen;

    // max score
    float max_score = -INFINITY;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;
        float dot = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dot += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                   LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    // sum exp
    float sum_exp = 0.0f;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;
        float dot = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dot += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                   LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        float score = dot * scale;
        sum_exp += expf(score - max_score);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    // compute sum_j dP_j * P_j
    float sum_dP_mul_P = 0.0f;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;

        float dot_qk = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dot_qk += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                      LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        float p = expf(dot_qk * scale - max_score) * inv_sum;

        float dP = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dP += LoadAsFloat(&dy[Idx4(b, t, h, dd, Tlen, H, D)]) *
                  LoadAsFloat(&v[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        sum_dP_mul_P += dP * p;
    }

    // dq element
    float dq_val = 0.0f;
    for (int64_t s = 0; s < Tlen; ++s) {
        if (is_causal && s > t) continue;

        float dot_qk = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dot_qk += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                      LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        float p = expf(dot_qk * scale - max_score) * inv_sum;

        float dP = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dP += LoadAsFloat(&dy[Idx4(b, t, h, dd, Tlen, H, D)]) *
                  LoadAsFloat(&v[Idx4(b, s, h, dd, Tlen, H, D)]);
        }

        float dS = p * (dP - sum_dP_mul_P);

        // derivative wrt q includes scale factor (since score = scale * dot(q,k))
        dq_val += dS * scale * LoadAsFloat(&k[Idx4(b, s, h, d, Tlen, H, D)]);
    }

    dq[Idx4(b, t, h, d, Tlen, H, D)] = StoreFromFloat<T>(dq_val);
}

// Compute dk: parallel over (b,s,h,d), sum over t
template <typename T>
__global__ void SDPABwdDKKernel(T* dk,
                               const T* dy,
                               const T* q,
                               const T* k,
                               const T* v,
                               int64_t B, int64_t Tlen, int64_t H, int64_t D,
                               float scale,
                               bool is_causal) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * Tlen * H * D; // dk has same total elements
    if (linear >= total) return;

    int64_t d = linear % D;
    int64_t tmp = linear / D;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t s = tmp % Tlen;
    int64_t b = tmp / Tlen;

    float dk_val = 0.0f;

    for (int64_t t = 0; t < Tlen; ++t) {
        if (is_causal && s > t) continue;

        // recompute softmax stats for (b,t,h)
        float max_score = -INFINITY;
        for (int64_t ss = 0; ss < Tlen; ++ss) {
            if (is_causal && ss > t) continue;
            float dot = 0.0f;
            for (int64_t dd = 0; dd < D; ++dd) {
                dot += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                       LoadAsFloat(&k[Idx4(b, ss, h, dd, Tlen, H, D)]);
            }
            float score = dot * scale;
            if (score > max_score) max_score = score;
        }
        float sum_exp = 0.0f;
        for (int64_t ss = 0; ss < Tlen; ++ss) {
            if (is_causal && ss > t) continue;
            float dot = 0.0f;
            for (int64_t dd = 0; dd < D; ++dd) {
                dot += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                       LoadAsFloat(&k[Idx4(b, ss, h, dd, Tlen, H, D)]);
            }
            sum_exp += expf(dot * scale - max_score);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // sum_j dP_j * P_j for this (b,t,h)
        float sum_dP_mul_P = 0.0f;
        for (int64_t ss = 0; ss < Tlen; ++ss) {
            if (is_causal && ss > t) continue;

            float dot_qk = 0.0f;
            for (int64_t dd = 0; dd < D; ++dd) {
                dot_qk += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                          LoadAsFloat(&k[Idx4(b, ss, h, dd, Tlen, H, D)]);
            }
            float p = expf(dot_qk * scale - max_score) * inv_sum;

            float dP = 0.0f;
            for (int64_t dd = 0; dd < D; ++dd) {
                dP += LoadAsFloat(&dy[Idx4(b, t, h, dd, Tlen, H, D)]) *
                      LoadAsFloat(&v[Idx4(b, ss, h, dd, Tlen, H, D)]);
            }
            sum_dP_mul_P += dP * p;
        }

        // compute dS at (t,s)
        float dot_ts = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dot_ts += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                      LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        float p_ts = expf(dot_ts * scale - max_score) * inv_sum;

        float dP_ts = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dP_ts += LoadAsFloat(&dy[Idx4(b, t, h, dd, Tlen, H, D)]) *
                     LoadAsFloat(&v[Idx4(b, s, h, dd, Tlen, H, D)]);
        }

        float dS_ts = p_ts * (dP_ts - sum_dP_mul_P);

        // dk accum: dS * scale * q
        dk_val += dS_ts * scale * LoadAsFloat(&q[Idx4(b, t, h, d, Tlen, H, D)]);
    }

    dk[Idx4(b, s, h, d, Tlen, H, D)] = StoreFromFloat<T>(dk_val);
}

// Compute dv: parallel over (b,s,h,d), sum over t: dv = sum_t P(t,s) * dy(t,d)
template <typename T>
__global__ void SDPABwdDVKernel(T* dv,
                               const T* dy,
                               const T* q,
                               const T* k,
                               int64_t B, int64_t Tlen, int64_t H, int64_t D,
                               float scale,
                               bool is_causal) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * Tlen * H * D;
    if (linear >= total) return;

    int64_t d = linear % D;
    int64_t tmp = linear / D;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t s = tmp % Tlen;
    int64_t b = tmp / Tlen;

    float dv_val = 0.0f;

    for (int64_t t = 0; t < Tlen; ++t) {
        if (is_causal && s > t) continue;

        // softmax stats for (b,t,h)
        float max_score = -INFINITY;
        for (int64_t ss = 0; ss < Tlen; ++ss) {
            if (is_causal && ss > t) continue;
            float dot = 0.0f;
            for (int64_t dd = 0; dd < D; ++dd) {
                dot += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                       LoadAsFloat(&k[Idx4(b, ss, h, dd, Tlen, H, D)]);
            }
            float score = dot * scale;
            if (score > max_score) max_score = score;
        }
        float sum_exp = 0.0f;
        for (int64_t ss = 0; ss < Tlen; ++ss) {
            if (is_causal && ss > t) continue;
            float dot = 0.0f;
            for (int64_t dd = 0; dd < D; ++dd) {
                dot += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                       LoadAsFloat(&k[Idx4(b, ss, h, dd, Tlen, H, D)]);
            }
            sum_exp += expf(dot * scale - max_score);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // P(t,s)
        float dot_ts = 0.0f;
        for (int64_t dd = 0; dd < D; ++dd) {
            dot_ts += LoadAsFloat(&q[Idx4(b, t, h, dd, Tlen, H, D)]) *
                      LoadAsFloat(&k[Idx4(b, s, h, dd, Tlen, H, D)]);
        }
        float p_ts = expf(dot_ts * scale - max_score) * inv_sum;

        dv_val += p_ts * LoadAsFloat(&dy[Idx4(b, t, h, d, Tlen, H, D)]);
    }

    dv[Idx4(b, s, h, d, Tlen, H, D)] = StoreFromFloat<T>(dv_val);
}

// --------------------------- Launchers ---------------------------
template <size_t BLOCK_SIZE, typename T>
void LaunchSDPAForward(const std::shared_ptr<Tensor>& y,
                       const std::shared_ptr<Tensor>& q,
                       const std::shared_ptr<Tensor>& k,
                       const std::shared_ptr<Tensor>& v,
                       float scale,
                       bool is_causal) {
    const auto& dims = q->Dims();
    const int64_t B = dims[0], Tlen = dims[1], H = dims[2], D = dims[3];

    T* y_ptr = static_cast<T*>(y->DataPtr());
    const T* q_ptr = static_cast<const T*>(q->DataPtr());
    const T* k_ptr = static_cast<const T*>(k->DataPtr());
    const T* v_ptr = static_cast<const T*>(v->DataPtr());

    int64_t total = B * Tlen * H * D;
    if (total == 0) return;

    if (BLOCK_SIZE > 1024) {
        LOG_LOC(FATAL, "CUDA SDPA forward: 'BLOCK_SIZE used is larger than max threads per block'");
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto device = y->GetDevice();
    const auto& cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream*>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    SDPAForwardKernel<T><<<grid, block, 0, cuda_stream>>>(y_ptr, q_ptr, k_ptr, v_ptr, B, Tlen, H, D, scale, is_causal);
}

template <size_t BLOCK_SIZE, typename T>
void LaunchSDPABackward(const std::shared_ptr<Tensor>& dq,
                        const std::shared_ptr<Tensor>& dk,
                        const std::shared_ptr<Tensor>& dv,
                        const std::shared_ptr<Tensor>& dy,
                        const std::shared_ptr<Tensor>& q,
                        const std::shared_ptr<Tensor>& k,
                        const std::shared_ptr<Tensor>& v,
                        float scale,
                        bool is_causal) {
    const auto& dims = q->Dims();
    const int64_t B = dims[0], Tlen = dims[1], H = dims[2], D = dims[3];
    int64_t total = B * Tlen * H * D;
    if (total == 0) return;

    if (BLOCK_SIZE > 1024) {
        LOG_LOC(FATAL, "CUDA SDPA backward: 'BLOCK_SIZE used is larger than max threads per block'");
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    T* dq_ptr = static_cast<T*>(dq->DataPtr());
    T* dk_ptr = static_cast<T*>(dk->DataPtr());
    T* dv_ptr = static_cast<T*>(dv->DataPtr());

    const T* dy_ptr = static_cast<const T*>(dy->DataPtr());
    const T* q_ptr  = static_cast<const T*>(q->DataPtr());
    const T* k_ptr  = static_cast<const T*>(k->DataPtr());
    const T* v_ptr  = static_cast<const T*>(v->DataPtr());

    auto device = dq->GetDevice();
    const auto& cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream*>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    // dq
    SDPABwdDQKernel<T><<<grid, block, 0, cuda_stream>>>(dq_ptr, dy_ptr, q_ptr, k_ptr, v_ptr, B, Tlen, H, D, scale,
                                                        is_causal);
    // dk
    SDPABwdDKKernel<T><<<grid, block, 0, cuda_stream>>>(dk_ptr, dy_ptr, q_ptr, k_ptr, v_ptr, B, Tlen, H, D, scale,
                                                        is_causal);
    // dv (doesn't need v)
    SDPABwdDVKernel<T><<<grid, block, 0, cuda_stream>>>(dv_ptr, dy_ptr, q_ptr, k_ptr, B, Tlen, H, D, scale, is_causal);
}

// --------------------------- Public kernel APIs (Dispatcher targets) ---------------------------
// Must match your .cc Dispatcher call signature.

std::shared_ptr<Tensor> ScaledDotProductAttentionForward(const std::shared_ptr<Tensor>& q,
                                                         const std::shared_ptr<Tensor>& k,
                                                         const std::shared_ptr<Tensor>& v,
                                                         const std::shared_ptr<Tensor>& /*attn_mask*/,
                                                         double /*dropout_p*/,
                                                         bool is_causal,
                                                         double scale,
                                                         bool /*enable_gqa*/) {
    CHECK(q);
    CHECK(k);
    CHECK(v);
    CHECK_EQ(q->Dims().size(), 4);
    CHECK_EQ(k->Dims().size(), 4);
    CHECK_EQ(v->Dims().size(), 4);
    //CHECK_EQ(q->Dims(), k->Dims());
    //CHECK_EQ(q->Dims(), v->Dims());

    auto dtype = q->Dtype();
    auto y = std::make_shared<Tensor>(q->Dims(), dtype, q->GetDevice());

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchSDPAForward<256, float>(y, q, k, v, (float)scale, is_causal);), DataType::kFLOAT32)
        DISPATCH_CASE(
            WRAP(LaunchSDPAForward<256, nv_bfloat16>(y, q, k, v, (float)scale, is_causal);), DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA SDPA forward: 'Unsupported data type'");
    }
    return y;
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttentionBackward(const std::shared_ptr<Tensor>& dy,
                                                                       const std::shared_ptr<Tensor>& q,
                                                                       const std::shared_ptr<Tensor>& k,
                                                                       const std::shared_ptr<Tensor>& v,
                                                                       const std::shared_ptr<Tensor>& /*attn_mask*/,
                                                                       double /*dropout_p*/,
                                                                       bool is_causal,
                                                                       double scale,
                                                                       bool /*enable_gqa*/) {
    CHECK(dy);
    CHECK(q);
    CHECK(k);
    CHECK(v);
    //CHECK_EQ(dy->Dims(), q->Dims());
    //CHECK_EQ(q->Dims(), k->Dims());
    //CHECK_EQ(q->Dims(), v->Dims());

    // dtype handling: promote like softmax backward does (widest type).
    auto dy_dtype = dy->Dtype();
    auto q_dtype = q->Dtype();
    auto k_dtype = k->Dtype();
    auto v_dtype = v->Dtype();

    DataType promoted_type = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>,
                                          DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {dy_dtype, q_dtype, k_dtype, v_dtype},
        [=]<typename Tdy, typename Tq, typename Tk, typename Tv>() {
            return DataTypeMap_v<WidestType_t<WidestType_t<Tdy, Tq>, WidestType_t<Tk, Tv>>>;
        },
        "CUDA ScaledDotProductAttentionBackward");

    auto dy_p = dy_dtype == promoted_type ? dy : std::make_shared<Tensor>(dy->To(promoted_type));
    auto q_p  = q_dtype == promoted_type ? q  : std::make_shared<Tensor>(q->To(promoted_type));
    auto k_p  = k_dtype == promoted_type ? k  : std::make_shared<Tensor>(k->To(promoted_type));
    auto v_p  = v_dtype == promoted_type ? v  : std::make_shared<Tensor>(v->To(promoted_type));

    auto dq = std::make_shared<Tensor>(q->Dims(), promoted_type, q->GetDevice());
    auto dk = std::make_shared<Tensor>(k->Dims(), promoted_type, k->GetDevice());
    auto dv = std::make_shared<Tensor>(v->Dims(), promoted_type, v->GetDevice());

    // Fill zeros
    DispatchFunc<INFINI_ALL_TYPES>(promoted_type, [=]<typename T>() { dq->Fill<T>(0); }, "CUDA SDPA Backward");
    DispatchFunc<INFINI_ALL_TYPES>(promoted_type, [=]<typename T>() { dk->Fill<T>(0); }, "CUDA SDPA Backward");
    DispatchFunc<INFINI_ALL_TYPES>(promoted_type, [=]<typename T>() { dv->Fill<T>(0); }, "CUDA SDPA Backward");

    switch (promoted_type) {
        DISPATCH_CASE(
            WRAP(LaunchSDPABackward<128, float>(dq, dk, dv, dy_p, q_p, k_p, v_p, (float)scale, is_causal);),
            DataType::kFLOAT32)
        DISPATCH_CASE(
            WRAP(LaunchSDPABackward<128, nv_bfloat16>(dq, dk, dv, dy_p, q_p, k_p, v_p, (float)scale, is_causal);),
            DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA SDPA backward: 'Unsupported data type'");
    }

    // .cc 当前 Forward 固定 3 inputs，但 Backward dispatcher signature含 mask；
    // 这里返回 {dq, dk, dv, nullptr}，和你 .cc 的注释一致（mask grad = nullptr）。
    return {dq, dk, dv, nullptr};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SDPA_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionForward)
REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionBackward)

#undef REGISTER_CUDA_SDPA_KERNEL
