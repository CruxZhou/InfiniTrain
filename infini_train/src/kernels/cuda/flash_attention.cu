// flash_attention.cu
//
// FlashAttention-2 style forward and backward for Scaled Dot-Product Attention.
//
// Algorithm reference: Dao, "FlashAttention-2: Faster Attention with Better
// Parallelism and Work Partitioning" (arXiv:2307.08691).
//
// Key algorithmic ideas ported from Dao-AILab/flash-attention repository
// (csrc/flash_attn/src/flash_fwd_kernel.h, flash_bwd_kernel.h):
//   - Tiled Q·K^T computation with online softmax (running max + running sum)
//   - Fused softmax·V accumulation with rescaling (no N×N attention matrix in HBM)
//   - Backward: recompute attention from Q,K,V + precomputed LSE / D_i
//   - Causal masking at tile granularity (whole-tile skip + per-element mask)
//
// Adapted for infini_train's Tensor / Dispatcher / DeviceGuard interface.
// No PyTorch / ATen / CUTLASS dependency.
//
// Supported dtypes : float32, nv_bfloat16
// Supported head dims: 32, 64, 96, 128 (compile-time template parameter)
// Causal  : yes (is_causal flag)
// Dropout : only p = 0
// Attn mask: additive float mask, broadcastable (2D or 4D)
// GQA     : yes (H_q must be divisible by H_kv)

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>  // IMPORTANT: cudaMalloc/cudaFreeAsync/cudaGetErrorString/CUDART_INF_F

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
//#include "infini_train/include/common/cuda/cub_compat.cuh"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

// ============================================================================
// Indexing and cast helpers
// ============================================================================

// q,k,v,y : (B, T, H, D) contiguous row-major
// index = (((b*T + t)*H + h)*D + d)
__device__ __forceinline__ int64_t Idx4(int64_t b, int64_t t, int64_t h, int64_t d,
                                        int64_t T, int64_t H, int64_t D) {
    return (((b * T + t) * H + h) * D + d);
}

template <typename T>
__device__ __forceinline__ float LoadF(const T* ptr) {
    return common::cuda::Cast<float>(*ptr);
}

template <typename T>
__device__ __forceinline__ T StoreF(float x) {
    return common::cuda::Cast<T>(x);
}

// Load additive mask value; returns 0 if mask is nullptr.
__device__ __forceinline__ float LoadMask(const float* __restrict__ mask,
                                          int64_t b, int64_t h,
                                          int64_t q, int64_t kv,
                                          int64_t stride_b, int64_t stride_h,
                                          int64_t stride_m, int64_t stride_n) {
    if (mask == nullptr) return 0.0f;
    return mask[b * stride_b + h * stride_h + q * stride_m + kv * stride_n];
}

// ============================================================================
// FORWARD KERNEL — FlashAttention-2 style (simple, readable version)
// ============================================================================

template <typename T, int HD, int BLOCK_M, int BLOCK_N>
__global__ void __launch_bounds__(BLOCK_M, 1)
FlashFwdKernel(T* __restrict__       O_out,
               float* __restrict__   lse_out,   // (B, H_q, Tlen) or nullptr
               const T* __restrict__ Q,
               const T* __restrict__ K,
               const T* __restrict__ V,
               const float* __restrict__ mask,  // additive mask or nullptr
               const int64_t         Tlen,
               const int64_t         H_q,
               const int64_t         H_kv,
               const float           scale,
               const bool            is_causal,
               const int64_t         mask_stride_b,
               const int64_t         mask_stride_h,
               const int64_t         mask_stride_m,
               const int64_t         mask_stride_n) {
    const int tid         = threadIdx.x;                    // 0 .. BLOCK_M-1
    const int bh          = blockIdx.y;                     // batch * H_q + head
    const int64_t b       = bh / H_q;
    const int64_t h       = bh % H_q;                      // Q head index
    const int64_t kv_h    = h * H_kv / H_q;                // corresponding KV head
    const int64_t q_start = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t q_pos   = q_start + tid;
    const bool active     = (q_pos < Tlen);

    // --- Shared memory ---
    extern __shared__ float smem[];
    float* Q_smem = smem;                                   // BLOCK_M * HD
    float* K_tile = smem + BLOCK_M * HD;                    // BLOCK_N * HD
    float* V_tile = smem + (BLOCK_M + BLOCK_N) * HD;        // BLOCK_N * HD

    // --- Load this thread's Q row into shared memory ---
    if (active) {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            Q_smem[tid * HD + d] = LoadF(&Q[Idx4(b, q_pos, h, d, Tlen, H_q, HD)]);
    } else {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            Q_smem[tid * HD + d] = 0.0f;
    }
    __syncthreads();

    // --- Output accumulator & online softmax state ---
    float o_acc[HD];
    #pragma unroll
    for (int d = 0; d < HD; d++) o_acc[d] = 0.0f;

    float m_i = -INFINITY;   // running row-max of scores
    float l_i = 0.0f;            // running sum of exp(score - m_i)

    const int num_kv_blocks = (static_cast<int>(Tlen) + BLOCK_N - 1) / BLOCK_N;

    for (int j = 0; j < num_kv_blocks; j++) {
        const int64_t kv_start = static_cast<int64_t>(j) * BLOCK_N;

        // Causal: whole KV tile beyond all Q positions in this block -> stop
        if (is_causal && kv_start > q_start + BLOCK_M - 1) break;

        // --- Cooperative load of K_tile and V_tile (using kv_h for GQA) ---
        for (int r = tid; r < BLOCK_N; r += BLOCK_M) {
            const int64_t kv_pos = kv_start + r;
            if (kv_pos < Tlen) {
                #pragma unroll
                for (int d = 0; d < HD; d++) {
                    K_tile[r * HD + d] = LoadF(&K[Idx4(b, kv_pos, kv_h, d, Tlen, H_kv, HD)]);
                    V_tile[r * HD + d] = LoadF(&V[Idx4(b, kv_pos, kv_h, d, Tlen, H_kv, HD)]);
                }
            } else {
                #pragma unroll
                for (int d = 0; d < HD; d++) {
                    K_tile[r * HD + d] = 0.0f;
                    V_tile[r * HD + d] = 0.0f;
                }
            }
        }
        __syncthreads();  // barrier after tile load

        if (active) {
            // --- Sub-pass 1: tile-local row-max m_ij ---
            float m_ij = -INFINITY;
            for (int n = 0; n < BLOCK_N; n++) {
                const int64_t kv_pos = kv_start + n;
                if (kv_pos >= Tlen) continue;
                if (is_causal && kv_pos > q_pos) continue;

                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dot += Q_smem[tid * HD + d] * K_tile[n * HD + d];
                float score = dot * scale
                    + LoadMask(mask, b, h, q_pos, kv_pos,
                               mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
                m_ij = fmaxf(m_ij, score);
            }

            // --- Online softmax merge ---
            const float m_new = fmaxf(m_i, m_ij);
            const float alpha = expf(m_i - m_new);

            #pragma unroll
            for (int d = 0; d < HD; d++)
                o_acc[d] *= alpha;
            l_i *= alpha;

            // --- Sub-pass 2: accumulate exp(score - m_new) * V ---
            for (int n = 0; n < BLOCK_N; n++) {
                const int64_t kv_pos = kv_start + n;
                if (kv_pos >= Tlen) continue;
                if (is_causal && kv_pos > q_pos) continue;

                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dot += Q_smem[tid * HD + d] * K_tile[n * HD + d];

                float score = dot * scale
                    + LoadMask(mask, b, h, q_pos, kv_pos,
                               mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
                const float p = expf(score - m_new);
                l_i += p;

                #pragma unroll
                for (int d = 0; d < HD; d++)
                    o_acc[d] += p * V_tile[n * HD + d];
            }

            m_i = m_new;
        }

        __syncthreads();  // barrier end-of-tile (ALL threads must reach)
    }

    // --- Final normalisation & write-back ---
    if (active) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int d = 0; d < HD; d++)
            O_out[Idx4(b, q_pos, h, d, Tlen, H_q, HD)] = StoreF<T>(o_acc[d] * inv_l);

        if (lse_out) {
            lse_out[b * H_q * Tlen + h * Tlen + q_pos] = m_i + logf(l_i + 1e-20f);
        }
    }
}

// ============================================================================
// BACKWARD — Preprocess: recompute forward -> produce LSE and D_i
// ============================================================================

template <typename T, int HD, int BLOCK_M, int BLOCK_N>
__global__ void __launch_bounds__(BLOCK_M, 1)
FlashBwdPreprocessKernel(float* __restrict__   lse_out,  // (B, H_q, T)
                         float* __restrict__   D_out,    // (B, H_q, T)
                         const T* __restrict__ Q,
                         const T* __restrict__ K,
                         const T* __restrict__ V,
                         const T* __restrict__ dO,
                         const float* __restrict__ mask,
                         const int64_t         Tlen,
                         const int64_t         H_q,
                         const int64_t         H_kv,
                         const float           scale,
                         const bool            is_causal,
                         const int64_t         mask_stride_b,
                         const int64_t         mask_stride_h,
                         const int64_t         mask_stride_m,
                         const int64_t         mask_stride_n) {
    const int tid         = threadIdx.x;
    const int bh          = blockIdx.y;
    const int64_t b       = bh / H_q;
    const int64_t h       = bh % H_q;
    const int64_t kv_h    = h * H_kv / H_q;
    const int64_t q_start = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t q_pos   = q_start + tid;
    const bool active     = (q_pos < Tlen);

    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* K_tile = smem + BLOCK_M * HD;
    float* V_tile = smem + (BLOCK_M + BLOCK_N) * HD;

    // Load Q row into smem
    if (active) {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            Q_smem[tid * HD + d] = LoadF(&Q[Idx4(b, q_pos, h, d, Tlen, H_q, HD)]);
    } else {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            Q_smem[tid * HD + d] = 0.0f;
    }

    // Load dO row into registers
    float dO_reg[HD];
    if (active) {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            dO_reg[d] = LoadF(&dO[Idx4(b, q_pos, h, d, Tlen, H_q, HD)]);
    } else {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            dO_reg[d] = 0.0f;
    }
    __syncthreads();

    // Recompute forward
    float o_acc[HD];
    #pragma unroll
    for (int d = 0; d < HD; d++) o_acc[d] = 0.0f;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int num_kv_blocks = (static_cast<int>(Tlen) + BLOCK_N - 1) / BLOCK_N;

    for (int j = 0; j < num_kv_blocks; j++) {
        const int64_t kv_start = static_cast<int64_t>(j) * BLOCK_N;

        if (is_causal && kv_start > q_start + BLOCK_M - 1) break;

        for (int r = tid; r < BLOCK_N; r += BLOCK_M) {
            const int64_t kv_pos = kv_start + r;
            if (kv_pos < Tlen) {
                #pragma unroll
                for (int d = 0; d < HD; d++) {
                    K_tile[r * HD + d] = LoadF(&K[Idx4(b, kv_pos, kv_h, d, Tlen, H_kv, HD)]);
                    V_tile[r * HD + d] = LoadF(&V[Idx4(b, kv_pos, kv_h, d, Tlen, H_kv, HD)]);
                }
            } else {
                #pragma unroll
                for (int d = 0; d < HD; d++) {
                    K_tile[r * HD + d] = 0.0f;
                    V_tile[r * HD + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (active) {
            float m_ij = -INFINITY;
            for (int n = 0; n < BLOCK_N; n++) {
                const int64_t kv_pos = kv_start + n;
                if (kv_pos >= Tlen) continue;
                if (is_causal && kv_pos > q_pos) continue;

                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dot += Q_smem[tid * HD + d] * K_tile[n * HD + d];
                float score = dot * scale
                    + LoadMask(mask, b, h, q_pos, kv_pos,
                               mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
                m_ij = fmaxf(m_ij, score);
            }

            const float m_new = fmaxf(m_i, m_ij);
            const float alpha_val = expf(m_i - m_new);

            #pragma unroll
            for (int d = 0; d < HD; d++) o_acc[d] *= alpha_val;
            l_i *= alpha_val;

            for (int n = 0; n < BLOCK_N; n++) {
                const int64_t kv_pos = kv_start + n;
                if (kv_pos >= Tlen) continue;
                if (is_causal && kv_pos > q_pos) continue;

                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dot += Q_smem[tid * HD + d] * K_tile[n * HD + d];

                float score = dot * scale
                    + LoadMask(mask, b, h, q_pos, kv_pos,
                               mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
                const float p = expf(score - m_new);
                l_i += p;

                #pragma unroll
                for (int d = 0; d < HD; d++)
                    o_acc[d] += p * V_tile[n * HD + d];
            }

            m_i = m_new;
        }

        __syncthreads();
    }

    if (active) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

        float D_i = 0.0f;
        #pragma unroll
        for (int d = 0; d < HD; d++)
            D_i += dO_reg[d] * (o_acc[d] * inv_l);

        const int64_t idx = b * H_q * Tlen + h * Tlen + q_pos;
        lse_out[idx] = m_i + logf(l_i + 1e-20f);
        D_out[idx]   = D_i;
    }
}

// ============================================================================
// BACKWARD — dQ kernel
// ============================================================================

template <typename T, int HD, int BLOCK_M, int BLOCK_N>
__global__ void __launch_bounds__(BLOCK_M, 1)
FlashBwdDQKernel(T* __restrict__       dQ_out,
                 const T* __restrict__ Q,
                 const T* __restrict__ K,
                 const T* __restrict__ V,
                 const T* __restrict__ dO,
                 const float* __restrict__ lse,
                 const float* __restrict__ D_arr,
                 const float* __restrict__ mask,
                 const int64_t         Tlen,
                 const int64_t         H_q,
                 const int64_t         H_kv,
                 const float           scale,
                 const bool            is_causal,
                 const int64_t         mask_stride_b,
                 const int64_t         mask_stride_h,
                 const int64_t         mask_stride_m,
                 const int64_t         mask_stride_n) {
    const int tid         = threadIdx.x;
    const int bh          = blockIdx.y;
    const int64_t b       = bh / H_q;
    const int64_t h       = bh % H_q;
    const int64_t kv_h    = h * H_kv / H_q;
    const int64_t q_start = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t q_pos   = q_start + tid;
    const bool active     = (q_pos < Tlen);

    extern __shared__ float smem[];
    float* Q_smem = smem;                                   // BLOCK_M * HD
    float* K_tile = smem + BLOCK_M * HD;                    // BLOCK_N * HD
    float* V_tile = smem + (BLOCK_M + BLOCK_N) * HD;        // BLOCK_N * HD

    // Load Q and dO rows
    float dO_reg[HD];
    if (active) {
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            Q_smem[tid * HD + d] = LoadF(&Q[Idx4(b, q_pos, h, d, Tlen, H_q, HD)]);
            dO_reg[d] = LoadF(&dO[Idx4(b, q_pos, h, d, Tlen, H_q, HD)]);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            Q_smem[tid * HD + d] = 0.0f;
            dO_reg[d] = 0.0f;
        }
    }
    __syncthreads();

    float my_lse = 0.0f, my_D = 0.0f;
    if (active) {
        const int64_t idx = b * H_q * Tlen + h * Tlen + q_pos;
        my_lse = lse[idx];
        my_D   = D_arr[idx];
    }

    float dq_acc[HD];
    #pragma unroll
    for (int d = 0; d < HD; d++) dq_acc[d] = 0.0f;

    const int num_kv_blocks = (static_cast<int>(Tlen) + BLOCK_N - 1) / BLOCK_N;

    for (int j = 0; j < num_kv_blocks; j++) {
        const int64_t kv_start = static_cast<int64_t>(j) * BLOCK_N;
        if (is_causal && kv_start > q_start + BLOCK_M - 1) break;

        // Cooperative load of K_tile, V_tile (using kv_h)
        for (int r = tid; r < BLOCK_N; r += BLOCK_M) {
            const int64_t kv_pos = kv_start + r;
            if (kv_pos < Tlen) {
                #pragma unroll
                for (int d = 0; d < HD; d++) {
                    K_tile[r * HD + d] = LoadF(&K[Idx4(b, kv_pos, kv_h, d, Tlen, H_kv, HD)]);
                    V_tile[r * HD + d] = LoadF(&V[Idx4(b, kv_pos, kv_h, d, Tlen, H_kv, HD)]);
                }
            } else {
                #pragma unroll
                for (int d = 0; d < HD; d++) {
                    K_tile[r * HD + d] = 0.0f;
                    V_tile[r * HD + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (active) {
            for (int n = 0; n < BLOCK_N; n++) {
                const int64_t kv_pos = kv_start + n;
                if (kv_pos >= Tlen) continue;
                if (is_causal && kv_pos > q_pos) continue;

                float dot_qk = 0.0f;
                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dot_qk += Q_smem[tid * HD + d] * K_tile[n * HD + d];
                float score = dot_qk * scale
                    + LoadMask(mask, b, h, q_pos, kv_pos,
                               mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
                const float p_ij = expf(score - my_lse);

                float dp_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dp_ij += dO_reg[d] * V_tile[n * HD + d];

                const float ds_ij = p_ij * (dp_ij - my_D);
                const float s_ds = scale * ds_ij;

                #pragma unroll
                for (int d = 0; d < HD; d++)
                    dq_acc[d] += s_ds * K_tile[n * HD + d];
            }
        }

        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int d = 0; d < HD; d++)
            dQ_out[Idx4(b, q_pos, h, d, Tlen, H_q, HD)] = StoreF<T>(dq_acc[d]);
    }
}

// ============================================================================
// BACKWARD — dK / dV kernel
// ============================================================================

template <typename T, int HD, int BLOCK_M, int BLOCK_N>
__global__ void __launch_bounds__(BLOCK_N, 1)
FlashBwdDKVKernel(T* __restrict__       dK_out,
                  T* __restrict__       dV_out,
                  const T* __restrict__ Q,
                  const T* __restrict__ K,
                  const T* __restrict__ V,
                  const T* __restrict__ dO,
                  const float* __restrict__ lse,
                  const float* __restrict__ D_arr,
                  const float* __restrict__ mask,
                  const int64_t         Tlen,
                  const int64_t         H_q,
                  const int64_t         H_kv,
                  const float           scale,
                  const bool            is_causal,
                  const int64_t         mask_stride_b,
                  const int64_t         mask_stride_h,
                  const int64_t         mask_stride_m,
                  const int64_t         mask_stride_n) {
    const int tid          = threadIdx.x;               // 0 .. BLOCK_N-1
    const int bh           = blockIdx.y;                // b * H_kv + kv_h
    const int64_t b        = bh / H_kv;
    const int64_t kv_h     = bh % H_kv;
    const int64_t kv_start = static_cast<int64_t>(blockIdx.x) * BLOCK_N;
    const int64_t s_pos    = kv_start + tid;

    extern __shared__ float smem[];
    float* Q_tile   = smem;                                        // BLOCK_M * HD
    float* dO_tile  = smem + BLOCK_M * HD;                         // BLOCK_M * HD
    float* lse_tile = smem + 2 * BLOCK_M * HD;                     // BLOCK_M
    float* D_tile   = smem + 2 * BLOCK_M * HD + BLOCK_M;           // BLOCK_M

    const int64_t kv_base = (s_pos < Tlen)
                          ? Idx4(b, s_pos, kv_h, 0, Tlen, H_kv, HD)
                          : 0;

    float dk_acc[HD], dv_acc[HD];
    #pragma unroll
    for (int d = 0; d < HD; d++) { dk_acc[d] = 0.0f; dv_acc[d] = 0.0f; }

    const int num_q_blocks = (static_cast<int>(Tlen) + BLOCK_M - 1) / BLOCK_M;
    const int64_t group_size = H_q / H_kv;

    // Iterate over all Q heads that share this KV head
    for (int64_t g = 0; g < group_size; g++) {
        const int64_t q_h = kv_h * group_size + g;

        for (int i = 0; i < num_q_blocks; i++) {
            const int64_t q_start_tile = static_cast<int64_t>(i) * BLOCK_M;

            // Causal: if all q in tile < s_pos, skip
            if (is_causal && q_start_tile + BLOCK_M - 1 < s_pos) continue;

            // Cooperative load of Q_tile, dO_tile, stats for head q_h
            for (int r = tid; r < BLOCK_M; r += BLOCK_N) {
                const int64_t q_pos = q_start_tile + r;
                if (q_pos < Tlen) {
                    #pragma unroll
                    for (int d = 0; d < HD; d++) {
                        Q_tile[r * HD + d]  = LoadF(&Q[Idx4(b, q_pos, q_h, d, Tlen, H_q, HD)]);
                        dO_tile[r * HD + d] = LoadF(&dO[Idx4(b, q_pos, q_h, d, Tlen, H_q, HD)]);
                    }
                    const int64_t stat_idx = b * H_q * Tlen + q_h * Tlen + q_pos;
                    lse_tile[r] = lse[stat_idx];
                    D_tile[r]   = D_arr[stat_idx];
                } else {
                    #pragma unroll
                    for (int d = 0; d < HD; d++) {
                        Q_tile[r * HD + d]  = 0.0f;
                        dO_tile[r * HD + d] = 0.0f;
                    }
                    lse_tile[r] = 0.0f;
                    D_tile[r]   = 0.0f;
                }
            }
            __syncthreads();

            if (s_pos < Tlen) {
                for (int m = 0; m < BLOCK_M; m++) {
                    const int64_t q_pos = q_start_tile + m;
                    if (q_pos >= Tlen) continue;
                    if (is_causal && s_pos > q_pos) continue;

                    float dot_qk = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HD; d++)
                        dot_qk += Q_tile[m * HD + d] * LoadF(&K[kv_base + d]);
                    float score = dot_qk * scale
                        + LoadMask(mask, b, q_h, q_pos, s_pos,
                                   mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
                    const float p_val = expf(score - lse_tile[m]);

                    #pragma unroll
                    for (int d = 0; d < HD; d++)
                        dv_acc[d] += p_val * dO_tile[m * HD + d];

                    float dp_val = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HD; d++)
                        dp_val += dO_tile[m * HD + d] * LoadF(&V[kv_base + d]);

                    const float ds_val = p_val * (dp_val - D_tile[m]);
                    const float s_ds = scale * ds_val;

                    #pragma unroll
                    for (int d = 0; d < HD; d++)
                        dk_acc[d] += s_ds * Q_tile[m * HD + d];
                }
            }

            __syncthreads();
        }
    }

    if (s_pos < Tlen) {
        #pragma unroll
        for (int d = 0; d < HD; d++) {
            dK_out[Idx4(b, s_pos, kv_h, d, Tlen, H_kv, HD)] = StoreF<T>(dk_acc[d]);
            dV_out[Idx4(b, s_pos, kv_h, d, Tlen, H_kv, HD)] = StoreF<T>(dv_acc[d]);
        }
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

static cudaStream_t GetCudaStream(const std::shared_ptr<Tensor>& tensor) {
    auto device = tensor->GetDevice();
    return dynamic_cast<infini_train::core::cuda::CudaStream*>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

static constexpr int BM = 64;
static constexpr int BN = 32;

// Compute mask strides with broadcasting support.
// mask_dims should have 4 elements: (B_m, H_m, L, S).
// If a dimension is 1, its stride is set to 0 (broadcast).
static void ComputeMaskStrides(const std::vector<int64_t>& mask_dims,
                               int64_t& stride_b, int64_t& stride_h,
                               int64_t& stride_m, int64_t& stride_n) {
    // mask_dims: (B_m, H_m, L, S) — contiguous row-major
    const int64_t B_m = mask_dims[0];
    const int64_t H_m = mask_dims[1];
    const int64_t L   = mask_dims[2];
    const int64_t S   = mask_dims[3];

    // Contiguous strides
    stride_n = 1;
    stride_m = S;
    stride_h = (H_m > 1) ? L * S : 0;
    stride_b = (B_m > 1) ? H_m * L * S : 0;
}


template <typename T, int HD>
void LaunchFlashForward(const std::shared_ptr<Tensor>& y,
                        const std::shared_ptr<Tensor>& q,
                        const std::shared_ptr<Tensor>& k,
                        const std::shared_ptr<Tensor>& v,
                        const float* mask_ptr,
                        int64_t mask_stride_b, int64_t mask_stride_h,
                        int64_t mask_stride_m, int64_t mask_stride_n,
                        float scale, bool is_causal) {
    const auto& q_dims = q->Dims();
    const auto& k_dims = k->Dims();
    const int64_t B = q_dims[0], Tlen = q_dims[1], H_q = q_dims[2];
    const int64_t H_kv = k_dims[2];
    if (B * Tlen * H_q == 0) return;

    dim3 grid(static_cast<unsigned>((Tlen + BM - 1) / BM),
              static_cast<unsigned>(B * H_q));
    dim3 block(BM);

    const size_t smem_bytes = static_cast<size_t>(BM + 2 * BN) * HD * sizeof(float);
    auto stream = GetCudaStream(y);

    FlashFwdKernel<T, HD, BM, BN><<<grid, block, smem_bytes, stream>>>(
        static_cast<T*>(y->DataPtr()),
        nullptr,   // lse_out — recompute in backward
        static_cast<const T*>(q->DataPtr()),
        static_cast<const T*>(k->DataPtr()),
        static_cast<const T*>(v->DataPtr()),
        mask_ptr,
        Tlen, H_q, H_kv, scale, is_causal,
        mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
}

template <typename T, int HD>
void LaunchFlashBackward(const std::shared_ptr<Tensor>& dq,
                         const std::shared_ptr<Tensor>& dk,
                         const std::shared_ptr<Tensor>& dv,
                         const std::shared_ptr<Tensor>& dy,
                         const std::shared_ptr<Tensor>& q,
                         const std::shared_ptr<Tensor>& k,
                         const std::shared_ptr<Tensor>& v,
                         const float* mask_ptr,
                         int64_t mask_stride_b, int64_t mask_stride_h,
                         int64_t mask_stride_m, int64_t mask_stride_n,
                         float scale, bool is_causal) {
    const auto& q_dims = q->Dims();
    const auto& k_dims = k->Dims();
    const int64_t B = q_dims[0], Tlen = q_dims[1], H_q = q_dims[2];
    const int64_t H_kv = k_dims[2];
    if (B * Tlen * H_q == 0) return;

    auto stream = GetCudaStream(dq);

    // Scratch for LSE and D_i (indexed by Q heads)
    const int64_t stat_size = B * H_q * Tlen;
    float* lse_buf = nullptr;
    float* D_buf   = nullptr;

    cudaError_t err1 = cudaMalloc(reinterpret_cast<void**>(&lse_buf),
                                  static_cast<size_t>(stat_size) * sizeof(float));
    cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&D_buf),
                                  static_cast<size_t>(stat_size) * sizeof(float));
    CHECK(err1 == cudaSuccess)
        << "Failed to allocate LSE buffer (" << (stat_size * 4) << " bytes): "
        << cudaGetErrorString(err1);
    CHECK(err2 == cudaSuccess)
        << "Failed to allocate D buffer (" << (stat_size * 4) << " bytes): "
        << cudaGetErrorString(err2);

    // Step 1: preprocess (grid over Q heads)
    {
        dim3 grid_pre(static_cast<unsigned>((Tlen + BM - 1) / BM),
                      static_cast<unsigned>(B * H_q));
        dim3 block_pre(BM);
        const size_t smem_pre = static_cast<size_t>(BM + 2 * BN) * HD * sizeof(float);

        FlashBwdPreprocessKernel<T, HD, BM, BN><<<grid_pre, block_pre, smem_pre, stream>>>(
            lse_buf, D_buf,
            static_cast<const T*>(q->DataPtr()),
            static_cast<const T*>(k->DataPtr()),
            static_cast<const T*>(v->DataPtr()),
            static_cast<const T*>(dy->DataPtr()),
            mask_ptr,
            Tlen, H_q, H_kv, scale, is_causal,
            mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
    }

    // Step 2: dQ (grid over Q heads)
    {
        dim3 grid_dq(static_cast<unsigned>((Tlen + BM - 1) / BM),
                     static_cast<unsigned>(B * H_q));
        dim3 block_dq(BM);
        const size_t smem_dq = static_cast<size_t>(BM + 2 * BN) * HD * sizeof(float);

        FlashBwdDQKernel<T, HD, BM, BN><<<grid_dq, block_dq, smem_dq, stream>>>(
            static_cast<T*>(dq->DataPtr()),
            static_cast<const T*>(q->DataPtr()),
            static_cast<const T*>(k->DataPtr()),
            static_cast<const T*>(v->DataPtr()),
            static_cast<const T*>(dy->DataPtr()),
            lse_buf, D_buf,
            mask_ptr,
            Tlen, H_q, H_kv, scale, is_causal,
            mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
    }

    // Step 3: dK, dV (grid over KV heads)
    {
        dim3 grid_dkv(static_cast<unsigned>((Tlen + BN - 1) / BN),
                      static_cast<unsigned>(B * H_kv));
        dim3 block_dkv(BN);
        const size_t smem_dkv = static_cast<size_t>(2 * BM * HD + 2 * BM) * sizeof(float);

        FlashBwdDKVKernel<T, HD, BM, BN><<<grid_dkv, block_dkv, smem_dkv, stream>>>(
            static_cast<T*>(dk->DataPtr()),
            static_cast<T*>(dv->DataPtr()),
            static_cast<const T*>(q->DataPtr()),
            static_cast<const T*>(k->DataPtr()),
            static_cast<const T*>(v->DataPtr()),
            static_cast<const T*>(dy->DataPtr()),
            lse_buf, D_buf,
            mask_ptr,
            Tlen, H_q, H_kv, scale, is_causal,
            mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n);
    }

    // Free scratch
#if CUDART_VERSION >= 11020
    cudaFreeAsync(lse_buf, stream);
    cudaFreeAsync(D_buf,   stream);
#else
    cudaStreamSynchronize(stream);
    cudaFree(lse_buf);
    cudaFree(D_buf);
#endif
}

// ============================================================================
// Head-dim dispatch
// ============================================================================

template <typename T>
void DispatchForwardByHD(int HD,
                         const std::shared_ptr<Tensor>& y,
                         const std::shared_ptr<Tensor>& q,
                         const std::shared_ptr<Tensor>& k,
                         const std::shared_ptr<Tensor>& v,
                         const float* mask_ptr,
                         int64_t mask_stride_b, int64_t mask_stride_h,
                         int64_t mask_stride_m, int64_t mask_stride_n,
                         float scale, bool is_causal) {
    switch (HD) {
        case 32:  LaunchFlashForward<T, 32> (y, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        case 64:  LaunchFlashForward<T, 64> (y, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        case 96:  LaunchFlashForward<T, 96> (y, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        case 128: LaunchFlashForward<T, 128>(y, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        default:
            LOG_LOC(FATAL, "FlashAttention forward: unsupported head dim "
                    << HD << ". Supported: 32, 64, 96, 128.");
    }
}

template <typename T>
void DispatchBackwardByHD(int HD,
                          const std::shared_ptr<Tensor>& dq,
                          const std::shared_ptr<Tensor>& dk,
                          const std::shared_ptr<Tensor>& dv,
                          const std::shared_ptr<Tensor>& dy,
                          const std::shared_ptr<Tensor>& q,
                          const std::shared_ptr<Tensor>& k,
                          const std::shared_ptr<Tensor>& v,
                          const float* mask_ptr,
                          int64_t mask_stride_b, int64_t mask_stride_h,
                          int64_t mask_stride_m, int64_t mask_stride_n,
                          float scale, bool is_causal) {
    switch (HD) {
        case 32:
            LaunchFlashBackward<T, 32> (dq, dk, dv, dy, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        case 64:
            LaunchFlashBackward<T, 64> (dq, dk, dv, dy, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        case 96:
            LaunchFlashBackward<T, 96> (dq, dk, dv, dy, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        case 128:
            LaunchFlashBackward<T, 128>(dq, dk, dv, dy, q, k, v, mask_ptr, mask_stride_b, mask_stride_h, mask_stride_m, mask_stride_n, scale, is_causal); break;
        default:
            LOG_LOC(FATAL, "FlashAttention backward: unsupported head dim "
                    << HD << ". Supported: 32, 64, 96, 128.");
    }
}

// ============================================================================
// Mask preparation helper
// ============================================================================

// Prepare mask: convert to float32 if needed, normalize shape to 4D,
// compute broadcast strides.  Returns (float pointer on device, strides).
// If attn_mask is nullptr, returns (nullptr, 0,0,0,0) and leaves mask_f32 untouched.
static void PrepareMask(const std::shared_ptr<Tensor>& attn_mask,
                        std::shared_ptr<Tensor>& mask_f32,
                        const float*& mask_ptr,
                        int64_t& stride_b, int64_t& stride_h,
                        int64_t& stride_m, int64_t& stride_n) {
    mask_ptr = nullptr;
    stride_b = stride_h = stride_m = stride_n = 0;

    if (!attn_mask) return;

    // Convert to float32 if necessary
    if (attn_mask->Dtype() != DataType::kFLOAT32) {
        mask_f32 = std::make_shared<Tensor>(attn_mask->To(DataType::kFLOAT32));
    } else {
        mask_f32 = attn_mask;
    }

    const auto& dims = mask_f32->Dims();
    std::vector<int64_t> dims4;

    if (dims.size() == 2) {
        // (L, S) -> (1, 1, L, S)
        dims4 = {1, 1, dims[0], dims[1]};
    } else if (dims.size() == 3) {
        // (X, L, S) -> (X, 1, L, S)  — X could be B
        dims4 = {dims[0], 1, dims[1], dims[2]};
    } else if (dims.size() == 4) {
        dims4 = {dims[0], dims[1], dims[2], dims[3]};
    } else {
        LOG_LOC(FATAL, "FlashAttention: attn_mask must be 2D, 3D, or 4D, got "
                << dims.size() << "D.");
    }

    ComputeMaskStrides(dims4, stride_b, stride_h, stride_m, stride_n);
    mask_ptr = static_cast<const float*>(mask_f32->DataPtr());
}

// ============================================================================
// Public Dispatcher API — signatures unchanged
// ============================================================================

std::shared_ptr<Tensor> ScaledDotProductAttentionForward(
    const std::shared_ptr<Tensor>& q,
    const std::shared_ptr<Tensor>& k,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& attn_mask,
    double /*dropout_p*/,
    bool is_causal,
    double scale,
    bool enable_gqa) {
    CHECK(q);
    CHECK(k);
    CHECK(v);
    CHECK_EQ(q->Dims().size(), 4);
    CHECK_EQ(k->Dims().size(), 4);
    CHECK_EQ(v->Dims().size(), 4);

    const int64_t H_q  = q->Dims()[2];
    const int64_t H_kv = k->Dims()[2];
    if (!enable_gqa) {
        CHECK_EQ(H_q, H_kv)
            << "Q heads (" << H_q << ") != KV heads (" << H_kv
            << "). Set enable_gqa=true for grouped-query attention.";
    }
    CHECK(H_q % H_kv == 0)
        << "Q heads (" << H_q << ") must be divisible by KV heads (" << H_kv << ").";

    // Prepare mask
    std::shared_ptr<Tensor> mask_f32;
    const float* mask_ptr = nullptr;
    int64_t ms_b = 0, ms_h = 0, ms_m = 0, ms_n = 0;
    PrepareMask(attn_mask, mask_f32, mask_ptr, ms_b, ms_h, ms_m, ms_n);

    auto dtype = q->Dtype();
    auto y = std::make_shared<Tensor>(q->Dims(), dtype, q->GetDevice());
    const int HD = static_cast<int>(q->Dims()[3]);

    switch (dtype) {
        DISPATCH_CASE(
            WRAP(DispatchForwardByHD<float>(HD, y, q, k, v,
                                            mask_ptr, ms_b, ms_h, ms_m, ms_n,
                                            static_cast<float>(scale), is_causal);),
            DataType::kFLOAT32)
        DISPATCH_CASE(
            WRAP(DispatchForwardByHD<nv_bfloat16>(HD, y, q, k, v,
                                                  mask_ptr, ms_b, ms_h, ms_m, ms_n,
                                                  static_cast<float>(scale), is_causal);),
            DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "FlashAttention forward: unsupported dtype");
    }
    return y;
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttentionBackward(
    const std::shared_ptr<Tensor>& dy,
    const std::shared_ptr<Tensor>& q,
    const std::shared_ptr<Tensor>& k,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& attn_mask,
    double /*dropout_p*/,
    bool is_causal,
    double scale,
    bool enable_gqa) {
    CHECK(dy);
    CHECK(q);
    CHECK(k);
    CHECK(v);

    const int64_t H_q  = q->Dims()[2];
    const int64_t H_kv = k->Dims()[2];
    if (!enable_gqa) {
        CHECK_EQ(H_q, H_kv)
            << "Q heads (" << H_q << ") != KV heads (" << H_kv
            << "). Set enable_gqa=true for grouped-query attention.";
    }
    CHECK(H_q % H_kv == 0)
        << "Q heads (" << H_q << ") must be divisible by KV heads (" << H_kv << ").";

    // Type promotion (same logic as original)
    auto dy_dtype = dy->Dtype();
    auto q_dtype  = q->Dtype();
    auto k_dtype  = k->Dtype();
    auto v_dtype  = v->Dtype();

    DataType promoted_type = DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>,
                                          DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {dy_dtype, q_dtype, k_dtype, v_dtype},
        [=]<typename Tdy, typename Tq, typename Tk, typename Tv>() {
            return DataTypeMap_v<WidestType_t<WidestType_t<Tdy, Tq>, WidestType_t<Tk, Tv>>>;
        },
        "CUDA ScaledDotProductAttentionBackward");

    auto dy_p = dy_dtype == promoted_type ? dy : std::make_shared<Tensor>(dy->To(promoted_type));
    auto q_p  = q_dtype  == promoted_type ? q  : std::make_shared<Tensor>(q->To(promoted_type));
    auto k_p  = k_dtype  == promoted_type ? k  : std::make_shared<Tensor>(k->To(promoted_type));
    auto v_p  = v_dtype  == promoted_type ? v  : std::make_shared<Tensor>(v->To(promoted_type));

    // dQ has shape of Q (B, T, H_q, D), dK/dV have shape of K/V (B, T, H_kv, D)
    auto dq = std::make_shared<Tensor>(q->Dims(), promoted_type, q->GetDevice());
    auto dk = std::make_shared<Tensor>(k->Dims(), promoted_type, k->GetDevice());
    auto dv = std::make_shared<Tensor>(v->Dims(), promoted_type, q->GetDevice());

    // Prepare mask
    std::shared_ptr<Tensor> mask_f32;
    const float* mask_ptr = nullptr;
    int64_t ms_b = 0, ms_h = 0, ms_m = 0, ms_n = 0;
    PrepareMask(attn_mask, mask_f32, mask_ptr, ms_b, ms_h, ms_m, ms_n);

    const int HD = static_cast<int>(q->Dims()[3]);

    switch (promoted_type) {
        DISPATCH_CASE(
            WRAP(DispatchBackwardByHD<float>(HD, dq, dk, dv, dy_p, q_p, k_p, v_p,
                                             mask_ptr, ms_b, ms_h, ms_m, ms_n,
                                             static_cast<float>(scale), is_causal);),
            DataType::kFLOAT32)
        DISPATCH_CASE(
            WRAP(DispatchBackwardByHD<nv_bfloat16>(HD, dq, dk, dv, dy_p, q_p, k_p, v_p,
                                                   mask_ptr, ms_b, ms_h, ms_m, ms_n,
                                                   static_cast<float>(scale), is_causal);),
            DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "FlashAttention backward: unsupported dtype");
    }

    return {dq, dk, dv, nullptr};
}

} // namespace infini_train::kernels::cuda

// ============================================================================
// Kernel registration — identical to original
// ============================================================================
#define REGISTER_CUDA_SDPA_KERNEL(kernel_name)                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name,   \
                    infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionForward)
REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionBackward)

#undef REGISTER_CUDA_SDPA_KERNEL

