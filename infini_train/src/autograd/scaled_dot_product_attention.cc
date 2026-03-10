// scaled_dot_product_attention.cc
//
// Autograd wrapper for FlashAttention-2 style CUDA kernels.

#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include <cmath>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4)
        << "ScaledDotProductAttention expects 3 inputs (q, k, v) "
        << "or 4 inputs (q, k, v, attn_mask)";

    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];
    const std::shared_ptr<Tensor> attn_mask =
        (input_tensors.size() == 4 ? input_tensors[3] : nullptr);

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

    const int64_t num_heads_q = q->Dims()[2];
    const int64_t num_heads_kv = k->Dims()[2];

    if (enable_gqa_) {
        CHECK_GT(num_heads_kv, 0) << "num_heads_kv must be positive";
        CHECK_EQ(num_heads_q % num_heads_kv, 0)
            << "enable_gqa=true requires q_heads % kv_heads == 0";
    } else {
        CHECK_EQ(num_heads_q, num_heads_kv)
            << "enable_gqa=false requires q_heads == kv_heads";
    }

    CHECK_EQ(dropout_p_, 0.0)
        << "FlashAttention CUDA path currently requires dropout_p == 0. "
        << "Wire Philox RNG state before enabling flash dropout.";

    CHECK(attn_mask == nullptr)
        << "FlashAttention CUDA path currently does not support arbitrary attn_mask. "
        << "Fallback to the eager SDPA path when attn_mask is present.";

    const double d = static_cast<double>(q->Dims()[3]);
    const double scale = scale_.has_value() ? *scale_ : (1.0 / std::sqrt(d));
    const auto device = q->GetDevice().type();

    auto outputs = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionForward"},
        q,
        k,
        v,
        attn_mask,
        dropout_p_,
        is_causal_,
        scale,
        enable_gqa_);

    CHECK_EQ(outputs.size(), 2)
        << "FlashAttentionForward must return {output, softmax_lse}";

    saved_tensors_.clear();
    saved_tensors_.push_back(q);
    saved_tensors_.push_back(k);
    saved_tensors_.push_back(v);
    saved_tensors_.push_back(attn_mask);
    saved_tensors_.push_back(outputs[0]);  // output
    saved_tensors_.push_back(outputs[1]);  // softmax_lse

    return {outputs[0]};
}

void ScaledDotProductAttention::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &,
    const std::vector<std::shared_ptr<Tensor>> &) {
    // no-op
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1)
        << "ScaledDotProductAttention backward expects exactly 1 grad output";

    const auto &grad_output = grad_outputs[0];

    CHECK_GE(saved_tensors_.size(), 6)
        << "Expected saved tensors: q, k, v, attn_mask, output, softmax_lse";

    const auto &q = saved_tensors_[0];
    const auto &k = saved_tensors_[1];
    const auto &v = saved_tensors_[2];
    const auto &attn_mask = saved_tensors_[3];
    const auto &output = saved_tensors_[4];
    const auto &softmax_lse = saved_tensors_[5];

    CHECK(q && k && v && output && softmax_lse && grad_output)
        << "Backward received null tensor";

    const double d = static_cast<double>(q->Dims()[3]);
    const double scale = scale_.has_value() ? *scale_ : (1.0 / std::sqrt(d));
    const auto device = q->GetDevice().type();

    auto grads = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionBackward"},
        grad_output,
        q,
        k,
        v,
        output,
        softmax_lse,
        dropout_p_,
        is_causal_,
        scale,
        enable_gqa_);

    CHECK_GE(grads.size(), 3)
        << "FlashAttentionBackward must return at least {dq, dk, dv}";

    auto dq = grads[0];
    auto dk = grads[1];
    auto dv = grads[2];

    if (attn_mask != nullptr) {
        return {dq, dk, dv, std::shared_ptr<Tensor>()};
    }
    return {dq, dk, dv};
}

}  // namespace infini_train::autograd
