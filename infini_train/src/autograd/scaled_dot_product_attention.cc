// scaled_dot_product_attention.cc
//
// Autograd wrapper for FlashAttention-2 style CUDA kernels (flash_attention.cu).

#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include <cmath>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {

    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4)
        << "ScaledDotProductAttention expects 3 inputs (q,k,v) or 4 inputs (q,k,v,attn_mask)";

    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];
    const std::shared_ptr<Tensor> attn_mask =
        (input_tensors.size() == 4 ? input_tensors[3] : nullptr);

    CHECK(q && k && v);
    CHECK_EQ(q->Dims().size(), 4) << "Expected q shape (B,T,H,D)";
    CHECK_EQ(k->Dims().size(), 4) << "Expected k shape (B,T,Hk,D)";
    CHECK_EQ(v->Dims().size(), 4) << "Expected v shape (B,T,Hk,D)";

    CHECK_EQ(q->Dims()[0], k->Dims()[0]);
    CHECK_EQ(q->Dims()[0], v->Dims()[0]);
    CHECK_EQ(q->Dims()[1], k->Dims()[1]);
    CHECK_EQ(q->Dims()[1], v->Dims()[1]);
    CHECK_EQ(q->Dims()[3], k->Dims()[3]);
    CHECK_EQ(q->Dims()[3], v->Dims()[3]);

    //CHECK_EQ(q->Dims()[2], k->Dims()[2]) << "CUDA FlashAttention kernel requires q_heads == kv_heads";
    //CHECK_EQ(q->Dims()[2], v->Dims()[2]) << "CUDA FlashAttention kernel requires q_heads == kv_heads";

    CHECK_EQ(dropout_p_, 0.0) << "dropout_p > 0 not implemented";

    const double d = static_cast<double>(q->Dims()[3]);
    const double scale = scale_.has_value() ? *scale_ : (1.0 / std::sqrt(d));

    auto device = q->GetDevice().type();

    auto y = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "ScaledDotProductAttentionForward"},
        q, k, v,
        attn_mask,
        dropout_p_,
        is_causal_,
        scale,
        enable_gqa_);

    saved_tensors_.clear();
    saved_tensors_.push_back(q);
    saved_tensors_.push_back(k);
    saved_tensors_.push_back(v);
    saved_tensors_.push_back(attn_mask);

    return {y};
}

void ScaledDotProductAttention::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &,
    const std::vector<std::shared_ptr<Tensor>> &) {
    // no-op
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {

    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK_GE(saved_tensors_.size(), 4) << "Expected saved tensors: q,k,v,mask";
    const auto &q = saved_tensors_[0];
    const auto &k = saved_tensors_[1];
    const auto &v = saved_tensors_[2];
    const auto &attn_mask = saved_tensors_[3];

    const double d = static_cast<double>(q->Dims()[3]);
    const double scale = scale_.has_value() ? *scale_ : (1.0 / std::sqrt(d));

    auto device = q->GetDevice().type();

    auto grads = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "ScaledDotProductAttentionBackward"},
        grad_output, q, k, v,
        attn_mask,
        dropout_p_,
        is_causal_,
        scale,
        enable_gqa_);

    CHECK_GE(grads.size(), 3);
    auto dq = grads[0];
    auto dk = grads[1];
    auto dv = grads[2];

    if (attn_mask) {
        return {dq, dk, dv, std::shared_ptr<Tensor>()};
    }
    return {dq, dk, dv};
}

} // namespace infini_train::autograd
