#include "residual/conv_block.h"

#include "cnn/ops.h"

#include <stdexcept>

namespace {

Tensor tensor_add(const Tensor& a, const Tensor& b) {
    if (a.n() != b.n() || a.c() != b.c() || a.h() != b.h() || a.w() != b.w()) {
        throw std::invalid_argument("tensor_add: shape mismatch");
    }
    Tensor out(a.n(), a.c(), a.h(), a.w());
    for (int i = 0; i < a.size(); ++i) {
        out.data()[static_cast<size_t>(i)] =
            a.data()[static_cast<size_t>(i)] + b.data()[static_cast<size_t>(i)];
    }
    return out;
}

void zero_conv_weights(Conv2d& conv) {
    Tensor w = conv.weights();
    w.fill(0.0);
    conv.set_weights(w);
    Tensor b = conv.biases();
    b.fill(0.0);
    conv.set_biases(b);
}

}  // namespace

ResidualConvBlock::ResidualConvBlock(int channels, double learning_rate) : channels_(channels) {
    if (channels <= 0) {
        throw std::invalid_argument("ResidualConvBlock: channels must be positive");
    }
    conv1_ = std::make_unique<Conv2d>(channels, channels, /*k*/ 3, /*s*/ 1, /*p*/ 1, learning_rate);
    conv2_ = std::make_unique<Conv2d>(channels, channels, /*k*/ 3, /*s*/ 1, /*p*/ 1, learning_rate);
    zero_conv_weights(*conv2_);
}

Tensor ResidualConvBlock::forward(const Tensor& input) {
    if (input.c() != channels_) {
        throw std::invalid_argument("ResidualConvBlock::forward channel mismatch");
    }
    input_ = input;
    Tensor z1 = conv1_->forward(input);
    Tensor a1 = cnn_ops::relu_forward(z1, relu_mask_);
    Tensor delta = conv2_->forward(a1);
    return tensor_add(input, delta);
}

Tensor ResidualConvBlock::backward(const Tensor& upstream_grad) {
    // y = x + F(x) ⇒ ∂L/∂x = ∂L/∂y + ∂L/∂F
    Tensor g_f = conv2_->backward(upstream_grad);
    g_f = cnn_ops::relu_backward(g_f, relu_mask_);
    g_f = conv1_->backward(g_f);
    return tensor_add(upstream_grad, g_f);
}

int ResidualConvBlock::channels() const { return channels_; }

int ResidualConvBlock::param_count() const {
    return conv1_->param_count() + conv2_->param_count();
}
