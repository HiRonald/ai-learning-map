#pragma once

#include "cnn/tensor.h"

// 2D 卷积：Y = Conv(X; W, b)
// 权重形状 conceptually (out_c, in_c, k, k)，平坦存于 weights_。
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
           double learning_rate);

    Tensor forward(const Tensor& input);
    // 返回 dL/dX；内部用 SGD 更新 W/b
    Tensor backward(const Tensor& upstream_grad);

    int in_channels() const;
    int out_channels() const;
    int kernel_size() const;
    int stride() const;
    int padding() const;
    int param_count() const;

    const Tensor& weights() const;
    const Tensor& biases() const;  // 存成 (1, out_c, 1, 1)

    void set_learning_rate(double lr);
    void set_weights(const Tensor& w);
    void set_biases(const Tensor& b);

private:
    int in_c_;
    int out_c_;
    int k_;
    int stride_;
    int pad_;
    double lr_;
    Tensor weights_;  // (out_c, in_c, k, k)
    Tensor biases_;   // (1, out_c, 1, 1)
    Tensor input_;    // 缓存前向输入

    int out_dim(int in_dim) const;
    double get_input_or_zero(int n, int c, int h, int w) const;
};
