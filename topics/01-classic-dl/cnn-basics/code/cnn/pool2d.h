#pragma once

#include "cnn/tensor.h"

#include <vector>

// MaxPool2d：每个窗口取 max；反向只把梯度送回 argmax 位置。
class MaxPool2d {
public:
    MaxPool2d(int kernel_size, int stride);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& upstream_grad);

    int kernel_size() const;
    int stride() const;

private:
    int k_;
    int stride_;
    Tensor input_;
    // 与输出同形，存每个窗口内赢家的平坦下标（相对输入单通道平面）
    std::vector<int> max_index_;

    int out_dim(int in_dim) const;
};
