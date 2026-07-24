#pragma once

#include "cnn/tensor.h"
#include "nn/matrix.h"

#include <vector>

namespace cnn_ops {

// 把 Matrix 行向量 (N, H*W) 收成单通道 Tensor (N,1,H,W)
Tensor images_from_matrix(const Matrix& x, int height, int width);

// (N,C,H,W) → (N, C*H*W)
Matrix flatten(const Tensor& t);

// (N, C*H*W) → (N,C,H,W)
Tensor unflatten(const Matrix& m, int channels, int height, int width);

Tensor relu_forward(const Tensor& x, std::vector<char>& mask);
Tensor relu_backward(const Tensor& upstream, const std::vector<char>& mask);

// 对单通道平面做卷积（不训练）：out = bias + sum_uv k[u,v]*in[...]
Tensor conv2d_fixed(const Tensor& input, const std::vector<double>& kernel_kxk, int k,
                    double bias = 0.0);

}  // namespace cnn_ops
