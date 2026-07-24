#pragma once

#include "nn/layer.h"
#include "nn/matrix.h"

#include <memory>

// 残差块：y = x + F(x)，F = Linear→ReLU→Linear（维数不变）
// 反向：dL/dx = dL/dy + dL/dF（恒等捷径 + F 的梯度）
class ResidualBlock {
public:
    ResidualBlock(int dim, double learning_rate);

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& upstream_grad);

    int dim() const;
    int param_count() const;

private:
    int dim_;
    std::shared_ptr<Layer> f1_;  // dim→dim, relu
    std::shared_ptr<Layer> f2_;  // dim→dim, identity
    Matrix input_;
};
