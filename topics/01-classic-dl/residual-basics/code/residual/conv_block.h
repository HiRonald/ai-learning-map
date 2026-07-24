#pragma once

#include "cnn/conv2d.h"
#include "cnn/tensor.h"

#include <memory>
#include <vector>

// 卷积残差块（通道/空间不变）：y = x + Conv(ReLU(Conv(x)))
// k=3,s=1,p=1；末端卷积近零初始化。
class ResidualConvBlock {
public:
    ResidualConvBlock(int channels, double learning_rate);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& upstream_grad);

    int channels() const;
    int param_count() const;

private:
    int channels_;
    std::unique_ptr<Conv2d> conv1_;
    std::unique_ptr<Conv2d> conv2_;
    std::vector<char> relu_mask_;
    Tensor input_;
};
