#pragma once

#include "nn/layer.h"
#include "nn/matrix.h"
#include "residual/block.h"

#include <memory>
#include <vector>

// 深度同维映射：dim → dim，用于 identity toy（学 y=x）
// plain：N 层 ReLU 全连接；residual：N 个残差块
class DeepMap {
public:
    enum class Mode { Plain, Residual };

    DeepMap(int dim, int num_blocks, Mode mode, double learning_rate);

    void train(const Matrix& x, const Matrix& y);
    Matrix predict(const Matrix& x);
    float eval_mse(const Matrix& y) const;

    void print_architecture() const;
    int param_count() const;
    const char* mode_name() const;

private:
    Mode mode_;
    double lr_;
    int dim_;
    int num_blocks_;
    std::vector<std::shared_ptr<Layer>> plain_layers_;
    std::vector<std::shared_ptr<ResidualBlock>> res_blocks_;
    Matrix output_;
    float last_loss_ = 0.0f;

    void forward(const Matrix& x);
    void backward(const Matrix& y);
};
