#pragma once

#include "lstm/cell.h"
#include "nn/layer.h"
#include "nn/matrix.h"

#include <memory>
#include <vector>

// LSTM 序列模型：扫完 T 步 → 用最后隐藏态 Dense → 标量
class LstmNet {
public:
    LstmNet(int input_size, int hidden_size, double learning_rate);

    // xs[t]: (B, input)；返回 (B, 1)
    Matrix forward(const std::vector<Matrix>& xs);

    float train(const std::vector<Matrix>& xs, const Matrix& targets);
    Matrix predict(const std::vector<Matrix>& xs);
    float eval_mse(const Matrix& targets) const;

    int param_count() const;
    void print_architecture() const;

    LstmCell& cell();
    const LstmCell& cell() const;

private:
    LstmCell cell_;
    std::shared_ptr<Layer> head_;  // H → 1, identity
    Matrix output_;
    float loss_ = 0.0f;
};
