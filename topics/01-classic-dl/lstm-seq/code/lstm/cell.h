#pragma once

#include "nn/matrix.h"

#include <vector>

// LSTM 单元（参数沿时间共享）：
//   f,i,o = σ(·)    g = tanh(·)
//   c_t = f ⊙ c_{t-1} + i ⊙ g
//   h_t = o ⊙ tanh(c_t)
// 权重按门打包：W_x, W_h, b 的列顺序为 [f | i | g | o]，各 H 列。
// 前向缓存每步；反向 BPTT 累加梯度，调用 apply_gradients() 再更新。
class LstmCell {
public:
    LstmCell(int input_size, int hidden_size, double learning_rate);

    // xs[t]: (B, input)；返回最后一步 h_T，形状 (B, hidden)
    Matrix forward(const std::vector<Matrix>& xs);

    // 从 ∂L/∂h_T 回传；累加 dW，返回 ∂L/∂h_0（通常丢弃）
    Matrix backward(const Matrix& grad_h_last);

    void apply_gradients();
    void zero_grad();

    int input_size() const;
    int hidden_size() const;
    int param_count() const;
    double learning_rate() const;

    void set_learning_rate(double lr);
    void set_weights_x(const Matrix& w);   // (I, 4H)
    void set_weights_h(const Matrix& w);   // (H, 4H)
    void set_bias(const Matrix& b);        // (1, 4H)

    const Matrix& weights_x() const;
    const Matrix& weights_h() const;
    const Matrix& bias() const;

    // 供 gates 可视化：前向后取各步（长度 T）
    const std::vector<Matrix>& hidden_states() const;
    const std::vector<Matrix>& cell_states() const;
    const std::vector<Matrix>& forget_gates() const;
    const std::vector<Matrix>& input_gates() const;
    const std::vector<Matrix>& output_gates() const;
    const std::vector<Matrix>& candidates() const;

private:
    int input_size_;
    int hidden_size_;
    double lr_;

    Matrix w_x_;  // (I, 4H)
    Matrix w_h_;  // (H, 4H)
    Matrix b_;    // (1, 4H)

    Matrix grad_w_x_;
    Matrix grad_w_h_;
    Matrix grad_b_;

    // 前向缓存
    std::vector<Matrix> xs_;
    std::vector<Matrix> h_prevs_;
    std::vector<Matrix> c_prevs_;
    std::vector<Matrix> fs_;
    std::vector<Matrix> is_;
    std::vector<Matrix> gs_;
    std::vector<Matrix> os_;
    std::vector<Matrix> cs_;
    std::vector<Matrix> hs_;
};
