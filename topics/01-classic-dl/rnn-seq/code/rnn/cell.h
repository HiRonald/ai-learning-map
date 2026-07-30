#pragma once

#include "nn/matrix.h"

#include <vector>

// 经典 RNN 单元（参数沿时间共享）：
//   h_t = tanh(x_t W_xh + h_{t-1} W_hh + b)
// 前向缓存每步；反向 BPTT 累加梯度，调用 apply_gradients() 再更新。
class RnnCell {
public:
    RnnCell(int input_size, int hidden_size, double learning_rate);

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
    void set_weights_xh(const Matrix& w);
    void set_weights_hh(const Matrix& w);
    void set_bias(const Matrix& b);

    const Matrix& weights_xh() const;
    const Matrix& weights_hh() const;
    const Matrix& bias() const;

    // 供 unroll 可视化：前向后取各步 h_t（长度 T）
    const std::vector<Matrix>& hidden_states() const;

private:
    int input_size_;
    int hidden_size_;
    double lr_;

    Matrix w_xh_;  // (I, H)
    Matrix w_hh_;  // (H, H)
    Matrix b_;     // (1, H)

    Matrix grad_w_xh_;
    Matrix grad_w_hh_;
    Matrix grad_b_;

    // 前向缓存
    std::vector<Matrix> xs_;       // T × (B, I)
    std::vector<Matrix> h_prevs_;  // T × (B, H)，含 h_0
    std::vector<Matrix> zs_;       // T × (B, H)
    std::vector<Matrix> hs_;       // T × (B, H) = h_1..h_T
};
