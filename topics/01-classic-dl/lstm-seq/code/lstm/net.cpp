#include "lstm/net.h"

#include "nn/activator.h"

#include <iostream>
#include <stdexcept>

namespace {

float mse_loss(const Matrix& output, const Matrix& target) {
    Matrix diff = output - target;
    return static_cast<float>(0.5 * diff.mul(diff).sum() / output.get_row_number());
}

Matrix mse_gradient(const Matrix& output, const Matrix& target) {
    return (output - target) / static_cast<double>(output.get_row_number());
}

}  // namespace

LstmNet::LstmNet(int input_size, int hidden_size, double learning_rate)
    : cell_(input_size, hidden_size, learning_rate),
      head_(std::make_shared<Layer>(hidden_size, 1, std::make_shared<Activator>("identity"),
                                   learning_rate)) {
    if (input_size <= 0 || hidden_size <= 0) {
        throw std::invalid_argument("LstmNet: sizes must be positive");
    }
}

Matrix LstmNet::forward(const std::vector<Matrix>& xs) {
    Matrix h_last = cell_.forward(xs);
    output_ = head_->forward(h_last);
    return output_;
}

float LstmNet::train(const std::vector<Matrix>& xs, const Matrix& targets) {
    if (targets.get_col_number() != 1) {
        throw std::invalid_argument("LstmNet::train: targets must be (B, 1)");
    }
    cell_.zero_grad();
    forward(xs);
    if (targets.get_row_number() != output_.get_row_number()) {
        throw std::invalid_argument("LstmNet::train: batch size mismatch");
    }

    loss_ = mse_loss(output_, targets);
    Matrix grad_y = mse_gradient(output_, targets);
    Matrix grad_h = head_->backward(grad_y);
    cell_.backward(grad_h);
    cell_.apply_gradients();
    return loss_;
}

Matrix LstmNet::predict(const std::vector<Matrix>& xs) { return forward(xs); }

float LstmNet::eval_mse(const Matrix& targets) const {
    return mse_loss(output_, targets);
}

int LstmNet::param_count() const {
    return cell_.param_count() + cell_.hidden_size() + 1;
}

void LstmNet::print_architecture() const {
    std::cout << "LstmNet: x_t (" << cell_.input_size() << ") → shared LSTM cell (h="
              << cell_.hidden_size() << ") × T → h_T → Dense(1, identity)"
              << "  params=" << param_count() << "\n";
}

LstmCell& LstmNet::cell() { return cell_; }
const LstmCell& LstmNet::cell() const { return cell_; }
