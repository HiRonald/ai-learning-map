#include "rnn/cell.h"

#include "nn/activator.h"

#include <cmath>
#include <stdexcept>

namespace {

Matrix tanh_of(const Matrix& z) {
    static Activator act("tanh");
    return act.forward(z);
}

Matrix tanh_grad_from_z(const Matrix& z) {
    static Activator act("tanh");
    return act.backward(z);
}

}  // namespace

RnnCell::RnnCell(int input_size, int hidden_size, double learning_rate)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      lr_(learning_rate),
      w_xh_(input_size, hidden_size),
      w_hh_(hidden_size, hidden_size),
      b_(1, hidden_size, 0.0),
      grad_w_xh_(input_size, hidden_size, 0.0),
      grad_w_hh_(hidden_size, hidden_size, 0.0),
      grad_b_(1, hidden_size, 0.0) {
    if (input_size <= 0 || hidden_size <= 0) {
        throw std::invalid_argument("RnnCell: sizes must be positive");
    }
    if (lr_ <= 0.0) {
        throw std::invalid_argument("RnnCell: learning rate must be positive");
    }
    // Xavier 均匀，对 tanh 友好
    const double lim_xh = std::sqrt(6.0 / (input_size_ + hidden_size_));
    const double lim_hh = std::sqrt(6.0 / (hidden_size_ + hidden_size_));
    w_xh_.randomize(-lim_xh, lim_xh);
    w_hh_.randomize(-lim_hh, lim_hh);
}

Matrix RnnCell::forward(const std::vector<Matrix>& xs) {
    if (xs.empty()) {
        throw std::invalid_argument("RnnCell::forward: empty sequence");
    }
    const int B = xs[0].get_row_number();
    if (B <= 0) {
        throw std::invalid_argument("RnnCell::forward: batch must be positive");
    }

    xs_.clear();
    h_prevs_.clear();
    zs_.clear();
    hs_.clear();
    xs_.reserve(xs.size());
    h_prevs_.reserve(xs.size());
    zs_.reserve(xs.size());
    hs_.reserve(xs.size());

    Matrix h(B, hidden_size_, 0.0);  // h_0 = 0
    for (const Matrix& x : xs) {
        if (x.get_row_number() != B || x.get_col_number() != input_size_) {
            throw std::invalid_argument("RnnCell::forward: x_t shape mismatch");
        }
        Matrix z = x.dot(w_xh_).add(h.dot(w_hh_)).add(b_);
        Matrix h_next = tanh_of(z);

        xs_.push_back(x);
        h_prevs_.push_back(h);
        zs_.push_back(z);
        hs_.push_back(h_next);
        h = h_next;
    }
    return h;
}

Matrix RnnCell::backward(const Matrix& grad_h_last) {
    const int T = static_cast<int>(hs_.size());
    if (T == 0) {
        throw std::runtime_error("RnnCell::backward: call forward first");
    }
    if (grad_h_last.get_row_number() != hs_.back().get_row_number() ||
        grad_h_last.get_col_number() != hidden_size_) {
        throw std::invalid_argument("RnnCell::backward: grad_h_last shape mismatch");
    }

    Matrix dh = grad_h_last;
    for (int t = T - 1; t >= 0; --t) {
        // h = tanh(z) ⇒ dz = dh ⊙ tanh'(z)
        Matrix dz = dh.mul(tanh_grad_from_z(zs_[t]));

        // 累加共享参数梯度
        grad_w_xh_ = grad_w_xh_ + xs_[t].transpose().dot(dz);
        grad_w_hh_ = grad_w_hh_ + h_prevs_[t].transpose().dot(dz);
        grad_b_ = grad_b_ + dz.sum_rows();

        // ∂L/∂h_{t-1}
        dh = dz.dot(w_hh_.transpose());
    }
    return dh;  // ∂L/∂h_0
}

void RnnCell::zero_grad() {
    grad_w_xh_.fill(0.0);
    grad_w_hh_.fill(0.0);
    grad_b_.fill(0.0);
}

void RnnCell::apply_gradients() {
    w_xh_ = w_xh_ - grad_w_xh_.mul(lr_);
    w_hh_ = w_hh_ - grad_w_hh_.mul(lr_);
    b_ = b_ - grad_b_.mul(lr_);
    zero_grad();
}

int RnnCell::input_size() const { return input_size_; }
int RnnCell::hidden_size() const { return hidden_size_; }

int RnnCell::param_count() const {
    return input_size_ * hidden_size_ + hidden_size_ * hidden_size_ + hidden_size_;
}

double RnnCell::learning_rate() const { return lr_; }

void RnnCell::set_learning_rate(double lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("RnnCell: learning rate must be positive");
    }
    lr_ = lr;
}

void RnnCell::set_weights_xh(const Matrix& w) {
    if (w.get_row_number() != input_size_ || w.get_col_number() != hidden_size_) {
        throw std::invalid_argument("RnnCell::set_weights_xh shape mismatch");
    }
    w_xh_ = w;
}

void RnnCell::set_weights_hh(const Matrix& w) {
    if (w.get_row_number() != hidden_size_ || w.get_col_number() != hidden_size_) {
        throw std::invalid_argument("RnnCell::set_weights_hh shape mismatch");
    }
    w_hh_ = w;
}

void RnnCell::set_bias(const Matrix& b) {
    if (b.get_row_number() != 1 || b.get_col_number() != hidden_size_) {
        throw std::invalid_argument("RnnCell::set_bias shape mismatch, expect (1, H)");
    }
    b_ = b;
}

const Matrix& RnnCell::weights_xh() const { return w_xh_; }
const Matrix& RnnCell::weights_hh() const { return w_hh_; }
const Matrix& RnnCell::bias() const { return b_; }
const std::vector<Matrix>& RnnCell::hidden_states() const { return hs_; }
