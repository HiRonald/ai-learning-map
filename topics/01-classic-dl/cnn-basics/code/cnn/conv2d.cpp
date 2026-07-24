#include "cnn/conv2d.h"

#include <cmath>
#include <stdexcept>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
               double learning_rate)
    : in_c_(in_channels),
      out_c_(out_channels),
      k_(kernel_size),
      stride_(stride),
      pad_(padding),
      lr_(learning_rate),
      weights_(out_channels, in_channels, kernel_size, kernel_size),
      biases_(1, out_channels, 1, 1, 0.0) {
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0) {
        throw std::invalid_argument("Conv2d: invalid hyperparameters");
    }
    if (lr_ <= 0.0) {
        throw std::invalid_argument("Conv2d: learning rate must be positive");
    }
    // He-ish：fan_in = in_c * k * k
    const double fan_in = static_cast<double>(in_c_) * k_ * k_;
    const double limit = std::sqrt(6.0 / fan_in);
    weights_.randomize(-limit, limit);
}

int Conv2d::out_dim(int in_dim) const {
    return (in_dim + 2 * pad_ - k_) / stride_ + 1;
}

double Conv2d::get_input_or_zero(int n, int c, int h, int w) const {
    if (h < 0 || w < 0 || h >= input_.h() || w >= input_.w()) {
        return 0.0;
    }
    return input_.at(n, c, h, w);
}

Tensor Conv2d::forward(const Tensor& input) {
    if (input.c() != in_c_) {
        throw std::invalid_argument("Conv2d::forward channel mismatch");
    }
    input_ = input;
    const int oh = out_dim(input.h());
    const int ow = out_dim(input.w());
    if (oh <= 0 || ow <= 0) {
        throw std::invalid_argument("Conv2d::forward non-positive output spatial size");
    }

    Tensor out(input.n(), out_c_, oh, ow, 0.0);
    for (int n = 0; n < input.n(); ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            const double bias = biases_.at(0, oc, 0, 0);
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    double sum = bias;
                    const int h0 = i * stride_ - pad_;
                    const int w0 = j * stride_ - pad_;
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int u = 0; u < k_; ++u) {
                            for (int v = 0; v < k_; ++v) {
                                sum += weights_.at(oc, ic, u, v) *
                                       get_input_or_zero(n, ic, h0 + u, w0 + v);
                            }
                        }
                    }
                    out.at(n, oc, i, j) = sum;
                }
            }
        }
    }
    return out;
}

Tensor Conv2d::backward(const Tensor& upstream_grad) {
    const int oh = out_dim(input_.h());
    const int ow = out_dim(input_.w());
    if (upstream_grad.n() != input_.n() || upstream_grad.c() != out_c_ ||
        upstream_grad.h() != oh || upstream_grad.w() != ow) {
        throw std::invalid_argument("Conv2d::backward gradient shape mismatch");
    }

    Tensor grad_w(out_c_, in_c_, k_, k_, 0.0);
    Tensor grad_b(1, out_c_, 1, 1, 0.0);
    Tensor grad_x(input_.n(), in_c_, input_.h(), input_.w(), 0.0);

    for (int n = 0; n < input_.n(); ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    const double g = upstream_grad.at(n, oc, i, j);
                    grad_b.at(0, oc, 0, 0) += g;
                    const int h0 = i * stride_ - pad_;
                    const int w0 = j * stride_ - pad_;
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int u = 0; u < k_; ++u) {
                            for (int v = 0; v < k_; ++v) {
                                const int ih = h0 + u;
                                const int iw = w0 + v;
                                const double x = get_input_or_zero(n, ic, ih, iw);
                                grad_w.at(oc, ic, u, v) += g * x;
                                if (ih >= 0 && iw >= 0 && ih < input_.h() && iw < input_.w()) {
                                    grad_x.at(n, ic, ih, iw) += g * weights_.at(oc, ic, u, v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // SGD：θ ← θ - lr * ∇θ
    for (int i = 0; i < weights_.size(); ++i) {
        weights_.data()[static_cast<size_t>(i)] -= lr_ * grad_w.data()[static_cast<size_t>(i)];
    }
    for (int oc = 0; oc < out_c_; ++oc) {
        biases_.at(0, oc, 0, 0) -= lr_ * grad_b.at(0, oc, 0, 0);
    }

    return grad_x;
}

int Conv2d::in_channels() const { return in_c_; }
int Conv2d::out_channels() const { return out_c_; }
int Conv2d::kernel_size() const { return k_; }
int Conv2d::stride() const { return stride_; }
int Conv2d::padding() const { return pad_; }

int Conv2d::param_count() const {
    return out_c_ * (in_c_ * k_ * k_ + 1);
}

const Tensor& Conv2d::weights() const { return weights_; }
const Tensor& Conv2d::biases() const { return biases_; }

void Conv2d::set_learning_rate(double lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("Conv2d: learning rate must be positive");
    }
    lr_ = lr;
}

void Conv2d::set_weights(const Tensor& w) {
    if (w.n() != out_c_ || w.c() != in_c_ || w.h() != k_ || w.w() != k_) {
        throw std::invalid_argument("Conv2d::set_weights shape mismatch");
    }
    weights_ = w;
}

void Conv2d::set_biases(const Tensor& b) {
    if (b.n() != 1 || b.c() != out_c_ || b.h() != 1 || b.w() != 1) {
        throw std::invalid_argument("Conv2d::set_biases shape mismatch");
    }
    biases_ = b;
}
