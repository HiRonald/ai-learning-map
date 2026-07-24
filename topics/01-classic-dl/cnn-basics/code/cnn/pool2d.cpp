#include "cnn/pool2d.h"

#include <limits>
#include <stdexcept>

MaxPool2d::MaxPool2d(int kernel_size, int stride) : k_(kernel_size), stride_(stride) {
    if (kernel_size <= 0 || stride <= 0) {
        throw std::invalid_argument("MaxPool2d: invalid hyperparameters");
    }
}

int MaxPool2d::out_dim(int in_dim) const {
    return (in_dim - k_) / stride_ + 1;
}

Tensor MaxPool2d::forward(const Tensor& input) {
    const int oh = out_dim(input.h());
    const int ow = out_dim(input.w());
    if (oh <= 0 || ow <= 0) {
        throw std::invalid_argument("MaxPool2d::forward non-positive output size");
    }
    input_ = input;
    Tensor out(input.n(), input.c(), oh, ow, 0.0);
    max_index_.assign(static_cast<size_t>(out.size()), -1);

    int out_flat = 0;
    for (int n = 0; n < input.n(); ++n) {
        for (int c = 0; c < input.c(); ++c) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    double best = -std::numeric_limits<double>::infinity();
                    int best_idx = -1;
                    const int h0 = i * stride_;
                    const int w0 = j * stride_;
                    for (int u = 0; u < k_; ++u) {
                        for (int v = 0; v < k_; ++v) {
                            const int ih = h0 + u;
                            const int iw = w0 + v;
                            const double val = input.at(n, c, ih, iw);
                            if (val > best) {
                                best = val;
                                best_idx = ih * input.w() + iw;
                            }
                        }
                    }
                    out.at(n, c, i, j) = best;
                    max_index_[static_cast<size_t>(out_flat++)] = best_idx;
                }
            }
        }
    }
    return out;
}

Tensor MaxPool2d::backward(const Tensor& upstream_grad) {
    const int oh = out_dim(input_.h());
    const int ow = out_dim(input_.w());
    if (upstream_grad.n() != input_.n() || upstream_grad.c() != input_.c() ||
        upstream_grad.h() != oh || upstream_grad.w() != ow) {
        throw std::invalid_argument("MaxPool2d::backward gradient shape mismatch");
    }

    Tensor grad_x(input_.n(), input_.c(), input_.h(), input_.w(), 0.0);
    int out_flat = 0;
    for (int n = 0; n < input_.n(); ++n) {
        for (int c = 0; c < input_.c(); ++c) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    const int idx = max_index_[static_cast<size_t>(out_flat++)];
                    const int ih = idx / input_.w();
                    const int iw = idx % input_.w();
                    grad_x.at(n, c, ih, iw) += upstream_grad.at(n, c, i, j);
                }
            }
        }
    }
    return grad_x;
}

int MaxPool2d::kernel_size() const { return k_; }
int MaxPool2d::stride() const { return stride_; }
