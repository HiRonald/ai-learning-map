#include "cnn/ops.h"

#include <stdexcept>

namespace cnn_ops {

Tensor images_from_matrix(const Matrix& x, int height, int width) {
    if (x.get_col_number() != height * width) {
        throw std::invalid_argument("images_from_matrix: feature size != H*W");
    }
    Tensor t(x.get_row_number(), 1, height, width);
    for (int n = 0; n < x.get_row_number(); ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                t.at(n, 0, h, w) = x.get_data(n, h * width + w);
            }
        }
    }
    return t;
}

Matrix flatten(const Tensor& t) {
    const int feats = t.c() * t.h() * t.w();
    Matrix m(t.n(), feats);
    for (int n = 0; n < t.n(); ++n) {
        int col = 0;
        for (int c = 0; c < t.c(); ++c) {
            for (int h = 0; h < t.h(); ++h) {
                for (int w = 0; w < t.w(); ++w) {
                    m.set_data(n, col++, t.at(n, c, h, w));
                }
            }
        }
    }
    return m;
}

Tensor unflatten(const Matrix& m, int channels, int height, int width) {
    if (m.get_col_number() != channels * height * width) {
        throw std::invalid_argument("unflatten: feature size mismatch");
    }
    Tensor t(m.get_row_number(), channels, height, width);
    for (int n = 0; n < m.get_row_number(); ++n) {
        int col = 0;
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    t.at(n, c, h, w) = m.get_data(n, col++);
                }
            }
        }
    }
    return t;
}

Tensor relu_forward(const Tensor& x, std::vector<char>& mask) {
    Tensor y(x.n(), x.c(), x.h(), x.w());
    mask.resize(static_cast<size_t>(x.size()));
    for (int i = 0; i < x.size(); ++i) {
        const double v = x.data()[static_cast<size_t>(i)];
        const bool keep = v > 0.0;
        mask[static_cast<size_t>(i)] = keep ? 1 : 0;
        y.data()[static_cast<size_t>(i)] = keep ? v : 0.0;
    }
    return y;
}

Tensor relu_backward(const Tensor& upstream, const std::vector<char>& mask) {
    if (static_cast<int>(mask.size()) != upstream.size()) {
        throw std::invalid_argument("relu_backward: mask size mismatch");
    }
    Tensor g(upstream.n(), upstream.c(), upstream.h(), upstream.w());
    for (int i = 0; i < upstream.size(); ++i) {
        g.data()[static_cast<size_t>(i)] =
            mask[static_cast<size_t>(i)] ? upstream.data()[static_cast<size_t>(i)] : 0.0;
    }
    return g;
}

Tensor conv2d_fixed(const Tensor& input, const std::vector<double>& kernel_kxk, int k,
                    double bias) {
    if (static_cast<int>(kernel_kxk.size()) != k * k) {
        throw std::invalid_argument("conv2d_fixed: kernel size mismatch");
    }
    if (input.c() != 1) {
        throw std::invalid_argument("conv2d_fixed: expects single-channel input");
    }
    const int pad = k / 2;
    const int oh = input.h();  // same padding for odd k
    const int ow = input.w();
    Tensor out(input.n(), 1, oh, ow, 0.0);
    for (int n = 0; n < input.n(); ++n) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                double sum = bias;
                for (int u = 0; u < k; ++u) {
                    for (int v = 0; v < k; ++v) {
                        const int ih = i + u - pad;
                        const int iw = j + v - pad;
                        double x = 0.0;
                        if (ih >= 0 && iw >= 0 && ih < input.h() && iw < input.w()) {
                            x = input.at(n, 0, ih, iw);
                        }
                        sum += kernel_kxk[static_cast<size_t>(u * k + v)] * x;
                    }
                }
                out.at(n, 0, i, j) = sum;
            }
        }
    }
    return out;
}

}  // namespace cnn_ops
