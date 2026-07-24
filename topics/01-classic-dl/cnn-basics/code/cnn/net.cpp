#include "cnn/net.h"

#include "cnn/ops.h"
#include "nn/activator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

Matrix softmax_rows(const Matrix& logits) {
    const int rows = logits.get_row_number();
    const int cols = logits.get_col_number();
    Matrix probs(rows, cols);
    for (int i = 0; i < rows; ++i) {
        double max_v = logits.get_data(i, 0);
        for (int j = 1; j < cols; ++j) {
            max_v = std::max(max_v, logits.get_data(i, j));
        }
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            double e = std::exp(logits.get_data(i, j) - max_v);
            probs.set_data(i, j, e);
            sum += e;
        }
        for (int j = 0; j < cols; ++j) {
            probs.set_data(i, j, probs.get_data(i, j) / sum);
        }
    }
    return probs;
}

float softmax_ce_loss(const Matrix& logits, const Matrix& label) {
    Matrix probs = softmax_rows(logits);
    double total = 0.0;
    const int rows = logits.get_row_number();
    const int cols = logits.get_col_number();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double y = label.get_data(i, j);
            if (y <= 0.0) {
                continue;
            }
            double p = std::max(1e-12, probs.get_data(i, j));
            total += -y * std::log(p);
        }
    }
    return static_cast<float>(total / rows);
}

Matrix softmax_ce_gradient(const Matrix& logits, const Matrix& label) {
    Matrix probs = softmax_rows(logits);
    return (probs - label) / static_cast<double>(logits.get_row_number());
}

}  // namespace

int LeNet::dense_params(int in_f, int out_f) {
    return in_f * out_f + out_f;
}

Tensor LeNet::pad_to_32(const Tensor& x28) {
    if (x28.c() != 1 || x28.h() != kImageSide || x28.w() != kImageSide) {
        throw std::invalid_argument("LeNet::pad_to_32 expects (N,1,28,28)");
    }
    Tensor out(x28.n(), 1, kPaddedSide, kPaddedSide, 0.0);
    constexpr int off = (kPaddedSide - kImageSide) / 2;  // 2
    for (int n = 0; n < x28.n(); ++n) {
        for (int h = 0; h < kImageSide; ++h) {
            for (int w = 0; w < kImageSide; ++w) {
                out.at(n, 0, h + off, w + off) = x28.at(n, 0, h, w);
            }
        }
    }
    return out;
}

LeNet::LeNet(double learning_rate)
    : lr_(learning_rate),
      conv1_(/*in*/ 1, /*out*/ 6, /*k*/ 5, /*s*/ 1, /*p*/ 0, learning_rate),
      pool1_(2, 2),
      conv2_(/*in*/ 6, /*out*/ 16, /*k*/ 5, /*s*/ 1, /*p*/ 0, learning_rate),
      pool2_(2, 2) {
    if (lr_ <= 0.0) {
        throw std::invalid_argument("LeNet: learning rate must be positive");
    }
    // 32 → conv5 → 28 → pool → 14 → conv5 → 10 → pool → 5；16*5*5=400
    fc1_ = std::make_shared<Layer>(kFlatFeats, 120, std::make_shared<Activator>("relu"), lr_);
    fc2_ = std::make_shared<Layer>(120, 84, std::make_shared<Activator>("relu"), lr_);
    fc3_ = std::make_shared<Layer>(84, kNumClasses, std::make_shared<Activator>("identity"), lr_);
}

void LeNet::forward(const Matrix& images_flat) {
    Tensor x28 = cnn_ops::images_from_matrix(images_flat, kImageSide, kImageSide);
    Tensor x = pad_to_32(x28);
    Tensor z1 = conv1_.forward(x);
    Tensor a1 = cnn_ops::relu_forward(z1, relu1_mask_);
    Tensor p1 = pool1_.forward(a1);
    Tensor z2 = conv2_.forward(p1);
    Tensor a2 = cnn_ops::relu_forward(z2, relu2_mask_);
    Tensor p2 = pool2_.forward(a2);
    Matrix flat = cnn_ops::flatten(p2);
    Matrix h1 = fc1_->forward(flat);
    Matrix h2 = fc2_->forward(h1);
    logits_ = fc3_->forward(h2);
}

void LeNet::backward(const Matrix& labels_one_hot) {
    last_loss_ = softmax_ce_loss(logits_, labels_one_hot);
    Matrix grad = softmax_ce_gradient(logits_, labels_one_hot);
    grad = fc3_->backward(grad);
    grad = fc2_->backward(grad);
    grad = fc1_->backward(grad);

    Tensor g = cnn_ops::unflatten(grad, 16, 5, 5);
    g = pool2_.backward(g);
    g = cnn_ops::relu_backward(g, relu2_mask_);
    g = conv2_.backward(g);
    g = pool1_.backward(g);
    g = cnn_ops::relu_backward(g, relu1_mask_);
    (void)conv1_.backward(g);  // pad 区域梯度丢弃即可（输入常数 0）
}

void LeNet::train(const Matrix& images_flat, const Matrix& labels_one_hot) {
    forward(images_flat);
    backward(labels_one_hot);
}

Matrix LeNet::predict(const Matrix& images_flat) {
    forward(images_flat);
    return logits_;
}

float LeNet::eval(const Matrix& labels_one_hot) const {
    return softmax_ce_loss(logits_, labels_one_hot);
}

int LeNet::param_count() const {
    return conv1_.param_count() + conv2_.param_count() + dense_params(kFlatFeats, 120) +
           dense_params(120, 84) + dense_params(84, kNumClasses);
}

void LeNet::print_architecture() const {
    std::cout << "LeNet-5 style (28→pad32, ReLU+MaxPool, softmax_ce, lr=" << lr_ << "):\n"
              << "  pad 28→32\n"
              << "  Conv2d 1→6  k=5 → 28x28  params=" << conv1_.param_count() << "\n"
              << "  ReLU → MaxPool 2x2 → 14x14\n"
              << "  Conv2d 6→16 k=5 → 10x10  params=" << conv2_.param_count() << "\n"
              << "  ReLU → MaxPool 2x2 → 5x5\n"
              << "  Flatten " << kFlatFeats << " → Dense 120 → 84 → 10\n"
              << "  total params ≈ " << param_count() << "\n";
}
