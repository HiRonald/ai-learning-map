#include "residual/cnn_net.h"

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

DeepCnn::DeepCnn(int channels, int num_res_blocks, Mode mode, double learning_rate)
    : mode_(mode),
      lr_(learning_rate),
      channels_(channels),
      num_res_blocks_(num_res_blocks),
      plain_convs_(2 * num_res_blocks),
      pool1_(2, 2),
      pool2_(2, 2) {
    if (channels <= 0 || num_res_blocks <= 0 || lr_ <= 0.0) {
        throw std::invalid_argument("DeepCnn: invalid hyperparameters");
    }
    stem_ = std::make_unique<Conv2d>(1, channels, 3, 1, 1, lr_);
    if (mode_ == Mode::Plain) {
        plain_relu_masks_.resize(static_cast<size_t>(plain_convs_));
        for (int i = 0; i < plain_convs_; ++i) {
            plain_convs_list_.push_back(
                std::make_unique<Conv2d>(channels, channels, 3, 1, 1, lr_));
        }
    } else {
        for (int i = 0; i < num_res_blocks_; ++i) {
            res_blocks_.push_back(std::make_unique<ResidualConvBlock>(channels, lr_));
        }
    }
    // 28→14→7
    flat_feats_ = channels_ * 7 * 7;
    head_ = std::make_shared<Layer>(flat_feats_, kNumClasses,
                                   std::make_shared<Activator>("identity"), lr_);
}

void DeepCnn::forward(const Matrix& images_flat) {
    Tensor x = cnn_ops::images_from_matrix(images_flat, kImageSide, kImageSide);
    Tensor z0 = stem_->forward(x);
    Tensor a0 = cnn_ops::relu_forward(z0, stem_relu_mask_);
    Tensor h = pool1_.forward(a0);  // (N,C,14,14)

    if (mode_ == Mode::Plain) {
        for (size_t i = 0; i < plain_convs_list_.size(); ++i) {
            Tensor z = plain_convs_list_[i]->forward(h);
            h = cnn_ops::relu_forward(z, plain_relu_masks_[i]);
        }
    } else {
        for (auto& block : res_blocks_) {
            h = block->forward(h);
        }
    }

    Tensor p2 = pool2_.forward(h);  // (N,C,7,7)
    Matrix flat = cnn_ops::flatten(p2);
    logits_ = head_->forward(flat);
}

void DeepCnn::backward(const Matrix& labels_one_hot) {
    last_loss_ = softmax_ce_loss(logits_, labels_one_hot);
    Matrix grad = softmax_ce_gradient(logits_, labels_one_hot);
    grad = head_->backward(grad);

    Tensor g = cnn_ops::unflatten(grad, channels_, 7, 7);
    g = pool2_.backward(g);

    if (mode_ == Mode::Plain) {
        for (int i = static_cast<int>(plain_convs_list_.size()) - 1; i >= 0; --i) {
            g = cnn_ops::relu_backward(g, plain_relu_masks_[static_cast<size_t>(i)]);
            g = plain_convs_list_[static_cast<size_t>(i)]->backward(g);
        }
    } else {
        for (int i = static_cast<int>(res_blocks_.size()) - 1; i >= 0; --i) {
            g = res_blocks_[static_cast<size_t>(i)]->backward(g);
        }
    }

    g = pool1_.backward(g);
    g = cnn_ops::relu_backward(g, stem_relu_mask_);
    (void)stem_->backward(g);
}

void DeepCnn::train(const Matrix& images_flat, const Matrix& labels_one_hot) {
    forward(images_flat);
    backward(labels_one_hot);
}

Matrix DeepCnn::predict(const Matrix& images_flat) {
    forward(images_flat);
    return logits_;
}

float DeepCnn::eval(const Matrix& labels_one_hot) const {
    return softmax_ce_loss(logits_, labels_one_hot);
}

int DeepCnn::param_count() const {
    int n = stem_->param_count();
    n += head_->get_input_size() * head_->get_output_size() + head_->get_output_size();
    if (mode_ == Mode::Plain) {
        for (const auto& c : plain_convs_list_) {
            n += c->param_count();
        }
    } else {
        for (const auto& b : res_blocks_) {
            n += b->param_count();
        }
    }
    return n;
}

const char* DeepCnn::mode_name() const {
    return mode_ == Mode::Plain ? "plain-cnn" : "residual-cnn";
}

void DeepCnn::print_architecture() const {
    std::cout << "DeepCnn [" << mode_name() << "] "
              << "stem 1→" << channels_ << " → pool → trunk("
              << (mode_ == Mode::Plain ? plain_convs_ : num_res_blocks_)
              << (mode_ == Mode::Plain ? " conv" : " res-block") << ") → pool → FC"
              << ", lr=" << lr_ << ", params≈" << param_count() << "\n";
    if (mode_ == Mode::Residual) {
        std::cout << "  trunk block: y = x + Conv(ReLU(Conv(x)))  (same C, k=3 p=1)\n";
    } else {
        std::cout << "  trunk: stacked Conv+ReLU (same #Conv as residual trunk)\n";
    }
}
