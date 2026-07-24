#include "residual/map_net.h"

#include "nn/activator.h"

#include <iostream>
#include <stdexcept>

namespace {

float mse_loss(const Matrix& output, const Matrix& label) {
    Matrix diff = output - label;
    return static_cast<float>(0.5 * diff.mul(diff).sum() / output.get_row_number());
}

Matrix mse_gradient(const Matrix& output, const Matrix& label) {
    return (output - label) / static_cast<double>(output.get_row_number());
}

}  // namespace

DeepMap::DeepMap(int dim, int num_blocks, Mode mode, double learning_rate)
    : mode_(mode), lr_(learning_rate), dim_(dim), num_blocks_(num_blocks) {
    if (dim <= 0 || num_blocks <= 0 || lr_ <= 0.0) {
        throw std::invalid_argument("DeepMap: invalid hyperparameters");
    }
    if (mode_ == Mode::Plain) {
        for (int i = 0; i < num_blocks_; ++i) {
            // 最后一层用 identity，便于拟合 y=x；中间 relu
            const bool last = (i + 1 == num_blocks_);
            plain_layers_.push_back(std::make_shared<Layer>(
                dim, dim,
                std::make_shared<Activator>(last ? "identity" : "relu"), lr_));
        }
    } else {
        for (int i = 0; i < num_blocks_; ++i) {
            res_blocks_.push_back(std::make_shared<ResidualBlock>(dim, lr_));
        }
    }
}

void DeepMap::forward(const Matrix& x) {
    Matrix h = x;
    if (mode_ == Mode::Plain) {
        for (auto& layer : plain_layers_) {
            h = layer->forward(h);
        }
    } else {
        for (auto& block : res_blocks_) {
            h = block->forward(h);
        }
    }
    output_ = h;
}

void DeepMap::backward(const Matrix& y) {
    last_loss_ = mse_loss(output_, y);
    Matrix grad = mse_gradient(output_, y);
    if (mode_ == Mode::Plain) {
        for (int i = static_cast<int>(plain_layers_.size()) - 1; i >= 0; --i) {
            grad = plain_layers_[static_cast<size_t>(i)]->backward(grad);
        }
    } else {
        for (int i = static_cast<int>(res_blocks_.size()) - 1; i >= 0; --i) {
            grad = res_blocks_[static_cast<size_t>(i)]->backward(grad);
        }
    }
}

void DeepMap::train(const Matrix& x, const Matrix& y) {
    forward(x);
    backward(y);
}

Matrix DeepMap::predict(const Matrix& x) {
    forward(x);
    return output_;
}

float DeepMap::eval_mse(const Matrix& y) const {
    return mse_loss(output_, y);
}

int DeepMap::param_count() const {
    int n = 0;
    if (mode_ == Mode::Plain) {
        for (const auto& layer : plain_layers_) {
            n += layer->get_input_size() * layer->get_output_size() + layer->get_output_size();
        }
    } else {
        for (const auto& block : res_blocks_) {
            n += block->param_count();
        }
    }
    return n;
}

const char* DeepMap::mode_name() const {
    return mode_ == Mode::Plain ? "plain" : "residual";
}

void DeepMap::print_architecture() const {
    std::cout << "DeepMap [" << mode_name() << "] " << num_blocks_ << "×" << dim_
              << "→" << dim_ << ", lr=" << lr_ << ", params≈" << param_count() << "\n";
    if (mode_ == Mode::Residual) {
        std::cout << "  y = x + F(x) per block — identity is the default path\n";
    } else {
        std::cout << "  stacked Linear/ReLU — must warp identity through every layer\n";
    }
}
