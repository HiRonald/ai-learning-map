#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

// MSE: L = 0.5 * mean((y_hat - y)^2)
float mse_loss(const Matrix& output, const Matrix& label) {
    Matrix diff = output - label;
    return static_cast<float>(0.5 * diff.mul(diff).sum() / output.get_row_number());
}

// dL/dy_hat = (y_hat - y) / batch
Matrix mse_gradient(const Matrix& output, const Matrix& label) {
    return (output - label) / static_cast<double>(output.get_row_number());
}

// 二分类交叉熵（输出需已是概率）：L = -mean(y*log(p) + (1-y)*log(1-p))
float bce_loss(const Matrix& output, const Matrix& label) {
    double total = 0.0;
    const int rows = output.get_row_number();
    const int cols = output.get_col_number();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double y = label.get_data(i, j);
            double p = std::max(1e-12, std::min(1.0 - 1e-12, output.get_data(i, j)));
            total += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }
    }
    return static_cast<float>(total / rows);
}

Matrix bce_gradient(const Matrix& output, const Matrix& label) {
    Matrix grad(output.get_row_number(), output.get_col_number());
    const int rows = output.get_row_number();
    const int cols = output.get_col_number();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double y = label.get_data(i, j);
            double p = std::max(1e-12, std::min(1.0 - 1e-12, output.get_data(i, j)));
            // dL/dp = (-y/p + (1-y)/(1-p)) / batch
            grad.set_data(i, j, (-y / p + (1.0 - y) / (1.0 - p)) / rows);
        }
    }
    return grad;
}

}  // namespace

Mlp::Mlp() : _loss_type(""), _loss(0.0f) {}

Mlp::Mlp(const std::vector<std::shared_ptr<Layer>>& layers, const std::string& loss_type)
    : _layers(layers), _loss(0.0f) {
    if (_layers.empty()) {
        throw std::invalid_argument("Mlp must contain at least one layer");
    }
    setup_loss(loss_type);
}

Mlp::~Mlp() {
    _layers.clear();
}

void Mlp::setup_loss(const std::string& loss_type) {
    _loss_type = loss_type;
    if (loss_type == "mse") {
        _loss_function = mse_loss;
        _loss_gradient_function = mse_gradient;
    } else if (loss_type == "cross_entropy" || loss_type == "bce") {
        _loss_function = bce_loss;
        _loss_gradient_function = bce_gradient;
    } else {
        throw std::invalid_argument("Invalid loss type: " + loss_type);
    }
}

void Mlp::set_layers(const std::vector<std::shared_ptr<Layer>>& layers) {
    if (layers.empty()) {
        throw std::invalid_argument("Mlp must contain at least one layer");
    }
    _layers = layers;
}

void Mlp::set_learning_rate(double learning_rate) {
    for (auto& layer : _layers) {
        layer->set_learning_rate(learning_rate);
    }
}

void Mlp::train(const Matrix& data, const Matrix& labels) {
    forward(data);
    backward(labels);
}

Matrix Mlp::predict(const Matrix& input) {
    forward(input);
    return _output;
}

float Mlp::eval(const Matrix& label) const {
    return _loss_function(_output, label);
}

void Mlp::forward(const Matrix& input) {
    Matrix output = input;
    for (size_t i = 0; i < _layers.size(); ++i) {
        output = _layers[i]->forward(output);
    }
    _output = output;
}

void Mlp::backward(const Matrix& label) {
    if (label.get_row_number() != _output.get_row_number() ||
        label.get_col_number() != _output.get_col_number()) {
        throw std::invalid_argument("Mlp::backward label shape mismatch");
    }

    // 记录标量 loss（仅用于监控；反向传播只依赖梯度）
    _loss = _loss_function(_output, label);

    // 从损失函数开始，把梯度从输出层一路传回输入层
    Matrix gradient = _loss_gradient_function(_output, label);
    for (int i = static_cast<int>(_layers.size()) - 1; i >= 0; --i) {
        gradient = _layers[static_cast<size_t>(i)]->backward(gradient);
    }
}

void Mlp::print_architecture() const {
    std::cout << "MLP architecture (" << _layers.size() << " layers, loss=" << _loss_type << "):\n";
    for (size_t i = 0; i < _layers.size(); ++i) {
        bool is_output = (i + 1 == _layers.size());
        std::string name = is_output ? ("OutputLayer" + std::to_string(i))
                                     : ("HiddenLayer" + std::to_string(i));
        _layers[i]->print("  " + name);
    }
}

const Matrix& Mlp::get_output() const { return _output; }
