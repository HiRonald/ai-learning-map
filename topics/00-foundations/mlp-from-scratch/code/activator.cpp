#include "activator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

Matrix relu_forward(const Matrix& input) {
    Matrix output(input.get_row_number(), input.get_col_number());
    for (int i = 0; i < input.get_row_number(); ++i) {
        for (int j = 0; j < input.get_col_number(); ++j) {
            output.set_data(i, j, std::max(0.0, input.get_data(i, j)));
        }
    }
    return output;
}

Matrix relu_backward(const Matrix& z) {
    Matrix output(z.get_row_number(), z.get_col_number());
    for (int i = 0; i < z.get_row_number(); ++i) {
        for (int j = 0; j < z.get_col_number(); ++j) {
            output.set_data(i, j, z.get_data(i, j) > 0.0 ? 1.0 : 0.0);
        }
    }
    return output;
}

Matrix sigmoid_forward(const Matrix& input) {
    Matrix output(input.get_row_number(), input.get_col_number());
    for (int i = 0; i < input.get_row_number(); ++i) {
        for (int j = 0; j < input.get_col_number(); ++j) {
            // 截断避免 exp 溢出
            double x = std::max(-50.0, std::min(50.0, input.get_data(i, j)));
            output.set_data(i, j, 1.0 / (1.0 + std::exp(-x)));
        }
    }
    return output;
}

// sigmoid'(z) = s(z) * (1 - s(z))，这里从 z 重新计算 s，而不是误用 z*(1-z)
Matrix sigmoid_backward(const Matrix& z) {
    Matrix s = sigmoid_forward(z);
    Matrix output(z.get_row_number(), z.get_col_number());
    for (int i = 0; i < z.get_row_number(); ++i) {
        for (int j = 0; j < z.get_col_number(); ++j) {
            double v = s.get_data(i, j);
            output.set_data(i, j, v * (1.0 - v));
        }
    }
    return output;
}

Matrix tanh_forward(const Matrix& input) {
    Matrix output(input.get_row_number(), input.get_col_number());
    for (int i = 0; i < input.get_row_number(); ++i) {
        for (int j = 0; j < input.get_col_number(); ++j) {
            output.set_data(i, j, std::tanh(input.get_data(i, j)));
        }
    }
    return output;
}

Matrix tanh_backward(const Matrix& z) {
    Matrix output(z.get_row_number(), z.get_col_number());
    for (int i = 0; i < z.get_row_number(); ++i) {
        for (int j = 0; j < z.get_col_number(); ++j) {
            double t = std::tanh(z.get_data(i, j));
            output.set_data(i, j, 1.0 - t * t);
        }
    }
    return output;
}

// 恒等激活：常用于回归输出层
Matrix identity_forward(const Matrix& input) { return input; }

Matrix identity_backward(const Matrix& z) {
    return Matrix(z.get_row_number(), z.get_col_number(), 1.0);
}

}  // namespace

Activator::Activator() : _name(""), _activation_function(nullptr), _derivative_function(nullptr) {}

Activator::Activator(const std::string& name) {
    set_name(name);
    if (name == "relu") {
        _activation_function = relu_forward;
        _derivative_function = relu_backward;
    } else if (name == "sigmoid") {
        _activation_function = sigmoid_forward;
        _derivative_function = sigmoid_backward;
    } else if (name == "tanh") {
        _activation_function = tanh_forward;
        _derivative_function = tanh_backward;
    } else if (name == "identity" || name == "linear") {
        _activation_function = identity_forward;
        _derivative_function = identity_backward;
    } else {
        throw std::invalid_argument("Invalid activator name: " + name);
    }
}

Activator::~Activator() = default;

void Activator::set_name(const std::string& name) { _name = name; }

std::string Activator::get_name() const { return _name; }

void Activator::set_activation_function(std::function<Matrix(const Matrix&)> activation_function) {
    _activation_function = std::move(activation_function);
}

void Activator::set_derivative_function(std::function<Matrix(const Matrix&)> derivative_function) {
    _derivative_function = std::move(derivative_function);
}

Matrix Activator::forward(const Matrix& input) const {
    if (!_activation_function) {
        throw std::runtime_error("Activator forward function is not set");
    }
    return _activation_function(input);
}

Matrix Activator::backward(const Matrix& input) const {
    if (!_derivative_function) {
        throw std::runtime_error("Activator derivative function is not set");
    }
    return _derivative_function(input);
}

void Activator::print() const {
    std::cout << "Activator: " << _name << "\n";
}
