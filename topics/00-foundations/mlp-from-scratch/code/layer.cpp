#include "layer.h"

#include <iostream>
#include <stdexcept>
#include <cmath>

Layer::Layer(int input_size, int output_size)
    : Layer(input_size, output_size, std::make_shared<Activator>("sigmoid"), 0.5) {}

Layer::Layer(int input_size, int output_size, std::shared_ptr<Activator> activator)
    : Layer(input_size, output_size, activator, 0.5) {}

Layer::Layer(int input_size, int output_size, std::shared_ptr<Activator> activator, double learning_rate)
    : _input_size(input_size),
      _output_size(output_size),
      _learning_rate(learning_rate),
      _weights(input_size, output_size),
      _biases(1, output_size),
      _activator(std::move(activator)) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Layer sizes must be positive");
    }
    if (!_activator) {
        throw std::invalid_argument("Layer activator must not be null");
    }
    // Xavier/Glorot 风格的均匀初始化，有助于 sigmoid/tanh 训练稳定
    double limit = std::sqrt(6.0 / (input_size + output_size));
    _weights.randomize(-limit, limit);
    _biases.fill(0.0);
}

Layer::~Layer() = default;

void Layer::set_input_size(int input_size) { _input_size = input_size; }
void Layer::set_output_size(int output_size) { _output_size = output_size; }

void Layer::set_weights(const Matrix& weights) {
    if (weights.get_row_number() != _input_size || weights.get_col_number() != _output_size) {
        throw std::invalid_argument("Layer::set_weights shape mismatch");
    }
    _weights = weights;
}

void Layer::set_biases(const Matrix& biases) {
    if (biases.get_row_number() != 1 || biases.get_col_number() != _output_size) {
        throw std::invalid_argument("Layer::set_biases shape mismatch, expect (1, out)");
    }
    _biases = biases;
}

void Layer::set_activator(std::shared_ptr<Activator> activator) {
    if (!activator) {
        throw std::invalid_argument("Layer activator must not be null");
    }
    _activator = std::move(activator);
}

void Layer::set_learning_rate(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    _learning_rate = learning_rate;
}

Matrix Layer::forward(const Matrix& input) {
    // input: (batch, in)
    if (input.get_col_number() != _input_size) {
        throw std::invalid_argument("Layer::forward input feature size mismatch");
    }
    _input = input;
    // z = xW + b
    _z = _input.dot(_weights).add(_biases);
    _output = _activator->forward(_z);
    return _output;
}

Matrix Layer::backward(const Matrix& upstream_gradient) {
    // upstream_gradient = dL/da，形状 (batch, out)
    if (upstream_gradient.get_row_number() != _output.get_row_number() ||
        upstream_gradient.get_col_number() != _output_size) {
        throw std::invalid_argument("Layer::backward gradient shape mismatch");
    }

    // dL/dz = dL/da ⊙ σ'(z)
    Matrix grad_z = upstream_gradient.mul(_activator->backward(_z));

    // dL/dW = x^T @ dL/dz
    Matrix grad_weights = _input.transpose().dot(grad_z);
    // dL/db = sum over batch
    Matrix grad_biases = grad_z.sum_rows();

    // 必须用更新前的 W 计算对输入的梯度，否则链式法则会串味
    Matrix grad_input = grad_z.dot(_weights.transpose());

    // 参数更新：θ <- θ - lr * ∇θ
    _weights = optimize(_weights, grad_weights);
    _biases = optimize(_biases, grad_biases);

    return grad_input;
}

Matrix Layer::optimize(const Matrix& param, const Matrix& gradient) const {
    return param - gradient.mul(_learning_rate);
}

void Layer::print(const std::string& name) const {
    std::string prefix = name.empty() ? "Layer" : name;
    std::cout << prefix << " [" << _input_size << " -> " << _output_size
              << ", activator=" << _activator->get_name()
              << ", lr=" << _learning_rate << "]\n";
    _weights.print(prefix + ".W");
    _biases.print(prefix + ".b");
}

int Layer::get_input_size() const { return _input_size; }
int Layer::get_output_size() const { return _output_size; }
const Matrix& Layer::get_weights() const { return _weights; }
const Matrix& Layer::get_biases() const { return _biases; }
