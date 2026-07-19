#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>

#include "nn/matrix.h"
#include "nn/layer.h"
#include "nn/activator.h"

// 多层感知机：按顺序堆叠若干全连接层。
// 隐藏层与输出层都是 Layer，区别主要在激活函数选择（输出层可单独配置）。
class Mlp {
public:
    Mlp();
    Mlp(const std::vector<std::shared_ptr<Layer>>& layers, const std::string& loss_type);
    ~Mlp();

    void set_layers(const std::vector<std::shared_ptr<Layer>>& layers);
    void set_learning_rate(double learning_rate);

    void train(const Matrix& data, const Matrix& labels);
    Matrix predict(const Matrix& data);
    float eval(const Matrix& label) const;

    void forward(const Matrix& input);
    void backward(const Matrix& label);

    void print_architecture(bool verbose = true) const;
    const Matrix& get_output() const;

private:
    std::vector<std::shared_ptr<Layer>> _layers;
    std::string _loss_type;
    std::function<float(const Matrix& output, const Matrix& label)> _loss_function;
    std::function<Matrix(const Matrix& output, const Matrix& label)> _loss_gradient_function;
    Matrix _output;
    float _loss;

    void setup_loss(const std::string& loss_type);
};
