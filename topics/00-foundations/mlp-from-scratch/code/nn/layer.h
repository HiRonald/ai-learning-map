#pragma once

#include "nn/matrix.h"
#include "nn/activator.h"

#include <memory>
#include <string>

// 全连接层：y = activator(x @ W + b)
// 权重形状 (in, out)，输入形状 (batch, in)，输出形状 (batch, out)
class Layer {
public:
    Layer(int input_size, int output_size);
    Layer(int input_size, int output_size, std::shared_ptr<Activator> activator);
    Layer(int input_size, int output_size, std::shared_ptr<Activator> activator, double learning_rate);
    ~Layer();

    void set_input_size(int input_size);
    void set_output_size(int output_size);
    void set_weights(const Matrix& weights);
    void set_biases(const Matrix& biases);
    void set_activator(std::shared_ptr<Activator> activator);
    void set_learning_rate(double learning_rate);

    // 前向：线性变换 + 激活
    Matrix forward(const Matrix& input);

    // 反向：根据上游梯度更新参数，并返回对本层输入的梯度
    Matrix backward(const Matrix& upstream_gradient);

    // verbose=true 时额外打印 W/b（小网络调试用；大网络建议 false）
    void print(const std::string& name = "", bool verbose = true) const;

    int get_input_size() const;
    int get_output_size() const;
    const Matrix& get_weights() const;
    const Matrix& get_biases() const;

private:
    int _input_size;
    int _output_size;
    double _learning_rate;
    Matrix _weights;   // (in, out)
    Matrix _biases;    // (1, out)
    Matrix _z;         // 线性变换结果 (batch, out)
    Matrix _input;     // 缓存输入，反向传播用
    Matrix _output;    // 激活后输出
    std::shared_ptr<Activator> _activator;

    Matrix optimize(const Matrix& param, const Matrix& gradient) const;
};
