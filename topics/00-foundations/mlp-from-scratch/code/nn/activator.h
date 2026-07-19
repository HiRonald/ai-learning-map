#pragma once

#include "nn/matrix.h"

#include <string>
#include <functional>
#include <memory>

// 激活函数封装：同时提供前向 f(z) 与导数 f'(z)。
// 导数一律对预激活值 z 求，避免和激活输出 a 混淆。
class Activator {
public:
    Activator();
    explicit Activator(const std::string& name);
    ~Activator();

    void set_name(const std::string& name);
    std::string get_name() const;

    void set_activation_function(std::function<Matrix(const Matrix&)> activation_function);
    void set_derivative_function(std::function<Matrix(const Matrix&)> derivative_function);

    Matrix forward(const Matrix& input) const;
    Matrix backward(const Matrix& input) const;  // 输入为 z，返回 f'(z)
    void print() const;

private:
    std::string _name;
    std::function<Matrix(const Matrix&)> _activation_function;
    std::function<Matrix(const Matrix&)> _derivative_function;
};
