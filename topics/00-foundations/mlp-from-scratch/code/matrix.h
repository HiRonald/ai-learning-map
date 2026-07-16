#pragma once

#include <vector>
#include <string>
#include <iostream>

// 行优先存储的二维矩阵，是整个网络数值计算的基础。
// 约定：样本按行存放，即 shape = (batch_size, feature_dim)。
class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double fill_value);
    Matrix(const Matrix& other);
    ~Matrix();

    Matrix& operator=(const Matrix& other);

    void print(const std::string& name = "") const;
    void set_data(int row, int col, double value);
    void set_data(const std::vector<double>& values);
    double get_data(int row, int col) const;
    int get_row_number() const;
    int get_col_number() const;

    // 按行切片 [start, end)，用于 mini-batch
    Matrix get_rows(int start_index, int end_index) const;
    Matrix get_cols(int start_index, int end_index) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix transpose() const;

    // 矩阵乘法 (m,k) @ (k,n) -> (m,n)
    Matrix dot(const Matrix& other) const;

    // 广播加法：常用于 bias，支持 (1,n)+(m,n) 与 (m,1)+(m,n)
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;

    // 逐元素运算
    Matrix mul(const Matrix& other) const;
    Matrix mul(double scalar) const;
    Matrix divide(const Matrix& other) const;
    Matrix div(const Matrix& other) const;  // divide 的别名，兼容旧调用

    Matrix log() const;
    double sum() const;

    // 按列求和，结果 shape = (1, cols)，用于 bias 梯度
    Matrix sum_rows() const;

    void fill(double value);
    void randomize(double min_value, double max_value);

private:
    int _row_number;
    int _col_number;
    std::vector<double> _data;

    void check_same_shape(const Matrix& other, const std::string& op) const;
};
