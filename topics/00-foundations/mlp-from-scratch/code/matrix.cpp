#include "matrix.h"

#include <stdexcept>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

Matrix::Matrix() : _row_number(0), _col_number(0) {}

Matrix::Matrix(int rows, int cols) : _row_number(rows), _col_number(cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    _data.assign(static_cast<size_t>(rows * cols), 0.0);
}

Matrix::Matrix(int rows, int cols, double fill_value)
    : _row_number(rows), _col_number(cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    _data.assign(static_cast<size_t>(rows * cols), fill_value);
}

Matrix::Matrix(const Matrix& other)
    : _row_number(other._row_number),
      _col_number(other._col_number),
      _data(other._data) {}

Matrix::~Matrix() = default;

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        _row_number = other._row_number;
        _col_number = other._col_number;
        _data = other._data;
    }
    return *this;
}

void Matrix::check_same_shape(const Matrix& other, const std::string& op) const {
    if (_row_number != other._row_number || _col_number != other._col_number) {
        throw std::invalid_argument(
            op + ": shape mismatch (" +
            std::to_string(_row_number) + "x" + std::to_string(_col_number) +
            " vs " +
            std::to_string(other._row_number) + "x" + std::to_string(other._col_number) + ")");
    }
}

void Matrix::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " (" << _row_number << "x" << _col_number << "):\n";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < _row_number; ++i) {
        std::cout << "  [";
        for (int j = 0; j < _col_number; ++j) {
            std::cout << get_data(i, j);
            if (j + 1 < _col_number) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
}

void Matrix::set_data(int row, int col, double value) {
    if (row < 0 || row >= _row_number || col < 0 || col >= _col_number) {
        throw std::out_of_range("Matrix::set_data index out of range");
    }
    _data[static_cast<size_t>(row * _col_number + col)] = value;
}

void Matrix::set_data(const std::vector<double>& values) {
    if (static_cast<int>(values.size()) != _row_number * _col_number) {
        throw std::invalid_argument("Matrix::set_data size does not match matrix shape");
    }
    _data = values;
}

double Matrix::get_data(int row, int col) const {
    if (row < 0 || row >= _row_number || col < 0 || col >= _col_number) {
        throw std::out_of_range("Matrix::get_data index out of range");
    }
    return _data[static_cast<size_t>(row * _col_number + col)];
}

int Matrix::get_row_number() const { return _row_number; }
int Matrix::get_col_number() const { return _col_number; }

Matrix Matrix::get_rows(int start_index, int end_index) const {
    if (start_index < 0 || end_index > _row_number || start_index >= end_index) {
        throw std::out_of_range("Matrix::get_rows invalid range");
    }
    Matrix result(end_index - start_index, _col_number);
    for (int i = start_index; i < end_index; ++i) {
        for (int j = 0; j < _col_number; ++j) {
            result.set_data(i - start_index, j, get_data(i, j));
        }
    }
    return result;
}

Matrix Matrix::get_cols(int start_index, int end_index) const {
    if (start_index < 0 || end_index > _col_number || start_index >= end_index) {
        throw std::out_of_range("Matrix::get_cols invalid range");
    }
    Matrix result(_row_number, end_index - start_index);
    for (int i = 0; i < _row_number; ++i) {
        for (int j = start_index; j < end_index; ++j) {
            result.set_data(i, j - start_index, get_data(i, j));
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const { return add(other); }
Matrix Matrix::operator-(const Matrix& other) const { return subtract(other); }

Matrix Matrix::operator*(double scalar) const { return mul(scalar); }

Matrix Matrix::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Matrix::operator/ division by zero");
    }
    return mul(1.0 / scalar);
}

Matrix Matrix::transpose() const {
    Matrix result(_col_number, _row_number);
    for (int i = 0; i < _row_number; ++i) {
        for (int j = 0; j < _col_number; ++j) {
            result.set_data(j, i, get_data(i, j));
        }
    }
    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    if (_col_number != other._row_number) {
        throw std::invalid_argument(
            "Matrix::dot shape mismatch: (" +
            std::to_string(_row_number) + "x" + std::to_string(_col_number) +
            ") @ (" +
            std::to_string(other._row_number) + "x" + std::to_string(other._col_number) + ")");
    }
    Matrix result(_row_number, other._col_number);
    for (int i = 0; i < _row_number; ++i) {
        for (int k = 0; k < _col_number; ++k) {
            double aik = get_data(i, k);
            for (int j = 0; j < other._col_number; ++j) {
                result.set_data(i, j, result.get_data(i, j) + aik * other.get_data(k, j));
            }
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    // 同形状直接加
    if (_row_number == other._row_number && _col_number == other._col_number) {
        Matrix result(_row_number, _col_number);
        for (size_t i = 0; i < _data.size(); ++i) {
            result._data[i] = _data[i] + other._data[i];
        }
        return result;
    }

    // 广播：(m,n) + (1,n)
    if (other._row_number == 1 && other._col_number == _col_number) {
        Matrix result(_row_number, _col_number);
        for (int i = 0; i < _row_number; ++i) {
            for (int j = 0; j < _col_number; ++j) {
                result.set_data(i, j, get_data(i, j) + other.get_data(0, j));
            }
        }
        return result;
    }

    // 广播：(m,n) + (m,1)
    if (other._row_number == _row_number && other._col_number == 1) {
        Matrix result(_row_number, _col_number);
        for (int i = 0; i < _row_number; ++i) {
            for (int j = 0; j < _col_number; ++j) {
                result.set_data(i, j, get_data(i, j) + other.get_data(i, 0));
            }
        }
        return result;
    }

    throw std::invalid_argument("Matrix::add unsupported broadcast shapes");
}

Matrix Matrix::subtract(const Matrix& other) const {
    check_same_shape(other, "Matrix::subtract");
    Matrix result(_row_number, _col_number);
    for (size_t i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] - other._data[i];
    }
    return result;
}

Matrix Matrix::mul(const Matrix& other) const {
    check_same_shape(other, "Matrix::mul");
    Matrix result(_row_number, _col_number);
    for (size_t i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] * other._data[i];
    }
    return result;
}

Matrix Matrix::mul(double scalar) const {
    Matrix result(_row_number, _col_number);
    for (size_t i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] * scalar;
    }
    return result;
}

Matrix Matrix::divide(const Matrix& other) const {
    check_same_shape(other, "Matrix::divide");
    Matrix result(_row_number, _col_number);
    for (size_t i = 0; i < _data.size(); ++i) {
        if (std::abs(other._data[i]) < 1e-12) {
            throw std::invalid_argument("Matrix::divide division by zero");
        }
        result._data[i] = _data[i] / other._data[i];
    }
    return result;
}

Matrix Matrix::div(const Matrix& other) const { return divide(other); }

Matrix Matrix::log() const {
    Matrix result(_row_number, _col_number);
    for (size_t i = 0; i < _data.size(); ++i) {
        double v = std::max(_data[i], 1e-12);  // 数值稳定，避免 log(0)
        result._data[i] = std::log(v);
    }
    return result;
}

double Matrix::sum() const {
    double total = 0.0;
    for (double v : _data) {
        total += v;
    }
    return total;
}

Matrix Matrix::sum_rows() const {
    Matrix result(1, _col_number);
    for (int j = 0; j < _col_number; ++j) {
        double col_sum = 0.0;
        for (int i = 0; i < _row_number; ++i) {
            col_sum += get_data(i, j);
        }
        result.set_data(0, j, col_sum);
    }
    return result;
}

void Matrix::fill(double value) {
    std::fill(_data.begin(), _data.end(), value);
}

void Matrix::randomize(double min_value, double max_value) {
    static std::mt19937 rng(42);  // 固定种子，方便复现实验
    std::uniform_real_distribution<double> dist(min_value, max_value);
    for (double& v : _data) {
        v = dist(rng);
    }
}
