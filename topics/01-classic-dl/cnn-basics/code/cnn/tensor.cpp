#include "cnn/tensor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

Tensor::Tensor() = default;

Tensor::Tensor(int n, int c, int h, int w) : Tensor(n, c, h, w, 0.0) {}

Tensor::Tensor(int n, int c, int h, int w, double fill)
    : n_(n), c_(c), h_(h), w_(w), data_(static_cast<size_t>(n) * c * h * w, fill) {
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        throw std::invalid_argument("Tensor dims must be positive");
    }
}

int Tensor::n() const { return n_; }
int Tensor::c() const { return c_; }
int Tensor::h() const { return h_; }
int Tensor::w() const { return w_; }
int Tensor::size() const { return n_ * c_ * h_ * w_; }

int Tensor::index(int ni, int ci, int hi, int wi) const {
    return ((ni * c_ + ci) * h_ + hi) * w_ + wi;
}

void Tensor::check_bounds(int ni, int ci, int hi, int wi) const {
    if (ni < 0 || ni >= n_ || ci < 0 || ci >= c_ || hi < 0 || hi >= h_ || wi < 0 || wi >= w_) {
        throw std::out_of_range("Tensor index out of range");
    }
}

double& Tensor::at(int ni, int ci, int hi, int wi) {
    check_bounds(ni, ci, hi, wi);
    return data_[static_cast<size_t>(index(ni, ci, hi, wi))];
}

double Tensor::at(int ni, int ci, int hi, int wi) const {
    check_bounds(ni, ci, hi, wi);
    return data_[static_cast<size_t>(index(ni, ci, hi, wi))];
}

const std::vector<double>& Tensor::data() const { return data_; }
std::vector<double>& Tensor::data() { return data_; }

void Tensor::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::randomize(double min_value, double max_value) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(min_value, max_value);
    for (double& v : data_) {
        v = dist(rng);
    }
}

void Tensor::print_shape(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "Tensor(" << n_ << ", " << c_ << ", " << h_ << ", " << w_ << ")\n";
}
