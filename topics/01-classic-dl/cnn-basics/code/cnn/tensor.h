#pragma once

#include <string>
#include <vector>

// NCHW 四维张量，行优先平坦存储。
// index(n,c,h,w) = ((n * C + c) * H + h) * W + w
class Tensor {
public:
    Tensor();
    Tensor(int n, int c, int h, int w);
    Tensor(int n, int c, int h, int w, double fill);

    int n() const;
    int c() const;
    int h() const;
    int w() const;
    int size() const;

    double& at(int ni, int ci, int hi, int wi);
    double at(int ni, int ci, int hi, int wi) const;

    const std::vector<double>& data() const;
    std::vector<double>& data();

    void fill(double value);
    void randomize(double min_value, double max_value);

    void print_shape(const std::string& name = "") const;

private:
    int n_ = 0;
    int c_ = 0;
    int h_ = 0;
    int w_ = 0;
    std::vector<double> data_;

    int index(int ni, int ci, int hi, int wi) const;
    void check_bounds(int ni, int ci, int hi, int wi) const;
};
