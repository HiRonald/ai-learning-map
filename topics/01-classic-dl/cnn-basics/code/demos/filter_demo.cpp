#include "demos/filter_demo.h"

#include "cnn/ops.h"
#include "cnn/tensor.h"
#include "demos/common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

void print_ascii_plane(const Tensor& t, int n, const string& title) {
    static const char* ramp = " .:-=+*#%@";
    constexpr int ramp_n = 10;
    // 归一化到 [0,1] 方便看结构
    double lo = t.at(n, 0, 0, 0);
    double hi = lo;
    for (int h = 0; h < t.h(); ++h) {
        for (int w = 0; w < t.w(); ++w) {
            double v = t.at(n, 0, h, w);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    const double span = max(1e-9, hi - lo);

    cout << title << "  (range [" << lo << ", " << hi << "])\n";
    for (int h = 0; h < t.h(); ++h) {
        cout << "  ";
        for (int w = 0; w < t.w(); ++w) {
            double v = (t.at(n, 0, h, w) - lo) / span;
            int idx = static_cast<int>(round(v * (ramp_n - 1)));
            idx = max(0, min(ramp_n - 1, idx));
            cout << ramp[idx] << ramp[idx];
        }
        cout << "\n";
    }
}

Tensor make_cross_image(int side) {
    Tensor img(1, 1, side, side, 0.1);
    const int mid = side / 2;
    for (int i = 0; i < side; ++i) {
        img.at(0, 0, mid, i) = 1.0;
        img.at(0, 0, i, mid) = 1.0;
    }
    // 加一个小方块，方便看出平移同模式
    for (int h = 3; h <= 6; ++h) {
        for (int w = 3; w <= 6; ++w) {
            img.at(0, 0, h, w) = 0.85;
        }
    }
    return img;
}

void print_kernel(const vector<double>& k, int side, const string& name) {
    cout << name << " (" << side << "x" << side << "):\n";
    for (int u = 0; u < side; ++u) {
        cout << "  ";
        for (int v = 0; v < side; ++v) {
            cout << k[static_cast<size_t>(u * side + v)];
            if (v + 1 < side) {
                cout << " ";
            }
        }
        cout << "\n";
    }
}

}  // namespace

namespace demos {

int run_filter() {
    print_separator("Demo: filter");
    cout << "Goal: see convolution as a small shared kernel sliding over an image.\n"
         << "No training — fixed Sobel / sharpen kernels on a synthetic 16x16 canvas.\n";

    Tensor img = make_cross_image(16);
    print_ascii_plane(img, 0, "\nInput (cross + square)");

    // Sobel-X：竖直边缘响应强
    const vector<double> sobel_x = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1,
    };
    // Sobel-Y：水平边缘
    const vector<double> sobel_y = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1,
    };
    // 锐化
    const vector<double> sharpen = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0,
    };

    print_separator("Kernels");
    print_kernel(sobel_x, 3, "Sobel-X");
    print_kernel(sobel_y, 3, "Sobel-Y");
    print_kernel(sharpen, 3, "Sharpen");

    Tensor sx = cnn_ops::conv2d_fixed(img, sobel_x, 3);
    Tensor sy = cnn_ops::conv2d_fixed(img, sobel_y, 3);
    Tensor sh = cnn_ops::conv2d_fixed(img, sharpen, 3);

    // 边缘强度：sqrt(sx^2 + sy^2)
    Tensor edge(1, 1, img.h(), img.w());
    for (int h = 0; h < img.h(); ++h) {
        for (int w = 0; w < img.w(); ++w) {
            const double gx = sx.at(0, 0, h, w);
            const double gy = sy.at(0, 0, h, w);
            edge.at(0, 0, h, w) = std::sqrt(gx * gx + gy * gy);
        }
    }

    print_separator("Responses");
    print_ascii_plane(sx, 0, "Sobel-X response");
    print_ascii_plane(sy, 0, "Sobel-Y response");
    print_ascii_plane(edge, 0, "Edge magnitude sqrt(sx^2+sy^2)");
    print_ascii_plane(sh, 0, "Sharpen response");

    cout << "\nTakeaway: one 3x3 kernel is reused at every location (weight sharing).\n"
         << "Vertical edges light up under Sobel-X; horizontal under Sobel-Y.\n"
         << "Next: `param` — count how many weights that sharing saves vs a dense layer.\n";
    return 0;
}

}  // namespace demos
