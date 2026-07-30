#include "demos/unroll_demo.h"

#include "demos/common.h"
#include "nn/matrix.h"
#include "rnn/cell.h"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

namespace demos {

int run_unroll() {
    print_separator("Demo: unroll — state along time");
    cout << "Fixed tiny RNN, no training.\n"
         << "Watch h_t absorb each new bit: same weights, growing memory.\n";

    constexpr int input = 1;
    constexpr int hidden = 3;
    RnnCell cell(input, hidden, /*lr=*/0.01);  // lr unused

    // 手设权重，让状态变化可读
    Matrix w_xh(input, hidden);
    w_xh.set_data({1.2, -0.8, 0.5});
    Matrix w_hh(hidden, hidden);
    // 轻度自连接，避免一步洗掉记忆
    w_hh.set_data({
        0.6, 0.1, 0.0,
        0.0, 0.6, 0.1,
        0.1, 0.0, 0.6,
    });
    Matrix b(1, hidden);
    b.set_data({0.0, 0.0, 0.0});
    cell.set_weights_xh(w_xh);
    cell.set_weights_hh(w_hh);
    cell.set_bias(b);

    // 序列：1, 0, 1, 1, 0
    const vector<double> bits = {1, 0, 1, 1, 0};
    vector<Matrix> xs;
    xs.reserve(bits.size());
    for (double bit : bits) {
        Matrix x(1, 1);
        x.set_data(0, 0, bit);
        xs.push_back(x);
    }

    cout << "\nsequence x = [";
    for (size_t i = 0; i < bits.size(); ++i) {
        cout << bits[i] << (i + 1 < bits.size() ? ", " : "");
    }
    cout << "]\n";
    cout << "h_0 = [0, 0, 0]\n\n";

    cell.forward(xs);
    const auto& hs = cell.hidden_states();

    cout << fixed << setprecision(4);
    cout << left << setw(4) << "t" << setw(8) << "x_t"
         << "h_t\n";
    for (size_t t = 0; t < hs.size(); ++t) {
        cout << left << setw(4) << (t + 1) << setw(8) << bits[t] << "[";
        for (int j = 0; j < hidden; ++j) {
            cout << hs[t].get_data(0, j);
            if (j + 1 < hidden) {
                cout << ", ";
            }
        }
        cout << "]\n";
    }

    print_separator("Takeaway");
    cout << "Same (W_xh, W_hh, b) at every step — that is weight sharing in time.\n"
         << "h_t is a running summary of x_1..x_t. Next: `temp` trains this on daily temperatures.\n";
    return 0;
}

}  // namespace demos
