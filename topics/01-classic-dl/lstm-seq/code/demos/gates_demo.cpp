#include "demos/gates_demo.h"

#include "demos/common.h"
#include "lstm/cell.h"
#include "nn/matrix.h"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

namespace demos {

int run_gates() {
    print_separator("Demo: gates — cell state through time");
    cout << "Fixed tiny LSTM (H=1), no training.\n"
         << "Watch f / i / o and c_t: forget stays open, a write early, then hold.\n";

    constexpr int input = 1;
    constexpr int hidden = 1;
    LstmCell cell(input, hidden, /*lr=*/0.01);  // lr unused

    // 列顺序 [f | i | g | o]
    // f: 偏置很大 → 几乎不忘
    // i: x 大时打开；g: 跟着 x 写；o: 常开方便看 h
    Matrix w_x(input, 4 * hidden);
    w_x.set_data({0.0, 8.0, 2.0, 0.0});
    Matrix w_h(hidden, 4 * hidden, 0.0);
    Matrix b(1, 4 * hidden);
    b.set_data({5.0, -4.0, 0.0, 5.0});
    cell.set_weights_x(w_x);
    cell.set_weights_h(w_h);
    cell.set_bias(b);

    const vector<double> xs_vals = {0.8, 0.0, 0.0, 0.0, 0.0};
    vector<Matrix> xs;
    xs.reserve(xs_vals.size());
    for (double v : xs_vals) {
        Matrix x(1, 1);
        x.set_data(0, 0, v);
        xs.push_back(x);
    }

    cout << "\nsequence x = [";
    for (size_t i = 0; i < xs_vals.size(); ++i) {
        cout << xs_vals[i] << (i + 1 < xs_vals.size() ? ", " : "");
    }
    cout << "]\n";
    cout << "c_0 = 0, h_0 = 0\n\n";

    cell.forward(xs);
    const auto& fs = cell.forget_gates();
    const auto& is = cell.input_gates();
    const auto& os = cell.output_gates();
    const auto& cs = cell.cell_states();
    const auto& hs = cell.hidden_states();

    cout << fixed << setprecision(3);
    cout << left << setw(4) << "t" << setw(8) << "x_t" << setw(8) << "f"
         << setw(8) << "i" << setw(8) << "o" << setw(8) << "c_t" << "h_t\n";
    for (size_t t = 0; t < hs.size(); ++t) {
        cout << left << setw(4) << (t + 1) << setw(8) << xs_vals[t]
             << setw(8) << fs[t].get_data(0, 0) << setw(8) << is[t].get_data(0, 0)
             << setw(8) << os[t].get_data(0, 0) << setw(8) << cs[t].get_data(0, 0)
             << hs[t].get_data(0, 0) << "\n";
    }

    print_separator("Takeaway");
    cout << "c_t is the slow lane: write once (i open), then f≈1 keeps it across zeros.\n"
         << "Vanilla RNN only has h_t — no separate highway. Next: `recall` trains this.\n";
    return 0;
}

}  // namespace demos
