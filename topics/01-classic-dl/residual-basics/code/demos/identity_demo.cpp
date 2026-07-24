#include "demos/identity_demo.h"

#include "demos/common.h"
#include "residual/map_net.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using namespace std;

namespace {

Matrix make_identity_data(int n, int dim, unsigned seed) {
    mt19937 rng(seed);
    normal_distribution<double> dist(0.0, 1.0);
    Matrix x(n, dim);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            x.set_data(i, j, dist(rng));
        }
    }
    return x;  // y = x
}

double mean_abs_error(const Matrix& pred, const Matrix& y) {
    double s = 0.0;
    const int n = pred.get_row_number() * pred.get_col_number();
    for (int i = 0; i < pred.get_row_number(); ++i) {
        for (int j = 0; j < pred.get_col_number(); ++j) {
            s += abs(pred.get_data(i, j) - y.get_data(i, j));
        }
    }
    return s / n;
}

struct RunResult {
    double final_mse = 0.0;
    double final_mae = 0.0;
};

RunResult train_map(DeepMap& net, const Matrix& train_x, const Matrix& train_y,
                    const Matrix& test_x, const Matrix& test_y, int epochs, int batch,
                    int log_every) {
    const int n = train_x.get_row_number();
    cout << fixed << setprecision(6);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < n; start += batch) {
            const int end = min(start + batch, n);
            net.train(train_x.get_rows(start, end), train_y.get_rows(start, end));
        }
        if (epoch % log_every == 0 || epoch + 1 == epochs) {
            Matrix pred = net.predict(test_x);
            float mse = net.eval_mse(test_y);
            double mae = mean_abs_error(pred, test_y);
            cout << "  epoch " << setw(2) << epoch
                 << " | test_mse=" << mse
                 << " | test_mae=" << mae << "\n";
        }
    }
    Matrix pred = net.predict(test_x);
    RunResult r;
    r.final_mse = net.eval_mse(test_y);
    r.final_mae = mean_abs_error(pred, test_y);
    return r;
}

}  // namespace

namespace demos {

int run_identity() {
    print_separator("Demo: identity through a deep net");
    cout << "Task: learn y = x in R^d with a deep stack.\n"
         << "Point: residual default path is identity — plain must warp through every layer.\n"
         << "Fairness: match # of Linear layers (plain uses 2× stages).\n";

    constexpr int dim = 32;
    constexpr int n_blocks = 8;          // residual blocks
    constexpr int plain_layers = 16;     // 2 linears per residual block
    constexpr int train_n = 512;
    constexpr int test_n = 128;
    constexpr int epochs = 60;
    constexpr int batch = 64;
    constexpr double lr = 0.02;

    Matrix train_x = make_identity_data(train_n, dim, 7);
    Matrix test_x = make_identity_data(test_n, dim, 8);
    // y = x
    Matrix train_y = train_x;
    Matrix test_y = test_x;

    cout << "dim=" << dim << ", train=" << train_n << ", test=" << test_n
         << ", epochs=" << epochs << ", lr=" << lr << "\n";

    print_separator("Plain stack");
    DeepMap plain(dim, plain_layers, DeepMap::Mode::Plain, lr);
    plain.print_architecture();
    RunResult plain_r =
        train_map(plain, train_x, train_y, test_x, test_y, epochs, batch, /*log_every=*/15);

    print_separator("Residual blocks");
    DeepMap res(dim, n_blocks, DeepMap::Mode::Residual, lr);
    res.print_architecture();
    RunResult res_r =
        train_map(res, train_x, train_y, test_x, test_y, epochs, batch, /*log_every=*/15);

    print_separator("Head-to-head");
    cout << fixed << setprecision(6);
    cout << left << setw(10) << "model" << right
         << setw(10) << "params"
         << setw(14) << "test_mse"
         << setw(14) << "test_mae" << "\n";
    cout << left << setw(10) << "plain" << right
         << setw(10) << plain.param_count()
         << setw(14) << plain_r.final_mse
         << setw(14) << plain_r.final_mae << "\n";
    cout << left << setw(10) << "residual" << right
         << setw(10) << res.param_count()
         << setw(14) << res_r.final_mse
         << setw(14) << res_r.final_mae << "\n";

    cout << "\nTakeaway: residual can stay near identity (F≈0); plain must re-learn it\n"
         << "through every layer — degradation shows up as higher error at the same depth.\n";

    // residual 应明显更好（允许一点点数值噪声）
    if (res_r.final_mse < plain_r.final_mse * 0.5 || res_r.final_mse < 0.05) {
        cout << "\nOK: residual beat plain on identity mapping.\n";
        return 0;
    }
    cout << "\nUnexpected: residual did not clearly win. Try more epochs or depth.\n";
    return 1;
}

}  // namespace demos
