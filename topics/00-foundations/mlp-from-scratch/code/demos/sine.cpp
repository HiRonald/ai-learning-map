#include "demos/sine.h"

#include "demos/common.h"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace std;

namespace {

#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

void make_sine(int n, Matrix& data, Matrix& labels) {
    data = Matrix(n, 1);
    labels = Matrix(n, 1);
    for (int i = 0; i < n; ++i) {
        double x = (2.0 * M_PI) * i / (n - 1);
        data.set_data(i, 0, x);
        labels.set_data(i, 0, std::sin(x));
    }
}

}  // namespace

namespace demos {

int run_sine() {
    print_separator("Demo: Sine regression");
    cout << "Goal: fit y = sin(x) on [0, 2π].\n"
         << "Role: shows the same MLP loop on a regression task\n"
         << "      (identity output, continuous targets).\n";

    const int n = 64;
    Matrix data, labels;
    make_sine(n, data, labels);
    cout << "samples=" << n << " on [0, 2π]\n";

    const double learning_rate = 0.05;
    auto mlp = build_mlp(/*in*/ 1, /*hidden*/ {32, 32}, /*out*/ 1,
                         "tanh", "identity", "mse", learning_rate);
    mlp->print_architecture(/*verbose=*/false);

    print_separator("Training");
    const int epochs = 4000;
    const int batch_size = 16;
    const int log_interval = 400;
    cout << fixed << setprecision(6);
    cout << "epochs=" << epochs
         << ", batch_size=" << batch_size
         << ", lr=" << learning_rate
         << ", loss=mse\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < n; start += batch_size) {
            int end = min(start + batch_size, n);
            mlp->train(data.get_rows(start, end), labels.get_rows(start, end));
        }

        if (epoch % log_interval == 0 || epoch + 1 == epochs) {
            Matrix pred = mlp->predict(data);
            float loss = mlp->eval(labels);
            double mae = mean_abs_error(pred, labels);
            cout << "Epoch " << setw(4) << epoch
                 << " | loss=" << loss
                 << " | mae=" << mae << "\n";
        }
    }

    print_separator("Inference (selected x)");
    Matrix predictions = mlp->predict(data);
    float final_loss = mlp->eval(labels);
    double final_mae = mean_abs_error(predictions, labels);

    cout << "Final loss=" << final_loss << ", mae=" << final_mae << "\n";

    // 抽查几个相位点：0, π/2, π, 3π/2, 2π
    const double xs[] = {0.0, M_PI / 2, M_PI, 3 * M_PI / 2, 2 * M_PI};
    Matrix probes(5, 1);
    for (int i = 0; i < 5; ++i) {
        probes.set_data(i, 0, xs[i]);
    }
    Matrix probe_pred = mlp->predict(probes);

    cout << "\nProbe points:\n";
    cout << "  x         |  sin(x)   |  pred\n";
    cout << "  ----------|-----------|----------\n";
    for (int i = 0; i < 5; ++i) {
        cout << "  " << setw(8) << xs[i]
             << " |  " << setw(8) << std::sin(xs[i])
             << " |  " << setw(8) << probe_pred.get_data(i, 0) << "\n";
    }

    if (final_mae < 0.08) {
        cout << "\nSine fitted reasonably well (mae < 0.08).\n";
        return 0;
    }
    cout << "\nMAE still high. Try more epochs or lower learning rate.\n";
    return 1;
}

}  // namespace demos
