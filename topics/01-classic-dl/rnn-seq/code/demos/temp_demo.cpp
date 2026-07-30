#include "demos/temp_demo.h"

#include "demos/common.h"
#include "nn/matrix.h"
#include "rnn/net.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

namespace {

// jsDelivr 镜像比 raw.githubusercontent.com 更稳（后者偶发超时）
constexpr const char* kTempUrl =
    "https://cdn.jsdelivr.net/gh/jbrownlee/Datasets@master/daily-min-temperatures.csv";

fs::path rnn_data_root() {
    if (const char* env = getenv("RNN_DATA_DIR")) {
        return fs::path(env);
    }
    return fs::path("topics/01-classic-dl/rnn-seq/data");
}

void ensure_temp_csv(const fs::path& dest) {
    if (fs::exists(dest) && fs::is_regular_file(dest) && fs::file_size(dest) > 0) {
        cout << "  cached: " << dest << "\n";
        return;
    }
    fs::create_directories(dest.parent_path());
    cout << "  downloading: " << kTempUrl << "\n";
    const string cmd =
        "curl -fsSL --retry 3 --retry-delay 1 -o \"" + dest.string() + "\" \"" + kTempUrl + "\"";
    cout << "  $ " << cmd << "\n";
    if (system(cmd.c_str()) != 0) {
        throw runtime_error("failed to download Daily Min Temperatures CSV");
    }
    if (!fs::exists(dest) || fs::file_size(dest) == 0) {
        throw runtime_error("temperature CSV missing after download: " + dest.string());
    }
    cout << "  ready: " << dest << "\n";
}

// Date,Temp  /  1981-01-01,20.7
vector<double> load_temps(const fs::path& path) {
    ifstream in(path);
    if (!in) {
        throw runtime_error("cannot open " + path.string());
    }
    string line;
    getline(in, line);  // header
    vector<double> series;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto comma = line.find_last_of(',');
        if (comma == string::npos) {
            continue;
        }
        string num = line.substr(comma + 1);
        num.erase(remove(num.begin(), num.end(), '"'), num.end());
        num.erase(remove_if(num.begin(), num.end(),
                            [](unsigned char c) { return isspace(c) != 0; }),
                  num.end());
        if (num.empty()) {
            continue;
        }
        series.push_back(stod(num));
    }
    if (series.size() < 100) {
        throw runtime_error("temperature series too short: " + to_string(series.size()));
    }
    return series;
}

struct WindowDataset {
    vector<vector<double>> xs;  // N × T
    vector<double> ys;          // N
};

WindowDataset make_windows(const vector<double>& series, int lookback) {
    WindowDataset d;
    for (size_t i = 0; i + static_cast<size_t>(lookback) < series.size(); ++i) {
        vector<double> w(static_cast<size_t>(lookback));
        for (int t = 0; t < lookback; ++t) {
            w[static_cast<size_t>(t)] = series[i + static_cast<size_t>(t)];
        }
        d.xs.push_back(std::move(w));
        d.ys.push_back(series[i + static_cast<size_t>(lookback)]);
    }
    return d;
}

struct Norm {
    double min_v = 0.0;
    double max_v = 1.0;
    double scale(double v) const {
        return (v - min_v) / (max_v - min_v + 1e-12);
    }
    double unscale(double v) const {
        return v * (max_v - min_v) + min_v;
    }
};

void pack_batch(const WindowDataset& data, const vector<int>& indices, int start, int end,
                const Norm& norm, vector<Matrix>& xs_out, Matrix& y_out) {
    const int B = end - start;
    const int T = static_cast<int>(data.xs[0].size());
    xs_out.assign(static_cast<size_t>(T), Matrix(B, 1));
    y_out = Matrix(B, 1);
    for (int i = 0; i < B; ++i) {
        const int idx = indices[static_cast<size_t>(start + i)];
        for (int t = 0; t < T; ++t) {
            xs_out[static_cast<size_t>(t)].set_data(
                i, 0, norm.scale(data.xs[static_cast<size_t>(idx)][static_cast<size_t>(t)]));
        }
        y_out.set_data(i, 0, norm.scale(data.ys[static_cast<size_t>(idx)]));
    }
}

double mae_original(const Matrix& pred_norm, const Matrix& y_norm, const Norm& norm) {
    double s = 0.0;
    const int n = pred_norm.get_row_number();
    for (int i = 0; i < n; ++i) {
        const double p = norm.unscale(pred_norm.get_data(i, 0));
        const double y = norm.unscale(y_norm.get_data(i, 0));
        s += abs(p - y);
    }
    return s / n;
}

}  // namespace

namespace demos {

int run_temp() {
    constexpr int lookback = 14;
    constexpr int hidden = 32;
    constexpr int epochs = 40;
    constexpr int batch = 64;
    constexpr double lr = 0.05;
    constexpr double train_ratio = 0.8;

    print_separator("Demo: temp — Vanilla RNN next-day temperature");
    cout << "Dataset: Daily Min Temperatures (Melbourne, 1981–1990).\n"
         << "Task: given the last " << lookback
         << " days, predict tomorrow's min temperature (°C).\n"
         << "Network: Vanilla RNN (shared cell) → Dense(1). Loss: MSE on min-max scaled values.\n";

    const fs::path csv = rnn_data_root() / "daily-min-temperatures.csv";
    print_separator("Data");
    ensure_temp_csv(csv);
    vector<double> series = load_temps(csv);
    cout << "  days=" << series.size()
         << "  range=[" << *min_element(series.begin(), series.end())
         << ", " << *max_element(series.begin(), series.end()) << "] °C\n";

    WindowDataset all = make_windows(series, lookback);
    const int n = static_cast<int>(all.ys.size());
    const int train_n = static_cast<int>(n * train_ratio);
    const int test_n = n - train_n;
    if (train_n < 100 || test_n < 50) {
        throw runtime_error("not enough windows after split");
    }
    cout << "  windows=" << n << " (train=" << train_n << ", test=" << test_n
         << "), lookback=" << lookback << "\n";

    // 只用训练集估尺度，避免偷看测试段
    Norm norm;
    norm.min_v = all.xs[0][0];
    norm.max_v = all.xs[0][0];
    for (int i = 0; i < train_n; ++i) {
        for (double v : all.xs[static_cast<size_t>(i)]) {
            norm.min_v = min(norm.min_v, v);
            norm.max_v = max(norm.max_v, v);
        }
        norm.min_v = min(norm.min_v, all.ys[static_cast<size_t>(i)]);
        norm.max_v = max(norm.max_v, all.ys[static_cast<size_t>(i)]);
    }
    cout << "  train scale min=" << norm.min_v << " max=" << norm.max_v << " °C\n";

    vector<int> train_idx(static_cast<size_t>(train_n));
    for (int i = 0; i < train_n; ++i) {
        train_idx[static_cast<size_t>(i)] = i;
    }
    vector<int> test_idx(static_cast<size_t>(test_n));
    for (int i = 0; i < test_n; ++i) {
        test_idx[static_cast<size_t>(i)] = train_n + i;
    }

    VanillaRnn net(/*input=*/1, hidden, lr);
    net.print_architecture();
    cout << "epochs=" << epochs << ", batch=" << batch << ", lr=" << lr << "\n";

    mt19937 rng(7);
    print_separator("Training");
    cout << fixed << setprecision(5);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle(train_idx.begin(), train_idx.end(), rng);
        float loss_sum = 0.0f;
        int steps = 0;
        for (int start = 0; start < train_n; start += batch) {
            const int end = min(start + batch, train_n);
            vector<Matrix> xs;
            Matrix y;
            pack_batch(all, train_idx, start, end, norm, xs, y);
            loss_sum += net.train(xs, y);
            ++steps;
        }

        if (epoch % 10 == 0 || epoch + 1 == epochs) {
            vector<Matrix> xs_te;
            Matrix y_te;
            pack_batch(all, test_idx, 0, test_n, norm, xs_te, y_te);
            Matrix pred = net.predict(xs_te);
            const float mse = net.eval_mse(y_te);
            const double mae = mae_original(pred, y_te, norm);
            cout << "  epoch " << setw(2) << epoch
                 << " | train_loss=" << (loss_sum / steps)
                 << " | test_mse=" << mse
                 << " | test_MAE=" << setprecision(2) << mae << " °C"
                 << setprecision(5) << "\n";
        }
    }

    vector<Matrix> xs_te;
    Matrix y_te;
    pack_batch(all, test_idx, 0, test_n, norm, xs_te, y_te);
    Matrix pred = net.predict(xs_te);
    const double final_mae = mae_original(pred, y_te, norm);

    print_separator("Samples (test, °C)");
    cout << left << setw(8) << "idx" << setw(10) << "y" << setw(10) << "y_hat"
         << "err\n";
    cout << fixed << setprecision(1);
    vector<int> show;
    for (int i = 0; i < min(4, test_n); ++i) {
        show.push_back(i);
    }
    for (int i = max(0, test_n - 4); i < test_n; ++i) {
        if (find(show.begin(), show.end(), i) == show.end()) {
            show.push_back(i);
        }
    }
    for (int i : show) {
        const double y = norm.unscale(y_te.get_data(i, 0));
        const double p = norm.unscale(pred.get_data(i, 0));
        cout << left << setw(8) << (train_n + i) << setw(10) << y << setw(10) << p << (p - y)
             << "\n";
    }

    double baseline = 0.0;
    for (int i = 0; i < test_n; ++i) {
        const int idx = test_idx[static_cast<size_t>(i)];
        const double last = all.xs[static_cast<size_t>(idx)].back();
        baseline += abs(last - all.ys[static_cast<size_t>(idx)]);
    }
    baseline /= test_n;

    print_separator("Takeaway");
    cout << setprecision(2);
    cout << "test_MAE=" << final_mae << " °C (lower is better).\n"
         << "naive baseline (predict yesterday): MAE=" << baseline << " °C\n"
         << "Same Vanilla RNN cell reused for each of the " << lookback
         << " days; final h predicts tomorrow.\n"
         << "Not SOTA — point is sequence state + shared weights. Longer memory → LSTM/Attention.\n";

    return final_mae < baseline ? 0 : 1;
}

}  // namespace demos
