#include "demos/recall_demo.h"

#include "demos/common.h"
#include "lstm/net.h"
#include "nn/matrix.h"
#include "rnn/net.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace std;

namespace {

constexpr int kInput = 2;  // [value, query]
constexpr int kSeqLen = 12;
constexpr int kHidden = 16;
constexpr int kTrainN = 2048;
constexpr int kTestN = 512;
constexpr int kEpochs = 40;
constexpr int kBatch = 64;
constexpr double kLr = 0.1;

struct Sample {
    vector<double> values;  // T
    double target = 0.0;    // 0 or 1
};

Sample make_sample(mt19937& rng) {
    bernoulli_distribution bit(0.5);
    uniform_real_distribution<double> uni(0.0, 1.0);
    Sample s;
    s.values.assign(static_cast<size_t>(kSeqLen), 0.0);
    s.target = bit(rng) ? 1.0 : 0.0;
    s.values[0] = s.target;
    for (int t = 1; t < kSeqLen - 1; ++t) {
        s.values[static_cast<size_t>(t)] = uni(rng);
    }
    s.values[static_cast<size_t>(kSeqLen - 1)] = 0.0;
    return s;
}

void pack_batch(const vector<Sample>& data, const vector<int>& indices, int start, int end,
                vector<Matrix>& xs_out, Matrix& y_out) {
    const int B = end - start;
    xs_out.assign(static_cast<size_t>(kSeqLen), Matrix(B, kInput, 0.0));
    y_out = Matrix(B, 1);
    for (int i = 0; i < B; ++i) {
        const Sample& s = data[static_cast<size_t>(indices[static_cast<size_t>(start + i)])];
        for (int t = 0; t < kSeqLen; ++t) {
            xs_out[static_cast<size_t>(t)].set_data(i, 0, s.values[static_cast<size_t>(t)]);
            xs_out[static_cast<size_t>(t)].set_data(i, 1, t + 1 == kSeqLen ? 1.0 : 0.0);
        }
        y_out.set_data(i, 0, s.target);
    }
}

double mae_of(const Matrix& pred, const Matrix& y) {
    double s = 0.0;
    const int n = pred.get_row_number();
    for (int i = 0; i < n; ++i) {
        s += abs(pred.get_data(i, 0) - y.get_data(i, 0));
    }
    return s / n;
}

double accuracy_of(const Matrix& pred, const Matrix& y) {
    int ok = 0;
    const int n = pred.get_row_number();
    for (int i = 0; i < n; ++i) {
        const double p = pred.get_data(i, 0) >= 0.5 ? 1.0 : 0.0;
        if (p == y.get_data(i, 0)) {
            ++ok;
        }
    }
    return static_cast<double>(ok) / n;
}

template <typename Net>
pair<double, double> train_and_eval(Net& net, const vector<Sample>& train,
                                    const vector<Sample>& test, mt19937& rng,
                                    const char* name) {
    vector<int> train_idx(train.size());
    for (size_t i = 0; i < train.size(); ++i) {
        train_idx[i] = static_cast<int>(i);
    }
    vector<int> test_idx(test.size());
    for (size_t i = 0; i < test.size(); ++i) {
        test_idx[i] = static_cast<int>(i);
    }

    cout << "\n--- " << name << " ---\n";
    net.print_architecture();
    cout << fixed << setprecision(4);

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        shuffle(train_idx.begin(), train_idx.end(), rng);
        float loss_sum = 0.0f;
        int steps = 0;
        for (int start = 0; start < static_cast<int>(train.size()); start += kBatch) {
            const int end = min(start + kBatch, static_cast<int>(train.size()));
            vector<Matrix> xs;
            Matrix y;
            pack_batch(train, train_idx, start, end, xs, y);
            const float loss = net.train(xs, y);
            if (!isfinite(loss)) {
                cout << "  diverged (NaN)\n";
                return {nan(""), nan("")};
            }
            loss_sum += loss;
            ++steps;
        }
        if (epoch % 10 == 0 || epoch + 1 == kEpochs) {
            vector<Matrix> xs_te;
            Matrix y_te;
            pack_batch(test, test_idx, 0, static_cast<int>(test.size()), xs_te, y_te);
            Matrix pred = net.predict(xs_te);
            cout << "  epoch " << setw(2) << epoch
                 << " | train_loss=" << (loss_sum / max(steps, 1))
                 << " | test_MAE=" << mae_of(pred, y_te)
                 << " | test_acc=" << accuracy_of(pred, y_te) << "\n";
        }
    }

    vector<Matrix> xs_te;
    Matrix y_te;
    pack_batch(test, test_idx, 0, static_cast<int>(test.size()), xs_te, y_te);
    Matrix pred = net.predict(xs_te);
    return {mae_of(pred, y_te), accuracy_of(pred, y_te)};
}

}  // namespace

namespace demos {

int run_recall() {
    print_separator("Demo: recall — delayed recall (synthetic)");
    cout << "Task: remember binary x_0 across " << (kSeqLen - 2)
         << " noise steps; at the final (query) step, output 0/1.\n"
         << "Input = [value, query]. Data: on-the-fly, no download.\n"
         << "Success = LSTM test accuracy >> 50% baseline.\n";

    mt19937 rng(42);
    vector<Sample> train;
    train.reserve(static_cast<size_t>(kTrainN));
    for (int i = 0; i < kTrainN; ++i) {
        train.push_back(make_sample(rng));
    }
    vector<Sample> test;
    test.reserve(static_cast<size_t>(kTestN));
    for (int i = 0; i < kTestN; ++i) {
        test.push_back(make_sample(rng));
    }

    cout << "  seq_len=" << kSeqLen << "  train=" << kTrainN << "  test=" << kTestN
         << "  hidden=" << kHidden << "  epochs=" << kEpochs << "  lr=" << kLr << "\n";

    mt19937 rng_lstm(7);
    LstmNet lstm(kInput, kHidden, kLr);
    const auto [mae_lstm, acc_lstm] = train_and_eval(lstm, train, test, rng_lstm, "LSTM");

    mt19937 rng_rnn(7);
    VanillaRnn rnn(kInput, kHidden, kLr);
    const auto [mae_rnn, acc_rnn] =
        train_and_eval(rnn, train, test, rng_rnn, "Vanilla RNN (same hyperparams)");

    print_separator("Samples (LSTM, test)");
    {
        vector<int> idx(min(8, kTestN));
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = static_cast<int>(i);
        }
        vector<Matrix> xs;
        Matrix y;
        pack_batch(test, idx, 0, static_cast<int>(idx.size()), xs, y);
        Matrix pred = lstm.predict(xs);
        cout << left << setw(8) << "y" << setw(10) << "y_hat" << "pred@0.5\n";
        cout << fixed << setprecision(3);
        for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
            const double yt = y.get_data(i, 0);
            const double p = pred.get_data(i, 0);
            cout << left << setw(8) << yt << setw(10) << p << (p >= 0.5 ? 1 : 0) << "\n";
        }
    }

    print_separator("Takeaway");
    cout << setprecision(3);
    cout << "baseline (always 0.5 → random bit): acc≈0.50\n"
         << "LSTM:        MAE=" << mae_lstm << "  acc=" << acc_lstm << "\n"
         << "Vanilla RNN: MAE=" << mae_rnn << "  acc=" << acc_rnn << "\n"
         << "LSTM cell keeps a slow c across noise; see `gates` for f/i/o.\n"
         << "Short T: plain RNN may also latch — longer deps → LSTM / Attention.\n";

    const bool ok = isfinite(acc_lstm) && acc_lstm >= 0.95;
    return ok ? 0 : 1;
}

}  // namespace demos
