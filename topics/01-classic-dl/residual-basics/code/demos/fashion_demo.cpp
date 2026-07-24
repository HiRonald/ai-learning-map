#include "demos/fashion_demo.h"

#include "demos/common.h"
#include "demos/download.h"
#include "demos/idx_io.h"
#include "residual/cnn_net.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

namespace {

constexpr int kNumClasses = 10;

const char* kUrlTrainImages =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz";
const char* kUrlTrainLabels =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz";
const char* kUrlTestImages =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz";
const char* kUrlTestLabels =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz";

vector<int> shuffled_indices(int n, unsigned seed) {
    vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
        idx[i] = i;
    }
    mt19937 rng(seed);
    shuffle(idx.begin(), idx.end(), rng);
    return idx;
}

void shuffle_pair(Matrix& x, Matrix& y, unsigned seed) {
    const int n = x.get_row_number();
    vector<int> order = shuffled_indices(n, seed);
    Matrix xs(n, x.get_col_number());
    Matrix ys(n, y.get_col_number());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < x.get_col_number(); ++j) {
            xs.set_data(i, j, x.get_data(order[i], j));
        }
        for (int j = 0; j < y.get_col_number(); ++j) {
            ys.set_data(i, j, y.get_data(order[i], j));
        }
    }
    x = xs;
    y = ys;
}

struct EvalStats {
    Matrix pred;
    double loss = 0.0;
    double acc = 0.0;
};

EvalStats eval_batched(DeepCnn& net, const Matrix& x, const Matrix& y, int batch) {
    const int n = x.get_row_number();
    EvalStats s;
    s.pred = Matrix(n, kNumClasses);
    double loss_sum = 0.0;
    for (int start = 0; start < n; start += batch) {
        const int end = min(start + batch, n);
        Matrix chunk = net.predict(x.get_rows(start, end));
        loss_sum += net.eval(y.get_rows(start, end)) * (end - start);
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < kNumClasses; ++j) {
                s.pred.set_data(i, j, chunk.get_data(i - start, j));
            }
        }
    }
    s.loss = loss_sum / n;
    s.acc = demos::argmax_accuracy(s.pred, y);
    return s;
}

struct TrainResult {
    double test_acc = 0.0;
    double wall_s = 0.0;
};

TrainResult train_cnn(DeepCnn& net, const Matrix& train_x, const Matrix& train_y,
                      const Matrix& test_x, const Matrix& test_y, int epochs, int batch,
                      int log_every) {
    const int train_n = train_x.get_row_number();
    cout << fixed << setprecision(6);
    const auto t0 = Clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < train_n; start += batch) {
            const int end = min(start + batch, train_n);
            net.train(train_x.get_rows(start, end), train_y.get_rows(start, end));
        }
        if (epoch % log_every == 0 || epoch + 1 == epochs) {
            EvalStats ev = eval_batched(net, test_x, test_y, batch);
            cout << "  epoch " << setw(2) << epoch
                 << " | test_loss=" << ev.loss
                 << " | test_acc=" << setprecision(2) << ev.acc * 100.0 << "%"
                 << setprecision(6) << "\n";
        }
    }
    EvalStats final_ev = eval_batched(net, test_x, test_y, batch);
    TrainResult r;
    r.test_acc = final_ev.acc;
    r.wall_s = chrono::duration<double>(Clock::now() - t0).count();
    return r;
}

}  // namespace

namespace demos {

int run_fashion() {
    print_separator("Demo: Fashion — plain CNN vs residual CNN");
    cout << "Same stem/head; trunk is stacked Conv+ReLU or residual conv blocks.\n"
         << "Match # of Conv layers in the trunk (plain = 2× residual blocks).\n"
         << "Connects residual-basics to cnn-basics (vision is where ResNet lives).\n";

    const auto root = data_root() / "fashion-mnist";
    print_separator("Download");
    ensure_file(root / "train-images-idx3-ubyte", kUrlTrainImages);
    ensure_file(root / "train-labels-idx1-ubyte", kUrlTrainLabels);
    ensure_file(root / "t10k-images-idx3-ubyte", kUrlTestImages);
    ensure_file(root / "t10k-labels-idx1-ubyte", kUrlTestLabels);

    IdxImages train_img = read_idx_images(root / "train-images-idx3-ubyte");
    IdxLabels train_lbl = read_idx_labels(root / "train-labels-idx1-ubyte");
    IdxImages test_img = read_idx_images(root / "t10k-images-idx3-ubyte");
    IdxLabels test_lbl = read_idx_labels(root / "t10k-labels-idx1-ubyte");

    // 朴素卷积偏慢：子集/深度收紧，仍要能看出 residual 更好训
    const int train_n = 2000;
    const int test_n = 400;
    vector<int> train_idx = shuffled_indices(train_img.count, 41);
    train_idx.resize(train_n);
    vector<int> test_idx = shuffled_indices(test_img.count, 42);
    test_idx.resize(test_n);

    Matrix train_x, train_y, test_x, test_y;
    materialize_subset(train_img, train_lbl, train_idx, kNumClasses, train_x, train_y);
    materialize_subset(test_img, test_lbl, test_idx, kNumClasses, test_x, test_y);
    shuffle_pair(train_x, train_y, 43);

    constexpr int channels = 8;
    constexpr int res_blocks = 3;  // trunk: 6 Conv
    constexpr int epochs = 5;
    constexpr int batch = 32;
    constexpr double lr = 0.05;

    cout << "subset train=" << train_n << " test=" << test_n
         << " | C=" << channels << " | res_blocks=" << res_blocks
         << " | epochs=" << epochs << "\n";

    print_separator("Plain CNN trunk");
    DeepCnn plain(channels, res_blocks, DeepCnn::Mode::Plain, lr);
    plain.print_architecture();
    TrainResult plain_r =
        train_cnn(plain, train_x, train_y, test_x, test_y, epochs, batch, /*log=*/1);

    print_separator("Residual CNN trunk");
    DeepCnn res(channels, res_blocks, DeepCnn::Mode::Residual, lr);
    res.print_architecture();
    TrainResult res_r =
        train_cnn(res, train_x, train_y, test_x, test_y, epochs, batch, /*log=*/1);

    print_separator("Head-to-head");
    cout << fixed;
    cout << left << setw(14) << "model" << right
         << setw(10) << "params"
         << setw(10) << "wall_s"
         << setw(12) << "test_acc" << "\n";
    cout << left << setw(14) << "plain-cnn" << right
         << setw(10) << plain.param_count()
         << setw(10) << setprecision(1) << plain_r.wall_s
         << setw(11) << setprecision(2) << plain_r.test_acc * 100.0 << "%\n";
    cout << left << setw(14) << "residual-cnn" << right
         << setw(10) << res.param_count()
         << setw(10) << setprecision(1) << res_r.wall_s
         << setw(11) << setprecision(2) << res_r.test_acc * 100.0 << "%\n";

    cout << "\nSame Conv budget; residual should train more stably / often higher acc.\n"
         << "For the sharpest x+F(x) demo (no vision), see `identity`.\n";

    if (res_r.test_acc > plain_r.test_acc && res_r.test_acc >= 0.55) {
        cout << "\nOK: residual CNN beat plain CNN on Fashion subset.\n";
        return 0;
    }
    // 若持平但都学会了，也算可接受（Fashion 小子集噪声大）
    if (res_r.test_acc >= 0.60 && plain_r.test_acc >= 0.50) {
        cout << "\nOK: both learned; residual not clearly ahead this run (variance OK).\n";
        return 0;
    }
    cout << "\nUnexpected: residual CNN did not look healthier. Check training.\n";
    return 1;
}

}  // namespace demos
