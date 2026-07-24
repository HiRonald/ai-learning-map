#include "demos/fashion_demo.h"

#include "cnn/net.h"
#include "demos/common.h"
#include "demos/download.h"
#include "demos/idx_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

namespace {

constexpr int kNumClasses = 10;
constexpr int kImageSide = 28;

const char* kClassNames[kNumClasses] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
};

const char* kUrlTrainImages =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz";
const char* kUrlTrainLabels =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz";
const char* kUrlTestImages =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz";
const char* kUrlTestLabels =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz";

void print_ascii_image(const Matrix& x, int row, const string& title) {
    static const char* ramp = " .:-=+*#%@";
    constexpr int ramp_n = 10;
    cout << title << "\n";
    for (int r = 0; r < kImageSide; ++r) {
        cout << "  ";
        for (int c = 0; c < kImageSide; ++c) {
            double v = x.get_data(row, r * kImageSide + c);
            int idx = static_cast<int>(round(v * (ramp_n - 1)));
            idx = max(0, min(ramp_n - 1, idx));
            cout << ramp[idx] << ramp[idx];
        }
        cout << "\n";
    }
}

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

// 每个 batch 只 forward 一次：同时收集 logits / loss / acc
EvalStats eval_cnn(LeNet& net, const Matrix& x, const Matrix& y, int batch) {
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

int mlp_param_count() {
    // 784→128→64→10
    return 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10;
}

}  // namespace

namespace demos {

int run_fashion() {
    print_separator("Demo: Fashion-MNIST — MLP vs LeNet (same subset)");
    cout << "Goal: fair head-to-head on identical data/epochs.\n"
         << "Expect: LeNet fewer params than big MLP; wall-clock slower (naive conv).\n"
         << "Fashion silhouettes are easy for MLP — accuracy gap may be modest.\n";

    const auto root = data_root() / "fashion-mnist";
    cout << "Data dir: " << root << "\n";
    print_separator("Download");
    ensure_file(root / "train-images-idx3-ubyte", kUrlTrainImages);
    ensure_file(root / "train-labels-idx1-ubyte", kUrlTrainLabels);
    ensure_file(root / "t10k-images-idx3-ubyte", kUrlTestImages);
    ensure_file(root / "t10k-labels-idx1-ubyte", kUrlTestLabels);

    print_separator("Load IDX");
    IdxImages train_img = read_idx_images(root / "train-images-idx3-ubyte");
    IdxLabels train_lbl = read_idx_labels(root / "train-labels-idx1-ubyte");
    IdxImages test_img = read_idx_images(root / "t10k-images-idx3-ubyte");
    IdxLabels test_lbl = read_idx_labels(root / "t10k-labels-idx1-ubyte");
    cout << "train=" << train_img.count << "  test=" << test_img.count
         << "  shape=" << train_img.rows << "x" << train_img.cols << "\n";

    const int train_n = 4000;
    const int test_n = 800;
    vector<int> train_idx = shuffled_indices(train_img.count, 31);
    train_idx.resize(train_n);
    vector<int> test_idx = shuffled_indices(test_img.count, 32);
    test_idx.resize(test_n);

    Matrix train_x, train_y, test_x, test_y;
    materialize_subset(train_img, train_lbl, train_idx, kNumClasses, train_x, train_y);
    materialize_subset(test_img, test_lbl, test_idx, kNumClasses, test_x, test_y);
    shuffle_pair(train_x, train_y, 33);

    const int epochs = 6;
    const int batch_size = 32;
    cout << "subset: train=" << train_n << ", test=" << test_n
         << ", epochs=" << epochs << ", batch=" << batch_size << "\n";

    // ----- MLP baseline（同子集，秒级）-----
    print_separator("Baseline MLP (same subset)");
    const double mlp_lr = 0.15;
    auto mlp = build_mlp(/*in*/ 784, /*hidden*/ {128, 64}, /*out*/ 10, "relu", "identity",
                         "softmax_ce", mlp_lr);
    mlp->print_architecture(/*verbose=*/false);
    cout << "params ≈ " << mlp_param_count() << ", lr=" << mlp_lr << "\n";

    const auto mlp_t0 = Clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < train_n; start += batch_size) {
            const int end = min(start + batch_size, train_n);
            mlp->train(train_x.get_rows(start, end), train_y.get_rows(start, end));
        }
    }
    Matrix mlp_pred = mlp->predict(test_x);
    const double mlp_acc = argmax_accuracy(mlp_pred, test_y);
    const float mlp_loss = [&]() {
        mlp->predict(test_x);
        return mlp->eval(test_y);
    }();
    const double mlp_sec =
        chrono::duration<double>(Clock::now() - mlp_t0).count();
    cout << fixed << setprecision(2);
    cout << "MLP  test_acc=" << mlp_acc * 100.0 << "%  loss=" << setprecision(4) << mlp_loss
         << "  wall=" << setprecision(2) << mlp_sec << "s\n";

    // ----- LeNet -----
    print_separator("LeNet");
    const double cnn_lr = 0.1;
    LeNet net(cnn_lr);
    net.print_architecture();

    cout << fixed << setprecision(6);
    const auto cnn_t0 = Clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < train_n; start += batch_size) {
            const int end = min(start + batch_size, train_n);
            net.train(train_x.get_rows(start, end), train_y.get_rows(start, end));
        }
        if (epoch % 2 == 0 || epoch + 1 == epochs) {
            EvalStats ev = eval_cnn(net, test_x, test_y, batch_size);
            cout << "Epoch " << setw(2) << epoch
                 << " | test_loss=" << ev.loss
                 << " | test_acc=" << setprecision(2) << ev.acc * 100.0 << "%"
                 << setprecision(6) << "\n";
        }
    }
    EvalStats cnn_ev = eval_cnn(net, test_x, test_y, batch_size);
    const double cnn_sec =
        chrono::duration<double>(Clock::now() - cnn_t0).count();

    print_separator("Head-to-head (same subset)");
    cout << fixed;
    cout << left << setw(8) << "model" << right
         << setw(10) << "params"
         << setw(10) << "wall_s"
         << setw(12) << "test_acc" << "\n";
    cout << left << setw(8) << "MLP" << right
         << setw(10) << mlp_param_count()
         << setw(10) << setprecision(2) << mlp_sec
         << setw(11) << setprecision(2) << mlp_acc * 100.0 << "%\n";
    cout << left << setw(8) << "LeNet" << right
         << setw(10) << net.param_count()
         << setw(10) << setprecision(2) << cnn_sec
         << setw(11) << setprecision(2) << cnn_ev.acc * 100.0 << "%\n";

    cout << setprecision(6);
    cout << "\nReading the table:\n"
         << "  • Params: LeNet 通常仍少于大 MLP（卷积省参数；FC 头 120/84 会占大头）。\n"
         << "  • Acc: Fashion 对 MLP 友好；LeNet 往往略好或接近。\n"
         << "  • Wall time: 朴素卷积实现慢，不是 LeNet「本质」慢。\n";

    vector<int> class_total(kNumClasses, 0);
    vector<int> class_correct(kNumClasses, 0);
    for (int i = 0; i < test_n; ++i) {
        int y = argmax_row(test_y, i);
        int p = argmax_row(cnn_ev.pred, i);
        ++class_total[y];
        if (y == p) {
            ++class_correct[y];
        }
    }
    cout << "\nLeNet per-class recall:\n";
    for (int c = 0; c < kNumClasses; ++c) {
        double r = class_total[c] == 0 ? 0.0 : 100.0 * class_correct[c] / class_total[c];
        cout << "  " << setw(12) << left << kClassNames[c] << right
             << "  " << setprecision(1) << r << "%\n";
    }
    cout << setprecision(6);

    print_separator("ASCII gallery (LeNet)");
    vector<int> correct_ids;
    vector<int> wrong_ids;
    for (int i = 0; i < test_n; ++i) {
        if (argmax_row(cnn_ev.pred, i) == argmax_row(test_y, i)) {
            correct_ids.push_back(i);
        } else {
            wrong_ids.push_back(i);
        }
    }

    auto show = [&](int i, const string& tag) {
        int y = argmax_row(test_y, i);
        int p = argmax_row(cnn_ev.pred, i);
        double max_logit = cnn_ev.pred.get_data(i, 0);
        for (int j = 1; j < kNumClasses; ++j) {
            max_logit = max(max_logit, cnn_ev.pred.get_data(i, j));
        }
        double sum = 0.0;
        double conf = 0.0;
        for (int j = 0; j < kNumClasses; ++j) {
            double e = exp(cnn_ev.pred.get_data(i, j) - max_logit);
            sum += e;
            if (j == p) {
                conf = e;
            }
        }
        conf /= sum;
        cout << "\n[" << tag << "] true=" << kClassNames[y]
             << "  pred=" << kClassNames[p]
             << "  conf=" << setprecision(2) << conf * 100.0 << "%"
             << setprecision(6) << "\n";
        print_ascii_image(test_x, i, "");
    };

    const int show_n = 2;
    for (int k = 0; k < show_n && k < static_cast<int>(correct_ids.size()); ++k) {
        show(correct_ids[static_cast<size_t>(k) * correct_ids.size() / show_n], "hit");
    }
    for (int k = 0; k < show_n && k < static_cast<int>(wrong_ids.size()); ++k) {
        show(wrong_ids[static_cast<size_t>(k) * wrong_ids.size() / show_n], "miss");
    }

    // 成功条件：CNN 能学到（≥70%），且参数少于 MLP——不要求必定碾压准确率
    if (cnn_ev.acc >= 0.70 && net.param_count() < mlp_param_count()) {
        cout << "\nOK: LeNet learned the subset with fewer params than the MLP baseline.\n";
        return 0;
    }
    cout << "\nUnexpected: LeNet acc < 70% or params not below MLP. Check training.\n";
    return 1;
}

}  // namespace demos
