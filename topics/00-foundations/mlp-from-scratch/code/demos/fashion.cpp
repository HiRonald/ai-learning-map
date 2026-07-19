#include "demos/fashion.h"

#include "demos/common.h"
#include "demos/download.h"
#include "demos/idx_io.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

namespace {

constexpr int kNumClasses = 10;
constexpr int kImageSide = 28;

const char* kClassNames[kNumClasses] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
};

// Zalando 官方镜像；首次运行会 curl + gunzip
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
    const int ramp_n = 10;
    cout << title << "\n";
    for (int r = 0; r < kImageSide; ++r) {
        cout << "  ";
        for (int c = 0; c < kImageSide; ++c) {
            double v = x.get_data(row, r * kImageSide + c);  // [0,1]
            int idx = static_cast<int>(std::round(v * (ramp_n - 1)));
            idx = max(0, min(ramp_n - 1, idx));
            cout << ramp[idx] << ramp[idx];  // 横向拉伸，终端里更方一点
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

}  // namespace

namespace demos {

int run_fashion() {
    print_separator("Demo: Fashion-MNIST");
    cout << "Goal: classify 28x28 grayscale clothing into 10 categories.\n"
         << "Role: first \"real\" dataset demo — download, parse IDX, train,\n"
         << "      then ASCII-print predictions (correct + mistakes).\n";

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

    // 纯 C++ 全量 60k 会偏慢；子集已经足够看出「学会认衣服」
    const int train_n = 40000;
    const int test_n = 8000;
    vector<int> train_idx = shuffled_indices(train_img.count, 21);
    train_idx.resize(train_n);
    vector<int> test_idx = shuffled_indices(test_img.count, 22);
    test_idx.resize(test_n);

    Matrix train_x, train_y, test_x, test_y;
    materialize_subset(train_img, train_lbl, train_idx, kNumClasses, train_x, train_y);
    materialize_subset(test_img, test_lbl, test_idx, kNumClasses, test_x, test_y);
    shuffle_pair(train_x, train_y, 23);

    cout << "subset: train=" << train_n << ", test=" << test_n
         << ", classes=" << kNumClasses << "\n";

    const double learning_rate = 0.15;
    // 输出 identity logits + softmax_ce；隐藏层 relu
    auto mlp = build_mlp(/*in*/ 784, /*hidden*/ {128, 64}, /*out*/ 10,
                         "relu", "identity", "softmax_ce", learning_rate);
    mlp->print_architecture(/*verbose=*/false);

    print_separator("Training");
    const int epochs = 10;
    const int batch_size = 64;
    const int log_interval = 2;
    cout << fixed << setprecision(6);
    cout << "epochs=" << epochs
         << ", batch_size=" << batch_size
         << ", lr=" << learning_rate
         << ", loss=softmax_ce\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < train_n; start += batch_size) {
            int end = min(start + batch_size, train_n);
            mlp->train(train_x.get_rows(start, end), train_y.get_rows(start, end));
        }

        if (epoch % log_interval == 0 || epoch + 1 == epochs) {
            Matrix pred = mlp->predict(test_x);
            float loss = mlp->eval(test_y);
            double acc = argmax_accuracy(pred, test_y);
            cout << "Epoch " << setw(2) << epoch
                 << " | test_loss=" << loss
                 << " | test_acc=" << setprecision(2) << acc * 100.0 << "%"
                 << setprecision(6) << "\n";
        }
    }

    Matrix test_pred = mlp->predict(test_x);
    double final_acc = argmax_accuracy(test_pred, test_y);
    cout << "\nFinal test accuracy: " << setprecision(2) << final_acc * 100.0 << "%\n"
         << setprecision(6);

    // 每类召回一点感觉：对角线粗看
    vector<int> class_total(kNumClasses, 0);
    vector<int> class_correct(kNumClasses, 0);
    for (int i = 0; i < test_n; ++i) {
        int y = argmax_row(test_y, i);
        int p = argmax_row(test_pred, i);
        ++class_total[y];
        if (y == p) {
            ++class_correct[y];
        }
    }
    cout << "\nPer-class recall on test subset:\n";
    for (int c = 0; c < kNumClasses; ++c) {
        double r = class_total[c] == 0
                       ? 0.0
                       : 100.0 * class_correct[c] / class_total[c];
        cout << "  " << setw(12) << left << kClassNames[c] << right
             << "  " << setprecision(1) << r << "%\n";
    }
    cout << setprecision(6);

    print_separator("ASCII gallery");
    // 找若干对/错样本开盲盒
    vector<int> correct_ids;
    vector<int> wrong_ids;
    for (int i = 0; i < test_n; ++i) {
        if (argmax_row(test_pred, i) == argmax_row(test_y, i)) {
            correct_ids.push_back(i);
        } else {
            wrong_ids.push_back(i);
        }
    }

    auto show = [&](int i, const string& tag) {
        int y = argmax_row(test_y, i);
        int p = argmax_row(test_pred, i);
        // 简易置信度：softmax 行上的 max（重新算一遍避免改 API）
        double max_logit = test_pred.get_data(i, 0);
        for (int j = 1; j < kNumClasses; ++j) {
            max_logit = max(max_logit, test_pred.get_data(i, j));
        }
        double sum = 0.0;
        double conf = 0.0;
        for (int j = 0; j < kNumClasses; ++j) {
            double e = exp(test_pred.get_data(i, j) - max_logit);
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

    const int show_n = 3;
    for (int k = 0; k < show_n && k < static_cast<int>(correct_ids.size()); ++k) {
        show(correct_ids[static_cast<size_t>(k) * correct_ids.size() / show_n], "hit");
    }
    for (int k = 0; k < show_n && k < static_cast<int>(wrong_ids.size()); ++k) {
        show(wrong_ids[static_cast<size_t>(k) * wrong_ids.size() / show_n], "miss");
    }

    if (final_acc >= 0.70) {
        cout << "\nFashion-MNIST subset learned (>= 70% test acc). "
             << "Shirt vs T-shirt/Coat 仍容易混——MLP 本来就没有空间归纳偏置。\n";
        return 0;
    }
    cout << "\nTest accuracy below 70%. Try more epochs or a larger train subset.\n";
    return 1;
}

}  // namespace demos
