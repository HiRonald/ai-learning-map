#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "demos/common.h"
#include "demos/download.h"
#include "demos/idx_io.h"

using namespace std;
namespace fs = std::filesystem;

namespace {

constexpr int kNumClasses = 10;

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

struct ModeConfig {
    const char* name;
    int train_n;
    int val_n;
    int test_n;
    int epochs;
    int batch_size;
    double learning_rate;
    int patience;  // <0：不早停；否则 val_acc 连续 patience 个 epoch 无提升则停
    const char* story;
};

// normal：数据够用，train≈val 一起涨
// overfit：砍训练量 + 拉长 epoch，逼出 train↑ val 停住/掉
// earlystop：与 overfit 同设置，但 val 平台期停训并恢复最优权重
const ModeConfig kNormal = {
    "normal", 20000, 5000, 5000, 10, 64, 0.15, -1,
    "Enough train data: train and val should rise together.",
};
const ModeConfig kOverfit = {
    "overfit", 3000, 5000, 5000, 40, 64, 0.15, -1,
    "Tiny train set + many epochs: watch train keep rising while val stalls.",
};
const ModeConfig kEarlyStop = {
    "earlystop", 3000, 5000, 5000, 40, 64, 0.15, 5,
    "Same starved setup as overfit, but stop when val_acc plateaus; restore best weights.",
};

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

struct Metrics {
    float loss = 0.0f;
    double acc = 0.0;
};

Metrics eval_split(Mlp& mlp, const Matrix& x, const Matrix& y) {
    Matrix pred = mlp.predict(x);
    Metrics m;
    m.loss = mlp.eval(y);
    m.acc = demos::argmax_accuracy(pred, y);
    return m;
}

int majority_class(const Matrix& one_hot) {
    vector<int> counts(kNumClasses, 0);
    for (int i = 0; i < one_hot.get_row_number(); ++i) {
        ++counts[demos::argmax_row(one_hot, i)];
    }
    return static_cast<int>(max_element(counts.begin(), counts.end()) - counts.begin());
}

double constant_accuracy(const Matrix& one_hot, int cls) {
    int correct = 0;
    const int n = one_hot.get_row_number();
    for (int i = 0; i < n; ++i) {
        if (demos::argmax_row(one_hot, i) == cls) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / n;
}

void print_per_class_recall(const Matrix& logits, const Matrix& one_hot) {
    vector<int> total(kNumClasses, 0);
    vector<int> hit(kNumClasses, 0);
    const int n = logits.get_row_number();
    for (int i = 0; i < n; ++i) {
        int y = demos::argmax_row(one_hot, i);
        int p = demos::argmax_row(logits, i);
        ++total[y];
        if (y == p) {
            ++hit[y];
        }
    }
    cout << "\nPer-class recall (val):\n";
    for (int c = 0; c < kNumClasses; ++c) {
        double r = total[c] == 0 ? 0.0 : 100.0 * hit[c] / total[c];
        cout << "  " << setw(12) << left << kClassNames[c] << right
             << "  " << setprecision(1) << r << "%\n";
    }
    cout << setprecision(6);
}

fs::path curve_path(const string& mode) {
    fs::path dir = "topics/00-foundations/train-eval-basics/data";
    if (const char* env = getenv("TRAIN_EVAL_DATA_DIR")) {
        dir = env;
    }
    fs::create_directories(dir);
    return dir / ("curve_" + mode + ".csv");
}

void print_usage(const char* argv0) {
    cout
        << "Usage: " << argv0 << " [mode]\n"
        << "\n"
        << "Modes:\n"
        << "  normal     train=20k / val=5k / test=5k, 10 epochs (default)\n"
        << "  overfit    train=3k  / val=5k / test=5k, 40 epochs\n"
        << "  earlystop  same as overfit + patience=5, restore best weights\n"
        << "  list       Print modes\n"
        << "\n"
        << "Examples:\n"
        << "  make run-eval\n"
        << "  make run-eval MODE=overfit\n"
        << "  make run-eval MODE=earlystop\n";
}

int run_mode(const ModeConfig& cfg) {
    demos::print_separator(string("train-eval: ") + cfg.name);
    cout << cfg.story << "\n"
         << "Split roles: train=update weights, val=watch generalization,\n"
         << "             test=look once at the end (do not tune on it).\n";

    const auto root = demos::data_root() / "fashion-mnist";
    cout << "Fashion data dir: " << root << "\n";
    demos::print_separator("Download");
    demos::ensure_file(root / "train-images-idx3-ubyte", kUrlTrainImages);
    demos::ensure_file(root / "train-labels-idx1-ubyte", kUrlTrainLabels);
    demos::ensure_file(root / "t10k-images-idx3-ubyte", kUrlTestImages);
    demos::ensure_file(root / "t10k-labels-idx1-ubyte", kUrlTestLabels);

    demos::print_separator("Load + split");
    demos::IdxImages train_img = demos::read_idx_images(root / "train-images-idx3-ubyte");
    demos::IdxLabels train_lbl = demos::read_idx_labels(root / "train-labels-idx1-ubyte");
    demos::IdxImages test_img = demos::read_idx_images(root / "t10k-images-idx3-ubyte");
    demos::IdxLabels test_lbl = demos::read_idx_labels(root / "t10k-labels-idx1-ubyte");

    const int need_from_train = cfg.train_n + cfg.val_n;
    if (need_from_train > train_img.count || cfg.test_n > test_img.count) {
        cerr << "Requested split larger than available Fashion-MNIST.\n";
        return 1;
    }

    // 官方 train → 再切 train/val；官方 test → 只当最终 test
    vector<int> pool = shuffled_indices(train_img.count, 41);
    vector<int> train_idx(pool.begin(), pool.begin() + cfg.train_n);
    vector<int> val_idx(pool.begin() + cfg.train_n,
                        pool.begin() + cfg.train_n + cfg.val_n);
    vector<int> test_idx = shuffled_indices(test_img.count, 42);
    test_idx.resize(cfg.test_n);

    Matrix train_x, train_y, val_x, val_y, test_x, test_y;
    demos::materialize_subset(train_img, train_lbl, train_idx, kNumClasses, train_x, train_y);
    demos::materialize_subset(train_img, train_lbl, val_idx, kNumClasses, val_x, val_y);
    demos::materialize_subset(test_img, test_lbl, test_idx, kNumClasses, test_x, test_y);
    shuffle_pair(train_x, train_y, 43);

    cout << "split: train=" << cfg.train_n
         << "  val=" << cfg.val_n
         << "  test=" << cfg.test_n << "\n";

    const int maj = majority_class(train_y);
    const double maj_val = constant_accuracy(val_y, maj);
    const double maj_test = constant_accuracy(test_y, maj);
    cout << "majority baseline (always predict '" << kClassNames[maj] << "'):\n"
         << "  val_acc=" << setprecision(2) << maj_val * 100.0 << "%"
         << "  test_acc=" << maj_test * 100.0 << "%\n"
         << setprecision(6)
         << "  (Fashion is ~balanced → baseline ≈ 10%. Model must beat this.)\n";

    auto mlp = demos::build_mlp(/*in*/ 784, /*hidden*/ {128, 64}, /*out*/ 10,
                                "relu", "identity", "softmax_ce", cfg.learning_rate);
    mlp->print_architecture(/*verbose=*/false);

    demos::print_separator("Training (log train vs val)");
    cout << fixed << setprecision(6);
    cout << "epochs=" << cfg.epochs
         << ", batch_size=" << cfg.batch_size
         << ", lr=" << cfg.learning_rate;
    if (cfg.patience >= 0) {
        cout << ", early_stop patience=" << cfg.patience
             << " (monitor=val_acc, restore best)";
    }
    cout << "\n"
         << "  '*' marks a new best val_acc\n\n";
    cout << setw(6) << "epoch"
         << " | " << setw(10) << "train_loss"
         << " | " << setw(9) << "train_acc"
         << " | " << setw(10) << "val_loss"
         << " | " << setw(9) << "val_acc"
         << " |" << "\n";
    cout << string(60, '-') << "\n";

    const fs::path csv = curve_path(cfg.name);
    ofstream csv_out(csv);
    csv_out << "epoch,train_loss,train_acc,val_loss,val_acc,is_best\n";

    double best_val_acc = -1.0;
    int best_epoch = -1;
    int stale = 0;
    int last_epoch = -1;
    bool stopped_early = false;
    Mlp::ParamSnapshot best_params = mlp->capture_params();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        last_epoch = epoch;
        for (int start = 0; start < cfg.train_n; start += cfg.batch_size) {
            int end = min(start + cfg.batch_size, cfg.train_n);
            mlp->train(train_x.get_rows(start, end), train_y.get_rows(start, end));
        }

        Metrics tr = eval_split(*mlp, train_x, train_y);
        Metrics va = eval_split(*mlp, val_x, val_y);
        bool is_best = false;
        if (va.acc > best_val_acc) {
            best_val_acc = va.acc;
            best_epoch = epoch;
            best_params = mlp->capture_params();
            stale = 0;
            is_best = true;
        } else if (cfg.patience >= 0) {
            ++stale;
        }

        cout << setw(6) << epoch
             << " | " << setw(10) << tr.loss
             << " | " << setw(8) << setprecision(2) << tr.acc * 100.0 << "%"
             << " | " << setw(10) << setprecision(6) << va.loss
             << " | " << setw(8) << setprecision(2) << va.acc * 100.0 << "%"
             << setprecision(6)
             << " |" << (is_best ? " *" : "") << "\n";

        csv_out << epoch << "," << tr.loss << "," << tr.acc << ","
                << va.loss << "," << va.acc << "," << (is_best ? 1 : 0) << "\n";

        if (cfg.patience >= 0 && stale >= cfg.patience) {
            stopped_early = true;
            cout << "\nEarly stop: no val_acc improvement for " << cfg.patience
                 << " epochs (best @ epoch " << best_epoch << ").\n";
            break;
        }
    }
    csv_out.close();

    if (cfg.patience >= 0) {
        // 早停：最终模型必须是 best-val 权重，不是停下来那一刻已经变坏的权重
        mlp->restore_params(best_params);
    }

    cout << "\nWrote learning curve: " << csv << "\n";
    cout << "Best val_acc=" << setprecision(2) << best_val_acc * 100.0
         << "% at epoch " << best_epoch
         << " (ran " << (last_epoch + 1) << "/" << cfg.epochs << " epochs";
    if (stopped_early) {
        cout << ", stopped early; weights restored to best)";
    } else if (cfg.patience >= 0) {
        cout << "; weights restored to best)";
    } else {
        cout << "; final eval uses last-epoch weights)";
    }
    cout << setprecision(6) << "\n";

    // 只在这里碰 test
    demos::print_separator(
        cfg.patience >= 0 ? "Final test (once, on best-val weights)"
                          : "Final test (once, on last-epoch weights)");
    Metrics te = eval_split(*mlp, test_x, test_y);
    cout << "test_loss=" << te.loss
         << "  test_acc=" << setprecision(2) << te.acc * 100.0 << "%\n"
         << setprecision(6)
         << "Compare: best_val=" << setprecision(2) << best_val_acc * 100.0
         << "%  vs  test=" << te.acc * 100.0 << "%\n"
         << setprecision(6)
         << "(If you tuned hyperparameters on test, this number would be optimistic.)\n";

    Matrix val_pred = mlp->predict(val_x);
    print_per_class_recall(val_pred, val_y);
    cout << "\nNote: Shirt / T-shirt / Coat often drag recall down — "
            "overall acc hides that.\n";

    const string name = cfg.name;
    if (name == "overfit") {
        cout << "\nOverfit checklist: did train_acc keep climbing after val_acc peaked?\n"
             << "Next: make run-eval MODE=earlystop  — same setup, cut the wasted epochs.\n";
    } else if (name == "earlystop") {
        cout << "\nEarly-stop checklist: did training halt soon after val plateaued?\n"
             << "Compare curve_overfit.csv vs curve_earlystop.csv — same peak, fewer epochs.\n";
    } else {
        cout << "\nNext: make run-eval MODE=overfit  — same net, starved data.\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    string mode = "normal";
    if (argc >= 2) {
        mode = argv[1];
    }

    if (mode == "-h" || mode == "--help" || mode == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (mode == "list") {
        cout << "normal\noverfit\nearlystop\n";
        return 0;
    }
    if (mode == "normal") {
        return run_mode(kNormal);
    }
    if (mode == "overfit") {
        return run_mode(kOverfit);
    }
    if (mode == "earlystop") {
        return run_mode(kEarlyStop);
    }

    cerr << "Unknown mode: " << mode << "\n\n";
    print_usage(argv[0]);
    return 2;
}
