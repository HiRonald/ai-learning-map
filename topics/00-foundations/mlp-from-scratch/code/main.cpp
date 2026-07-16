#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cmath>

#include "mlp.h"

using namespace std;

namespace {

void print_separator(const string& title) {
    cout << "\n========== " << title << " ==========\n";
}

// 将连续输出阈值化为 0/1，便于观察 XOR 分类效果
Matrix binarize(const Matrix& pred, double threshold = 0.5) {
    Matrix out(pred.get_row_number(), pred.get_col_number());
    for (int i = 0; i < pred.get_row_number(); ++i) {
        for (int j = 0; j < pred.get_col_number(); ++j) {
            out.set_data(i, j, pred.get_data(i, j) >= threshold ? 1.0 : 0.0);
        }
    }
    return out;
}

double accuracy(const Matrix& pred, const Matrix& labels) {
    Matrix hard = binarize(pred);
    int correct = 0;
    int total = labels.get_row_number() * labels.get_col_number();
    for (int i = 0; i < labels.get_row_number(); ++i) {
        for (int j = 0; j < labels.get_col_number(); ++j) {
            if (std::abs(hard.get_data(i, j) - labels.get_data(i, j)) < 1e-9) {
                ++correct;
            }
        }
    }
    return static_cast<double>(correct) / total;
}

}  // namespace

int main() {
    print_separator("MLP XOR Demo");
    cout << "Goal: learn f(x1, x2) = x1 XOR x2 with a small fully-connected network.\n";

    // XOR 真值表：4 个样本，每行是一个样本
    Matrix data(4, 2);
    data.set_data({0, 0,
                   0, 1,
                   1, 0,
                   1, 1});
    Matrix labels(4, 1);
    labels.set_data({0, 1, 1, 0});

    data.print("Training inputs");
    labels.print("Training labels");

    // 网络结构：2 -> 4 -> 1
    // 隐藏层用 sigmoid；输出层也用 sigmoid，把输出压到 (0,1)，适合 0/1 目标
    const int input_size = 2;
    const int output_size = 1;
    const vector<int> hidden_sizes = {4};
    const double learning_rate = 1.0;

    vector<shared_ptr<Layer>> layers;
    int prev_size = input_size;
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        auto act = make_shared<Activator>("sigmoid");
        layers.push_back(make_shared<Layer>(prev_size, hidden_sizes[i], act, learning_rate));
        prev_size = hidden_sizes[i];
    }
    // 输出层单独构造，便于以后换成 identity / softmax 等
    auto output_act = make_shared<Activator>("sigmoid");
    layers.push_back(make_shared<Layer>(prev_size, output_size, output_act, learning_rate));

    Mlp mlp(layers, "mse");
    mlp.print_architecture();

    print_separator("Training");
    const int epoch_number = 5000;
    const int batch_size = 4;       // 全批量：每次用全部 4 个样本
    const int log_interval = 500;   // 每隔多少 epoch 打印一次

    cout << fixed << setprecision(6);
    cout << "epochs=" << epoch_number
         << ", batch_size=" << batch_size
         << ", lr=" << learning_rate
         << ", loss=mse\n";

    for (int epoch = 0; epoch < epoch_number; ++epoch) {
        for (int start = 0; start < data.get_row_number(); start += batch_size) {
            int end = min(start + batch_size, data.get_row_number());
            Matrix batch_x = data.get_rows(start, end);
            Matrix batch_y = labels.get_rows(start, end);
            mlp.train(batch_x, batch_y);
        }

        if (epoch % log_interval == 0 || epoch + 1 == epoch_number) {
            Matrix pred = mlp.predict(data);
            float loss = mlp.eval(labels);
            double acc = accuracy(pred, labels);
            cout << "Epoch " << setw(4) << epoch
                 << " | loss=" << loss
                 << " | acc=" << setprecision(2) << acc * 100.0 << "%"
                 << setprecision(6) << "\n";
        }
    }

    print_separator("Inference");
    Matrix predictions = mlp.predict(data);
    Matrix hard_labels = binarize(predictions);
    float final_loss = mlp.eval(labels);
    double final_acc = accuracy(predictions, labels);

    cout << "Final loss=" << final_loss
         << ", accuracy=" << setprecision(2) << final_acc * 100.0 << "%\n"
         << setprecision(6);

    cout << "\nSample-wise results:\n";
    cout << "  x1  x2  |  label  |  pred(raw)  |  pred(0/1)\n";
    cout << "  --------|---------|-------------|-----------\n";
    for (int i = 0; i < data.get_row_number(); ++i) {
        cout << "  " << data.get_data(i, 0) << "   " << data.get_data(i, 1)
             << "  |   " << labels.get_data(i, 0)
             << "   |   " << setw(8) << predictions.get_data(i, 0)
             << "  |    " << hard_labels.get_data(i, 0) << "\n";
    }

    if (final_acc >= 0.99) {
        cout << "\nXOR learned successfully.\n";
    } else {
        cout << "\nTraining finished, but accuracy is not perfect yet."
             << " Try more epochs or tune learning rate.\n";
    }

    return 0;
}
