#include "demos/common.h"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace std;

namespace demos {

void print_separator(const string& title) {
    cout << "\n========== " << title << " ==========\n";
}

Matrix binarize(const Matrix& pred, double threshold) {
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

int argmax_row(const Matrix& m, int row) {
    int best = 0;
    double best_v = m.get_data(row, 0);
    for (int j = 1; j < m.get_col_number(); ++j) {
        double v = m.get_data(row, j);
        if (v > best_v) {
            best_v = v;
            best = j;
        }
    }
    return best;
}

double argmax_accuracy(const Matrix& logits, const Matrix& one_hot_labels) {
    int correct = 0;
    const int n = logits.get_row_number();
    for (int i = 0; i < n; ++i) {
        if (argmax_row(logits, i) == argmax_row(one_hot_labels, i)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / n;
}

double mean_abs_error(const Matrix& pred, const Matrix& labels) {
    double total = 0.0;
    const int n = pred.get_row_number() * pred.get_col_number();
    for (int i = 0; i < pred.get_row_number(); ++i) {
        for (int j = 0; j < pred.get_col_number(); ++j) {
            total += std::abs(pred.get_data(i, j) - labels.get_data(i, j));
        }
    }
    return total / n;
}

shared_ptr<Mlp> build_mlp(int input_size,
                          const vector<int>& hidden_sizes,
                          int output_size,
                          const string& hidden_act,
                          const string& output_act,
                          const string& loss_type,
                          double learning_rate) {
    vector<shared_ptr<Layer>> layers;
    int prev = input_size;
    for (int h : hidden_sizes) {
        layers.push_back(make_shared<Layer>(
            prev, h, make_shared<Activator>(hidden_act), learning_rate));
        prev = h;
    }
    layers.push_back(make_shared<Layer>(
        prev, output_size, make_shared<Activator>(output_act), learning_rate));
    return make_shared<Mlp>(layers, loss_type);
}

void train_classifier(Mlp& mlp,
                      const Matrix& data,
                      const Matrix& labels,
                      int epochs,
                      int batch_size,
                      int log_interval) {
    cout << fixed << setprecision(6);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < data.get_row_number(); start += batch_size) {
            int end = min(start + batch_size, data.get_row_number());
            mlp.train(data.get_rows(start, end), labels.get_rows(start, end));
        }

        if (epoch % log_interval == 0 || epoch + 1 == epochs) {
            Matrix pred = mlp.predict(data);
            float loss = mlp.eval(labels);
            double acc = accuracy(pred, labels);
            cout << "Epoch " << setw(4) << epoch
                 << " | loss=" << loss
                 << " | acc=" << setprecision(2) << acc * 100.0 << "%"
                 << setprecision(6) << "\n";
        }
    }
}

}  // namespace demos
