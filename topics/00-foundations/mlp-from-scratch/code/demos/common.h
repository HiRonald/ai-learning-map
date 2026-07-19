#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nn/mlp.h"

namespace demos {

void print_separator(const std::string& title);

// 将连续输出阈值化为 0/1
Matrix binarize(const Matrix& pred, double threshold = 0.5);

// 二分类准确率（阈值 0.5）
double accuracy(const Matrix& pred, const Matrix& labels);

// 多类准确率：对 logits / 概率按行 argmax，与 one-hot 标签比较
double argmax_accuracy(const Matrix& logits, const Matrix& one_hot_labels);

int argmax_row(const Matrix& m, int row);

// 回归平均绝对误差
double mean_abs_error(const Matrix& pred, const Matrix& labels);

// 按 hidden + 独立输出层拼网络
std::shared_ptr<Mlp> build_mlp(int input_size,
                               const std::vector<int>& hidden_sizes,
                               int output_size,
                               const std::string& hidden_act,
                               const std::string& output_act,
                               const std::string& loss_type,
                               double learning_rate);

// 通用分类训练循环：按 batch 扫 epoch，定期打 loss / acc
void train_classifier(Mlp& mlp,
                      const Matrix& data,
                      const Matrix& labels,
                      int epochs,
                      int batch_size,
                      int log_interval);

}  // namespace demos
