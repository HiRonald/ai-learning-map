#pragma once

#include "cnn/conv2d.h"
#include "cnn/pool2d.h"
#include "nn/layer.h"
#include "nn/matrix.h"

#include <memory>
#include <vector>

// LeNet-5 风格（Fashion / MNIST 28×28 → 零填到 32×32）：
//   Conv5→Pool → Conv5→Pool → FC120→FC84→FC10
// 原论文用 tanh + 平均池化；这里用 ReLU + MaxPool，和本仓库其它主题一致。
class LeNet {
public:
    explicit LeNet(double learning_rate);

    void train(const Matrix& images_flat, const Matrix& labels_one_hot);
    Matrix predict(const Matrix& images_flat);
    float eval(const Matrix& labels_one_hot) const;

    void print_architecture() const;
    int param_count() const;

    static constexpr int kImageSide = 28;
    static constexpr int kPaddedSide = 32;
    static constexpr int kNumClasses = 10;
    static constexpr int kFlatFeats = 16 * 5 * 5;  // 400

private:
    double lr_;
    Conv2d conv1_;  // 1→6, k=5, p=0 on 32×32 → 28×28
    MaxPool2d pool1_;
    Conv2d conv2_;  // 6→16, k=5 → 10×10
    MaxPool2d pool2_;
    std::shared_ptr<Layer> fc1_;  // 400→120
    std::shared_ptr<Layer> fc2_;  // 120→84
    std::shared_ptr<Layer> fc3_;  // 84→10

    std::vector<char> relu1_mask_;
    std::vector<char> relu2_mask_;
    Matrix logits_;
    float last_loss_ = 0.0f;

    void forward(const Matrix& images_flat);
    void backward(const Matrix& labels_one_hot);

    static Tensor pad_to_32(const Tensor& x28);
    static int dense_params(int in_f, int out_f);
};
