#pragma once

#include "cnn/conv2d.h"
#include "cnn/pool2d.h"
#include "nn/layer.h"
#include "nn/matrix.h"
#include "residual/conv_block.h"

#include <memory>
#include <vector>

// 小 CNN：stem(Conv→ReLU→Pool) → trunk(plain Conv 堆叠 或 residual 块) → Pool → FC
// plain / residual 对齐 trunk 里的 Conv 层数（1 residual block = 2 Conv）。
class DeepCnn {
public:
    enum class Mode { Plain, Residual };

    DeepCnn(int channels, int num_res_blocks, Mode mode, double learning_rate);

    void train(const Matrix& images_flat, const Matrix& labels_one_hot);
    Matrix predict(const Matrix& images_flat);
    float eval(const Matrix& labels_one_hot) const;

    void print_architecture() const;
    int param_count() const;
    const char* mode_name() const;

    static constexpr int kImageSide = 28;
    static constexpr int kNumClasses = 10;

private:
    Mode mode_;
    double lr_;
    int channels_;
    int num_res_blocks_;
    int plain_convs_;  // = 2 * num_res_blocks when plain

    std::unique_ptr<Conv2d> stem_;
    MaxPool2d pool1_;
    std::vector<std::unique_ptr<Conv2d>> plain_convs_list_;
    std::vector<std::unique_ptr<ResidualConvBlock>> res_blocks_;
    MaxPool2d pool2_;
    std::shared_ptr<Layer> head_;

    std::vector<char> stem_relu_mask_;
    std::vector<std::vector<char>> plain_relu_masks_;

    Matrix logits_;
    float last_loss_ = 0.0f;
    int flat_feats_ = 0;  // channels * 7 * 7

    void forward(const Matrix& images_flat);
    void backward(const Matrix& labels_one_hot);
};
