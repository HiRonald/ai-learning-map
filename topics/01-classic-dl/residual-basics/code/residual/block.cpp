#include "residual/block.h"

#include "nn/activator.h"

#include <stdexcept>

ResidualBlock::ResidualBlock(int dim, double learning_rate)
    : dim_(dim),
      f1_(std::make_shared<Layer>(dim, dim, std::make_shared<Activator>("relu"), learning_rate)),
      f2_(std::make_shared<Layer>(dim, dim, std::make_shared<Activator>("identity"),
                                 learning_rate)) {
    if (dim <= 0) {
        throw std::invalid_argument("ResidualBlock: dim must be positive");
    }
    // 近零初始化残差支路末端：起步接近 y≈x，避免深堆叠爆炸（ResNet 常见做法）
    Matrix w2 = f2_->get_weights();
    w2.fill(0.0);
    f2_->set_weights(w2);
}

Matrix ResidualBlock::forward(const Matrix& input) {
    if (input.get_col_number() != dim_) {
        throw std::invalid_argument("ResidualBlock::forward dim mismatch");
    }
    input_ = input;
    Matrix h = f1_->forward(input);
    Matrix delta = f2_->forward(h);
    return input + delta;
}

Matrix ResidualBlock::backward(const Matrix& upstream_grad) {
    if (upstream_grad.get_row_number() != input_.get_row_number() ||
        upstream_grad.get_col_number() != dim_) {
        throw std::invalid_argument("ResidualBlock::backward shape mismatch");
    }
    // y = x + F(x)  ⇒  ∂L/∂x = ∂L/∂y + ∂L/∂F
    Matrix g_f = f2_->backward(upstream_grad);
    g_f = f1_->backward(g_f);
    return upstream_grad + g_f;
}

int ResidualBlock::dim() const { return dim_; }

int ResidualBlock::param_count() const {
    // 两层各 dim*dim + dim
    return 2 * (dim_ * dim_ + dim_);
}
