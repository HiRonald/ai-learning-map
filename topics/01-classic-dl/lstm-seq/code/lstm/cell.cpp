#include "lstm/cell.h"

#include "nn/activator.h"

#include <cmath>
#include <stdexcept>

namespace {

Matrix sigmoid_of(const Matrix& z) {
    static Activator act("sigmoid");
    return act.forward(z);
}

Matrix tanh_of(const Matrix& z) {
    static Activator act("tanh");
    return act.forward(z);
}

// σ'(z) = s(1-s)；这里直接用已算好的门值 s
Matrix sigmoid_grad_from_s(const Matrix& s) {
    Matrix ones(s.get_row_number(), s.get_col_number(), 1.0);
    return s.mul(ones - s);
}

Matrix tanh_grad_from_a(const Matrix& a) {
    Matrix ones(a.get_row_number(), a.get_col_number(), 1.0);
    return ones - a.mul(a);
}

Matrix slice_cols(const Matrix& m, int start, int width) {
    return m.get_cols(start, start + width);
}

}  // namespace

LstmCell::LstmCell(int input_size, int hidden_size, double learning_rate)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      lr_(learning_rate),
      w_x_(input_size, 4 * hidden_size),
      w_h_(hidden_size, 4 * hidden_size),
      b_(1, 4 * hidden_size, 0.0),
      grad_w_x_(input_size, 4 * hidden_size, 0.0),
      grad_w_h_(hidden_size, 4 * hidden_size, 0.0),
      grad_b_(1, 4 * hidden_size, 0.0) {
    if (input_size <= 0 || hidden_size <= 0) {
        throw std::invalid_argument("LstmCell: sizes must be positive");
    }
    if (lr_ <= 0.0) {
        throw std::invalid_argument("LstmCell: learning rate must be positive");
    }
    const int H = hidden_size_;
    const double lim_x = std::sqrt(6.0 / (input_size_ + 4 * H));
    const double lim_h = std::sqrt(6.0 / (H + 4 * H));
    w_x_.randomize(-lim_x, lim_x);
    w_h_.randomize(-lim_h, lim_h);
    // 遗忘门偏置偏正：一开始更愿意「留住」记忆（对 delayed recall 友好）
    for (int j = 0; j < H; ++j) {
        b_.set_data(0, j, 1.0);
    }
}

Matrix LstmCell::forward(const std::vector<Matrix>& xs) {
    if (xs.empty()) {
        throw std::invalid_argument("LstmCell::forward: empty sequence");
    }
    const int B = xs[0].get_row_number();
    const int H = hidden_size_;
    if (B <= 0) {
        throw std::invalid_argument("LstmCell::forward: batch must be positive");
    }

    xs_.clear();
    h_prevs_.clear();
    c_prevs_.clear();
    fs_.clear();
    is_.clear();
    gs_.clear();
    os_.clear();
    cs_.clear();
    hs_.clear();
    xs_.reserve(xs.size());
    h_prevs_.reserve(xs.size());
    c_prevs_.reserve(xs.size());
    fs_.reserve(xs.size());
    is_.reserve(xs.size());
    gs_.reserve(xs.size());
    os_.reserve(xs.size());
    cs_.reserve(xs.size());
    hs_.reserve(xs.size());

    Matrix h(B, H, 0.0);
    Matrix c(B, H, 0.0);
    for (const Matrix& x : xs) {
        if (x.get_row_number() != B || x.get_col_number() != input_size_) {
            throw std::invalid_argument("LstmCell::forward: x_t shape mismatch");
        }
        // pre = [z_f | z_i | z_g | z_o]
        Matrix pre = x.dot(w_x_).add(h.dot(w_h_)).add(b_);
        Matrix f = sigmoid_of(slice_cols(pre, 0, H));
        Matrix i = sigmoid_of(slice_cols(pre, H, H));
        Matrix g = tanh_of(slice_cols(pre, 2 * H, H));
        Matrix o = sigmoid_of(slice_cols(pre, 3 * H, H));
        Matrix c_next = f.mul(c).add(i.mul(g));
        Matrix h_next = o.mul(tanh_of(c_next));

        xs_.push_back(x);
        h_prevs_.push_back(h);
        c_prevs_.push_back(c);
        fs_.push_back(f);
        is_.push_back(i);
        gs_.push_back(g);
        os_.push_back(o);
        cs_.push_back(c_next);
        hs_.push_back(h_next);
        h = h_next;
        c = c_next;
    }
    return h;
}

Matrix LstmCell::backward(const Matrix& grad_h_last) {
    const int T = static_cast<int>(hs_.size());
    const int H = hidden_size_;
    if (T == 0) {
        throw std::runtime_error("LstmCell::backward: call forward first");
    }
    if (grad_h_last.get_row_number() != hs_.back().get_row_number() ||
        grad_h_last.get_col_number() != H) {
        throw std::invalid_argument("LstmCell::backward: grad_h_last shape mismatch");
    }

    Matrix dh_next = grad_h_last;
    Matrix dc_next(grad_h_last.get_row_number(), H, 0.0);

    for (int t = T - 1; t >= 0; --t) {
        const Matrix& f = fs_[static_cast<size_t>(t)];
        const Matrix& i = is_[static_cast<size_t>(t)];
        const Matrix& g = gs_[static_cast<size_t>(t)];
        const Matrix& o = os_[static_cast<size_t>(t)];
        const Matrix& c = cs_[static_cast<size_t>(t)];
        const Matrix& c_prev = c_prevs_[static_cast<size_t>(t)];
        const Matrix& h_prev = h_prevs_[static_cast<size_t>(t)];
        const Matrix& x = xs_[static_cast<size_t>(t)];

        Matrix tanh_c = tanh_of(c);
        // h = o ⊙ tanh(c)
        Matrix do_t = dh_next.mul(tanh_c);
        Matrix dc = dc_next.add(dh_next.mul(o).mul(tanh_grad_from_a(tanh_c)));

        // c = f ⊙ c_prev + i ⊙ g
        Matrix df = dc.mul(c_prev);
        Matrix di = dc.mul(g);
        Matrix dg = dc.mul(i);
        Matrix dc_prev = dc.mul(f);

        Matrix dz_f = df.mul(sigmoid_grad_from_s(f));
        Matrix dz_i = di.mul(sigmoid_grad_from_s(i));
        Matrix dz_g = dg.mul(tanh_grad_from_a(g));
        Matrix dz_o = do_t.mul(sigmoid_grad_from_s(o));

        // 拼回 (B, 4H)
        const int B = dz_f.get_row_number();
        Matrix dpre(B, 4 * H);
        for (int r = 0; r < B; ++r) {
            for (int j = 0; j < H; ++j) {
                dpre.set_data(r, j, dz_f.get_data(r, j));
                dpre.set_data(r, H + j, dz_i.get_data(r, j));
                dpre.set_data(r, 2 * H + j, dz_g.get_data(r, j));
                dpre.set_data(r, 3 * H + j, dz_o.get_data(r, j));
            }
        }

        grad_w_x_ = grad_w_x_ + x.transpose().dot(dpre);
        grad_w_h_ = grad_w_h_ + h_prev.transpose().dot(dpre);
        grad_b_ = grad_b_ + dpre.sum_rows();

        dh_next = dpre.dot(w_h_.transpose());
        dc_next = dc_prev;
    }
    return dh_next;  // ∂L/∂h_0
}

void LstmCell::zero_grad() {
    grad_w_x_.fill(0.0);
    grad_w_h_.fill(0.0);
    grad_b_.fill(0.0);
}

void LstmCell::apply_gradients() {
    w_x_ = w_x_ - grad_w_x_.mul(lr_);
    w_h_ = w_h_ - grad_w_h_.mul(lr_);
    b_ = b_ - grad_b_.mul(lr_);
    zero_grad();
}

int LstmCell::input_size() const { return input_size_; }
int LstmCell::hidden_size() const { return hidden_size_; }

int LstmCell::param_count() const {
    const int H = hidden_size_;
    return input_size_ * 4 * H + H * 4 * H + 4 * H;
}

double LstmCell::learning_rate() const { return lr_; }

void LstmCell::set_learning_rate(double lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("LstmCell: learning rate must be positive");
    }
    lr_ = lr;
}

void LstmCell::set_weights_x(const Matrix& w) {
    if (w.get_row_number() != input_size_ || w.get_col_number() != 4 * hidden_size_) {
        throw std::invalid_argument("LstmCell::set_weights_x shape mismatch");
    }
    w_x_ = w;
}

void LstmCell::set_weights_h(const Matrix& w) {
    if (w.get_row_number() != hidden_size_ || w.get_col_number() != 4 * hidden_size_) {
        throw std::invalid_argument("LstmCell::set_weights_h shape mismatch");
    }
    w_h_ = w;
}

void LstmCell::set_bias(const Matrix& b) {
    if (b.get_row_number() != 1 || b.get_col_number() != 4 * hidden_size_) {
        throw std::invalid_argument("LstmCell::set_bias shape mismatch, expect (1, 4H)");
    }
    b_ = b;
}

const Matrix& LstmCell::weights_x() const { return w_x_; }
const Matrix& LstmCell::weights_h() const { return w_h_; }
const Matrix& LstmCell::bias() const { return b_; }
const std::vector<Matrix>& LstmCell::hidden_states() const { return hs_; }
const std::vector<Matrix>& LstmCell::cell_states() const { return cs_; }
const std::vector<Matrix>& LstmCell::forget_gates() const { return fs_; }
const std::vector<Matrix>& LstmCell::input_gates() const { return is_; }
const std::vector<Matrix>& LstmCell::output_gates() const { return os_; }
const std::vector<Matrix>& LstmCell::candidates() const { return gs_; }
