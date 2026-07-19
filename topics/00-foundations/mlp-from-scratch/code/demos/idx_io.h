#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "nn/matrix.h"

namespace demos {

struct IdxImages {
    int count = 0;
    int rows = 0;
    int cols = 0;
    std::vector<std::uint8_t> pixels;  // count * rows * cols，行优先
};

struct IdxLabels {
    int count = 0;
    std::vector<std::uint8_t> labels;
};

IdxImages read_idx_images(const std::filesystem::path& path);
IdxLabels read_idx_labels(const std::filesystem::path& path);

// 把选中的样本打成 Matrix：像素 /255 → [0,1]；标签 → one-hot
void materialize_subset(const IdxImages& images,
                        const IdxLabels& labels,
                        const std::vector<int>& indices,
                        int num_classes,
                        Matrix& out_x,
                        Matrix& out_y);

}  // namespace demos
