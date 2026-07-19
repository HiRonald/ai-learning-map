#include "demos/idx_io.h"

#include <fstream>
#include <stdexcept>

namespace demos {

namespace {

std::uint32_t read_be_u32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading IDX header");
    }
    return (static_cast<std::uint32_t>(b[0]) << 24) |
           (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) |
           static_cast<std::uint32_t>(b[3]);
}

}  // namespace

IdxImages read_idx_images(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    const std::uint32_t magic = read_be_u32(in);
    if (magic != 2051) {
        throw std::runtime_error("bad image magic in " + path.string());
    }
    IdxImages out;
    out.count = static_cast<int>(read_be_u32(in));
    out.rows = static_cast<int>(read_be_u32(in));
    out.cols = static_cast<int>(read_be_u32(in));
    const std::size_t n = static_cast<std::size_t>(out.count) * out.rows * out.cols;
    out.pixels.resize(n);
    in.read(reinterpret_cast<char*>(out.pixels.data()), static_cast<std::streamsize>(n));
    if (!in) {
        throw std::runtime_error("unexpected EOF in image payload: " + path.string());
    }
    return out;
}

IdxLabels read_idx_labels(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    const std::uint32_t magic = read_be_u32(in);
    if (magic != 2049) {
        throw std::runtime_error("bad label magic in " + path.string());
    }
    IdxLabels out;
    out.count = static_cast<int>(read_be_u32(in));
    out.labels.resize(static_cast<std::size_t>(out.count));
    in.read(reinterpret_cast<char*>(out.labels.data()), out.count);
    if (!in) {
        throw std::runtime_error("unexpected EOF in label payload: " + path.string());
    }
    return out;
}

void materialize_subset(const IdxImages& images,
                        const IdxLabels& labels,
                        const std::vector<int>& indices,
                        int num_classes,
                        Matrix& out_x,
                        Matrix& out_y) {
    if (images.count != labels.count) {
        throw std::runtime_error("image/label count mismatch");
    }
    const int feat = images.rows * images.cols;
    const int n = static_cast<int>(indices.size());
    out_x = Matrix(n, feat);
    out_y = Matrix(n, num_classes);
    out_y.fill(0.0);

    for (int i = 0; i < n; ++i) {
        const int src = indices[static_cast<std::size_t>(i)];
        if (src < 0 || src >= images.count) {
            throw std::runtime_error("subset index out of range");
        }
        const std::size_t base =
            static_cast<std::size_t>(src) * static_cast<std::size_t>(feat);
        for (int f = 0; f < feat; ++f) {
            out_x.set_data(i, f, images.pixels[base + static_cast<std::size_t>(f)] / 255.0);
        }
        const int cls = static_cast<int>(labels.labels[static_cast<std::size_t>(src)]);
        if (cls < 0 || cls >= num_classes) {
            throw std::runtime_error("label out of range");
        }
        out_y.set_data(i, cls, 1.0);
    }
}

}  // namespace demos
