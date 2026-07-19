#include "demos/download.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace demos {

fs::path data_root() {
    if (const char* env = std::getenv("MLP_DATA_DIR")) {
        return fs::path(env);
    }
    // make run 会注入 MLP_DATA_DIR；直接跑二进制时默认落在主题 data/
    return fs::path("topics/00-foundations/mlp-from-scratch/data");
}

namespace {

bool usable_file(const fs::path& path) {
    return fs::exists(path) && fs::is_regular_file(path) && fs::file_size(path) > 0;
}

void run_or_throw(const std::string& cmd, const std::string& what) {
    std::cout << "  $ " << cmd << "\n";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error(what + " failed (exit " + std::to_string(rc) + ")");
    }
}

}  // namespace

void ensure_file(const fs::path& dest, const std::string& url) {
    if (usable_file(dest)) {
        std::cout << "  cached: " << dest << "\n";
        return;
    }

    fs::create_directories(dest.parent_path());
    const fs::path gz = dest.string() + ".gz";
    if (fs::exists(gz)) {
        fs::remove(gz);
    }

    std::cout << "  downloading: " << url << "\n";
    // -f: HTTP 错误当失败；-L: 跟随重定向
    const std::string curl_cmd =
        "curl -fsSL --retry 3 --retry-delay 1 -o \"" + gz.string() + "\" \"" + url + "\"";
    run_or_throw(curl_cmd, "download " + url);

    const std::string gunzip_cmd = "gunzip -f \"" + gz.string() + "\"";
    run_or_throw(gunzip_cmd, "gunzip " + gz.string());

    if (!usable_file(dest)) {
        throw std::runtime_error("expected file missing after download: " + dest.string());
    }
    std::cout << "  ready: " << dest << "\n";
}

}  // namespace demos
