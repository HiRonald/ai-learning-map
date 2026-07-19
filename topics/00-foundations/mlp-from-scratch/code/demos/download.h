#pragma once

#include <filesystem>
#include <string>

namespace demos {

// 数据根目录：优先环境变量 MLP_DATA_DIR，否则用相对仓库的默认路径
std::filesystem::path data_root();

// 若 dest 不存在，则 curl 下载 url 到 dest.gz 再 gunzip
void ensure_file(const std::filesystem::path& dest, const std::string& url);

}  // namespace demos
