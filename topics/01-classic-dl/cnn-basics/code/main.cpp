#include <iostream>
#include <string>

#include "demos/fashion_demo.h"
#include "demos/filter_demo.h"
#include "demos/param_demo.h"

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [demo]\n"
        << "\n"
        << "Demos:\n"
        << "  filter    Fixed kernels sliding on a toy image (default)\n"
        << "  param     Dense vs Conv parameter count\n"
        << "  fashion   Fashion-MNIST tiny CNN (reuses MLP data cache)\n"
        << "  list      Print available demos\n"
        << "\n"
        << "Examples:\n"
        << "  make run-cnn\n"
        << "  make run-cnn DEMO=param\n"
        << "  make run-cnn DEMO=fashion\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string demo = "filter";
    if (argc >= 2) {
        demo = argv[1];
    }

    if (demo == "-h" || demo == "--help" || demo == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (demo == "list") {
        std::cout << "filter\nparam\nfashion\n";
        return 0;
    }
    if (demo == "filter") {
        return demos::run_filter();
    }
    if (demo == "param") {
        return demos::run_param();
    }
    if (demo == "fashion") {
        return demos::run_fashion();
    }

    std::cerr << "Unknown demo: " << demo << "\n\n";
    print_usage(argv[0]);
    return 2;
}
