#include <iostream>
#include <string>

#include "demos/fashion.h"
#include "demos/sine.h"
#include "demos/xor.h"

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [demo]\n"
        << "\n"
        << "Demos:\n"
        << "  xor       Smoke test: 4-sample XOR (default)\n"
        << "  sine      Regression: fit y = sin(x)\n"
        << "  fashion   Fashion-MNIST (auto-download) + ASCII gallery\n"
        << "  list      Print available demos\n"
        << "\n"
        << "Examples:\n"
        << "  make run\n"
        << "  make run DEMO=sine\n"
        << "  make run DEMO=fashion   # first run downloads ~30MB\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string demo = "xor";
    if (argc >= 2) {
        demo = argv[1];
    }

    if (demo == "-h" || demo == "--help" || demo == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (demo == "list") {
        std::cout << "xor\nsine\nfashion\n";
        return 0;
    }
    if (demo == "xor") {
        return demos::run_xor();
    }
    if (demo == "sine") {
        return demos::run_sine();
    }
    if (demo == "fashion") {
        return demos::run_fashion();
    }

    std::cerr << "Unknown demo: " << demo << "\n\n";
    print_usage(argv[0]);
    return 2;
}
