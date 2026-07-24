#include <iostream>
#include <string>

#include "demos/fashion_demo.h"
#include "demos/identity_demo.h"

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [demo]\n"
        << "\n"
        << "Demos:\n"
        << "  identity  Deep plain vs residual on y=x (default)\n"
        << "  fashion   Fashion: plain CNN vs residual CNN trunk\n"
        << "  list\n"
        << "\n"
        << "Examples:\n"
        << "  make run-residual\n"
        << "  make run-residual DEMO=fashion\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string demo = "identity";
    if (argc >= 2) {
        demo = argv[1];
    }

    if (demo == "-h" || demo == "--help" || demo == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (demo == "list") {
        std::cout << "identity\nfashion\n";
        return 0;
    }
    if (demo == "identity") {
        return demos::run_identity();
    }
    if (demo == "fashion") {
        return demos::run_fashion();
    }

    std::cerr << "Unknown demo: " << demo << "\n\n";
    print_usage(argv[0]);
    return 2;
}
