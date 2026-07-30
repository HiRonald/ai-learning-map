#include <iostream>
#include <string>

#include "demos/temp_demo.h"
#include "demos/unroll_demo.h"

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [demo]\n"
        << "\n"
        << "Demos:\n"
        << "  unroll  Fixed cell: print h_t along a short sequence (default)\n"
        << "  temp    Daily min temperatures: Vanilla RNN next-day forecast\n"
        << "  list\n"
        << "\n"
        << "Examples:\n"
        << "  make run-rnn\n"
        << "  make run-rnn DEMO=temp\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string demo = "unroll";
    if (argc >= 2) {
        demo = argv[1];
    }

    if (demo == "-h" || demo == "--help" || demo == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (demo == "list") {
        std::cout << "unroll\ntemp\n";
        return 0;
    }
    if (demo == "unroll") {
        return demos::run_unroll();
    }
    if (demo == "temp") {
        return demos::run_temp();
    }

    std::cerr << "Unknown demo: " << demo << "\n\n";
    print_usage(argv[0]);
    return 2;
}
