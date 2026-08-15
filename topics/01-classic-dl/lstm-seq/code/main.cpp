#include <iostream>
#include <string>

#include "demos/gates_demo.h"
#include "demos/recall_demo.h"

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [demo]\n"
        << "\n"
        << "Demos:\n"
        << "  gates   Fixed cell: print f/i/o and c_t along a short sequence (default)\n"
        << "  recall  Delayed recall: LSTM vs Vanilla RNN on synthetic long dependency\n"
        << "  list\n"
        << "\n"
        << "Examples:\n"
        << "  make run-lstm\n"
        << "  make run-lstm DEMO=recall\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string demo = "gates";
    if (argc >= 2) {
        demo = argv[1];
    }

    if (demo == "-h" || demo == "--help" || demo == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (demo == "list") {
        std::cout << "gates\nrecall\n";
        return 0;
    }
    if (demo == "gates") {
        return demos::run_gates();
    }
    if (demo == "recall") {
        return demos::run_recall();
    }

    std::cerr << "Unknown demo: " << demo << "\n\n";
    print_usage(argv[0]);
    return 2;
}
