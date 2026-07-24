#include "demos/param_demo.h"

#include "cnn/net.h"
#include "demos/common.h"

#include <iomanip>
#include <iostream>

using namespace std;

namespace {

long long dense_params(long long in_feats, long long out_feats) {
    return in_feats * out_feats + out_feats;  // W + b
}

long long conv_params(long long in_c, long long out_c, long long k) {
    return out_c * (in_c * k * k + 1);
}

}  // namespace

namespace demos {

int run_param() {
    print_separator("Demo: param");
    cout << "Goal: quantify weight sharing — same spatial map, Conv vs Dense params.\n";

    const int h = 28;
    const int w = 28;
    const int in_c = 1;
    const int out_c = 8;
    const int k = 3;

    const long long spatial = static_cast<long long>(h) * w;
    const long long in_feats = spatial * in_c;
    const long long out_feats = spatial * out_c;  // same-size map, like pad=1 stride=1

    const long long fc = dense_params(in_feats, out_feats);
    const long long conv = conv_params(in_c, out_c, k);

    cout << fixed;
    cout << "\nSetup: map " << in_c << "x" << h << "x" << w
         << "  →  " << out_c << "x" << h << "x" << w
         << "  (same spatial size)\n\n";

    cout << "Dense (every input pixel → every output location):\n"
         << "  params = in_feats * out_feats + out_feats\n"
         << "         = " << in_feats << " * " << out_feats << " + " << out_feats
         << "  = " << fc << "\n\n";

    cout << "Conv2d (shared " << k << "x" << k << " kernel per out channel):\n"
         << "  params = out_c * (in_c * k * k + 1)\n"
         << "         = " << out_c << " * (" << in_c << " * " << k << "*" << k << " + 1)"
         << "  = " << conv << "\n\n";

    const double ratio = static_cast<double>(fc) / static_cast<double>(conv);
    cout << setprecision(1) << "Ratio Dense / Conv ≈ " << ratio << "x\n"
         << setprecision(6);

    print_separator("Our fashion LeNet");
    LeNet net(0.1);
    net.print_architecture();

    const long long mlp_ref = dense_params(784, 128) + dense_params(128, 64) + dense_params(64, 10);
    cout << "\nReference MLP from mlp-from-scratch fashion demo:\n"
         << "  784→128→64→10  params ≈ " << mlp_ref << "\n"
         << "  LeNet total   params ≈ " << net.param_count() << "\n"
         << "  (LeNet FC head 120→84→10 still dominates params; conv stem is tiny.)\n";

    cout << "\nTakeaway: local connectivity + weight sharing is the inductive bias for grids.\n"
         << "Next: `fashion` — same-subset MLP vs LeNet.\n";
    return 0;
}

}  // namespace demos
