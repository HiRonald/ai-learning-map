#include "demos/xor.h"

#include "demos/common.h"

#include <iomanip>
#include <iostream>

using namespace std;

namespace demos {

int run_xor() {
    print_separator("Demo: XOR");
    cout << "Goal: learn f(x1, x2) = x1 XOR x2 with a tiny network.\n"
         << "Role: smoke test — 4 samples, proves nonlinear hidden layer is needed.\n";

    Matrix data(4, 2);
    data.set_data({0, 0,
                   0, 1,
                   1, 0,
                   1, 1});
    Matrix labels(4, 1);
    labels.set_data({0, 1, 1, 0});

    data.print("Training inputs");
    labels.print("Training labels");

    const double learning_rate = 1.0;
    auto mlp = build_mlp(/*in*/ 2, /*hidden*/ {4}, /*out*/ 1,
                         "sigmoid", "sigmoid", "mse", learning_rate);
    mlp->print_architecture();

    print_separator("Training");
    const int epochs = 5000;
    const int batch_size = 4;
    const int log_interval = 500;
    cout << "epochs=" << epochs
         << ", batch_size=" << batch_size
         << ", lr=" << learning_rate
         << ", loss=mse\n";

    train_classifier(*mlp, data, labels, epochs, batch_size, log_interval);

    print_separator("Inference");
    Matrix predictions = mlp->predict(data);
    Matrix hard = binarize(predictions);
    float final_loss = mlp->eval(labels);
    double final_acc = accuracy(predictions, labels);

    cout << fixed << setprecision(6);
    cout << "Final loss=" << final_loss
         << ", accuracy=" << setprecision(2) << final_acc * 100.0 << "%\n"
         << setprecision(6);

    cout << "\nSample-wise results:\n";
    cout << "  x1  x2  |  label  |  pred(raw)  |  pred(0/1)\n";
    cout << "  --------|---------|-------------|-----------\n";
    for (int i = 0; i < data.get_row_number(); ++i) {
        cout << "  " << data.get_data(i, 0) << "   " << data.get_data(i, 1)
             << "  |   " << labels.get_data(i, 0)
             << "   |   " << setw(8) << predictions.get_data(i, 0)
             << "  |    " << hard.get_data(i, 0) << "\n";
    }

    if (final_acc >= 0.99) {
        cout << "\nXOR learned successfully.\n";
        return 0;
    }
    cout << "\nTraining finished, but accuracy is not perfect yet."
         << " Try more epochs or tune learning rate.\n";
    return 1;
}

}  // namespace demos
