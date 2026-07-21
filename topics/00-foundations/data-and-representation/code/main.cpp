#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "demos/common.h"
#include "nn/matrix.h"

using namespace std;

namespace {

constexpr int kEmbedDim = 8;
constexpr unsigned kSeed = 7;
constexpr int kTrainEpochs = 400;
constexpr double kTrainLr = 0.35;

// 略扩一点玩具语料，让 cat↔mat / dog↔log 的转移更明显
const vector<string> kCorpus = {
    "the cat sat on the mat",
    "the dog sat on the log",
    "a cat and a dog",
    "the cat likes the mat",
    "the dog likes the log",
};

const char* kUnk = "<unk>";
const char* kPad = "<pad>";

string to_lower(string s) {
    for (char& c : s) {
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

// 极简 tokenizer：小写 + 按空白切分（故意不做 BPE）
vector<string> tokenize(const string& text) {
    istringstream iss(to_lower(text));
    vector<string> tokens;
    string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

struct Vocab {
    vector<string> id_to_token;
    map<string, int> token_to_id;

    int size() const { return static_cast<int>(id_to_token.size()); }

    int unk_id() const { return token_to_id.at(kUnk); }
    int pad_id() const { return token_to_id.at(kPad); }

    int id_of(const string& token) const {
        auto it = token_to_id.find(token);
        if (it == token_to_id.end()) {
            return unk_id();
        }
        return it->second;
    }
};

Vocab build_vocab(const vector<string>& corpus) {
    map<string, int> counts;
    for (const string& line : corpus) {
        for (const string& tok : tokenize(line)) {
            ++counts[tok];
        }
    }

    Vocab v;
    auto add = [&](const string& tok) {
        if (v.token_to_id.count(tok)) {
            return;
        }
        int id = static_cast<int>(v.id_to_token.size());
        v.token_to_id[tok] = id;
        v.id_to_token.push_back(tok);
    };
    add(kPad);
    add(kUnk);
    for (const auto& [tok, _] : counts) {
        add(tok);
    }
    return v;
}

vector<int> encode(const Vocab& vocab, const string& text) {
    vector<int> ids;
    for (const string& tok : tokenize(text)) {
        ids.push_back(vocab.id_of(tok));
    }
    return ids;
}

Matrix lookup(const Matrix& table, const vector<int>& ids) {
    const int seq = static_cast<int>(ids.size());
    const int dim = table.get_col_number();
    Matrix out(seq, dim);
    for (int i = 0; i < seq; ++i) {
        int id = ids[static_cast<size_t>(i)];
        for (int d = 0; d < dim; ++d) {
            out.set_data(i, d, table.get_data(id, d));
        }
    }
    return out;
}

double cosine_rows(const Matrix& table, int id_a, int id_b) {
    const int dim = table.get_col_number();
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int d = 0; d < dim; ++d) {
        double a = table.get_data(id_a, d);
        double b = table.get_data(id_b, d);
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0;
    }
    return dot / (sqrt(na) * sqrt(nb));
}

void print_ids(const Vocab& vocab, const vector<string>& tokens, const vector<int>& ids) {
    cout << "  tokens: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) {
            cout << " | ";
        }
        cout << tokens[i];
    }
    cout << "\n  ids:    ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) {
            cout << " | ";
        }
        cout << setw(4) << ids[i];
    }
    cout << "\n  check:  ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) {
            cout << " | ";
        }
        cout << vocab.id_to_token[static_cast<size_t>(ids[i])];
    }
    cout << "\n";
}

Matrix one_hot_rows(const vector<int>& ids, int vocab_size) {
    Matrix x(static_cast<int>(ids.size()), vocab_size, 0.0);
    for (size_t i = 0; i < ids.size(); ++i) {
        x.set_data(static_cast<int>(i), ids[i], 1.0);
    }
    return x;
}

Matrix one_hot_row(int id, int vocab_size) {
    return one_hot_rows(vector<int>{id}, vocab_size);
}

void fill_demo_table(Matrix& table) {
    for (int i = 0; i < table.get_row_number(); ++i) {
        for (int j = 0; j < table.get_col_number(); ++j) {
            unsigned h = static_cast<unsigned>(i * 131 + j * 17 + kSeed);
            h ^= h << 13;
            h ^= h >> 17;
            h ^= h << 5;
            double v = (static_cast<int>(h % 1000) / 1000.0) - 0.5;
            table.set_data(i, j, v);
        }
    }
}

void run_lookup_part(const Vocab& vocab) {
    demos::print_separator("Part A — lookup only (no training)");
    cout << "Pipeline: text -> tokens -> ids -> embedding rows (seq, dim).\n";

    demos::print_separator("A1) Toy corpus");
    for (size_t i = 0; i < kCorpus.size(); ++i) {
        cout << "  [" << i << "] " << kCorpus[i] << "\n";
    }

    demos::print_separator("A2) Vocab");
    cout << "vocab_size=" << vocab.size() << "  (includes " << kPad << ", " << kUnk << ")\n";
    for (int id = 0; id < vocab.size(); ++id) {
        cout << "  id=" << setw(2) << id << "  token='" << vocab.id_to_token[static_cast<size_t>(id)]
             << "'\n";
    }

    demos::print_separator("A3) Encode");
    const string& sample = kCorpus[0];
    vector<string> tokens = tokenize(sample);
    vector<int> ids = encode(vocab, sample);
    cout << "text: \"" << sample << "\"\n";
    print_ids(vocab, tokens, ids);

    demos::print_separator("A4) OOV -> <unk>");
    const string oov_text = "the tiger sat on the mat";
    cout << "text: \"" << oov_text << "\"\n";
    print_ids(vocab, tokenize(oov_text), encode(vocab, oov_text));

    demos::print_separator("A5) Random embedding table + lookup");
    Matrix table(vocab.size(), kEmbedDim);
    fill_demo_table(table);
    cout << "E shape: (" << table.get_row_number() << ", " << table.get_col_number()
         << ") = (vocab, dim)\n";
    Matrix emb = lookup(table, ids);
    cout << "lookup shape: (" << emb.get_row_number() << ", " << emb.get_col_number()
         << ") = (seq_len, dim)\n";
    emb.print("token vectors");

    int cat_id = vocab.id_of("cat");
    int dog_id = vocab.id_of("dog");
    cout << "\ncosine(cat, dog) @ random E = " << cosine_rows(table, cat_id, dog_id)
         << "  (not meaningful yet)\n";
}

// 上一词 one-hot → MLP → 下一词：第一层 W (vocab, dim) 就是可训练的 embedding
void run_bigram_part(const Vocab& vocab) {
    demos::print_separator("Part B — bigram MLP (train a tiny next-word model)");
    cout << "Trick: one-hot(prev) @ W == embedding lookup.\n"
         << "So the first layer weights ARE the embedding table — and they get trained.\n";

    vector<int> prev_ids;
    vector<int> next_ids;
    for (const string& line : kCorpus) {
        vector<int> ids = encode(vocab, line);
        for (size_t t = 1; t < ids.size(); ++t) {
            prev_ids.push_back(ids[t - 1]);
            next_ids.push_back(ids[t]);
        }
    }
    const int n = static_cast<int>(prev_ids.size());
    const int V = vocab.size();
    Matrix x = one_hot_rows(prev_ids, V);
    Matrix y = one_hot_rows(next_ids, V);
    cout << "training pairs (bigrams): " << n << "\n"
         << "net: " << V << " -> " << kEmbedDim << " -> " << V
         << "  loss=softmax_ce\n";

    auto mlp = demos::build_mlp(V, {kEmbedDim}, V, "relu", "identity", "softmax_ce", kTrainLr);
    mlp->print_architecture(/*verbose=*/false);

    cout << fixed << setprecision(6);
    for (int epoch = 0; epoch < kTrainEpochs; ++epoch) {
        mlp->train(x, y);
        if (epoch % 100 == 0 || epoch + 1 == kTrainEpochs) {
            Matrix pred = mlp->predict(x);
            float loss = mlp->eval(y);
            double acc = demos::argmax_accuracy(pred, y);
            cout << "Epoch " << setw(3) << epoch
                 << " | loss=" << loss
                 << " | train_acc=" << setprecision(1) << acc * 100.0 << "%"
                 << setprecision(6) << "\n";
        }
    }

    // 第一层 W：行 = token embedding
    Matrix E = mlp->capture_params().front().first;
    cout << "\nLearned embedding E = first-layer W, shape ("
         << E.get_row_number() << ", " << E.get_col_number() << ")\n";

    auto show_cos = [&](const char* a, const char* b) {
        cout << "  cosine(" << a << ", " << b << ") = "
             << setprecision(4) << cosine_rows(E, vocab.id_of(a), vocab.id_of(b))
             << setprecision(6) << "\n";
    };
    cout << "Pairwise cosines (toy signal, not Word2Vec quality):\n";
    show_cos("cat", "dog");
    show_cos("mat", "log");
    show_cos("cat", "mat");
    show_cos("dog", "log");
    show_cos("sat", "likes");

    demos::print_separator("B1) Next-word probes (argmax)");
    auto predict_next = [&](const string& prev) {
        int id = vocab.id_of(prev);
        Matrix logits = mlp->predict(one_hot_row(id, V));
        // 解码时屏蔽特殊符
        logits.set_data(0, vocab.pad_id(), -1e9);
        logits.set_data(0, vocab.unk_id(), -1e9);
        int nid = demos::argmax_row(logits, 0);
        cout << "  '" << prev << "' -> '" << vocab.id_to_token[static_cast<size_t>(nid)] << "'\n";
        return nid;
    };
    predict_next("sat");
    predict_next("on");
    predict_next("likes");
    predict_next("cat");
    predict_next("dog");

    demos::print_separator("B2) Greedy generate from seed");
    auto generate = [&](const string& seed, int steps) {
        int cur = vocab.id_of(seed);
        cout << "  " << seed;
        for (int i = 0; i < steps; ++i) {
            Matrix logits = mlp->predict(one_hot_row(cur, V));
            logits.set_data(0, vocab.pad_id(), -1e9);
            logits.set_data(0, vocab.unk_id(), -1e9);
            cur = demos::argmax_row(logits, 0);
            cout << " " << vocab.id_to_token[static_cast<size_t>(cur)];
        }
        cout << "\n";
    };
    generate("the", 5);
    generate("a", 4);
    generate("cat", 4);
    cout << "  (bigram LM is myopic — loops/ambiguity are expected on tiny data)\n";
}

}  // namespace

int main() {
    cout << fixed << setprecision(4);
    demos::print_separator("data-and-representation");
    cout << "Part A: representation pipeline (lookup).\n"
         << "Part B: train a bigram MLP — first layer = embedding.\n";

    Vocab vocab = build_vocab(kCorpus);
    run_lookup_part(vocab);
    run_bigram_part(vocab);

    demos::print_separator("Contrast");
    cout << "Fashion: pixels -> (784,) vector.\n"
         << "Text here: ids -> one-hot @ W -> hidden -> next-token logits.\n"
         << "nn.Embedding in PyTorch is the same as this first-layer trick.\n";

    demos::print_separator("Done");
    cout << "Next: pytorch-basics, or CNN/RNN for richer structure over these tensors.\n";
    return 0;
}
