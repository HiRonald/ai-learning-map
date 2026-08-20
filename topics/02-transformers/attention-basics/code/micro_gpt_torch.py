'''
PyTorch 对照版最小 GPT（同一套超参 / 数据 / 训练闭环）。

对照 micro_gpt.py：手写 Tensor 自动微分换成 torch.nn + autograd。
多头仍自己写（不用 nn.MultiheadAttention），方便和 numpy 版逐行对形状。

维度约定（全文注释沿用这些字母）:
  B  batch size，一次喂进模型的样本数
  T  sequence length，本 batch 的 token 数（可变，右 padding）
  C  embedding dim = EMB_DIM，token 通道数
  H  head 数 = HEAD_NUM
  D  每头维度 = HEAD_DIM = C / H
  V  vocab size，词表大小（含 <EOS>）

阅读顺序:
  1. 数据与分词
  2. RMSNorm / 多头 / GPT
  3. AdamW / 训练 / 采样
'''

from __future__ import annotations

import argparse
import math
import os
import random
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'

EMB_DIM = 16
LAYER_NUM = 1
HEAD_NUM = 4
HEAD_DIM = EMB_DIM // HEAD_NUM
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# 1. 数据与分词（与 micro_gpt.py 相同）
# ---------------------------------------------------------------------------

def data_dir() -> str:
    env = os.environ.get('ATTN_DATA_DIR')
    if env:
        return env
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def load_names() -> tuple[list[str], dict[str, int], list[str], int, int]:
    path = os.path.join(data_dir(), 'names.txt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'downloading names.txt → {path}')
        urllib.request.urlretrieve(NAMES_URL, path)

    names = [line.strip() for line in open(path) if line.strip()]
    random.shuffle(names)
    charlist = sorted(set(''.join(names)))
    stoi = {ch: i for i, ch in enumerate(charlist)}
    eos = len(charlist)
    stoi['<EOS>'] = eos
    block_size = max(len(name) for name in names) + 1
    return names, stoi, charlist, eos, block_size


def make_batch(
    batch_names: list[str],
    stoi: dict[str, int],
    eos: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """右 padding 到本 batch 最长。返回 x, y, mask，形状都是 (B, T)。"""
    seqs = [[eos] + [stoi[ch] for ch in name] + [eos] for name in batch_names]
    lengths = [min(len(seq) - 1, block_size) for seq in seqs]
    seq_len = max(lengths)  # T
    batch = len(seqs)       # B
    x = torch.full((batch, seq_len), eos, dtype=torch.long)       # (B, T) 输入 token
    y = torch.full((batch, seq_len), eos, dtype=torch.long)       # (B, T) 下一 token（监督）
    mask = torch.zeros((batch, seq_len), dtype=torch.float32)     # (B, T) 1=有效，0=padding
    for i, seq in enumerate(seqs):
        n = lengths[i]
        x[i, :n] = torch.tensor(seq[:n], dtype=torch.long)
        y[i, :n] = torch.tensor(seq[1:n + 1], dtype=torch.long)
        mask[i, :n] = 1.0
    return x, y, mask


# ---------------------------------------------------------------------------
# 2. RMSNorm / 多头 / GPT
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """(..., C) → (..., C)。只在最后一维 C 上归一化。"""
    ms = x.pow(2).mean(dim=-1, keepdim=True)  # (..., 1)
    return x / (ms + 1e-5).sqrt()


class CausalSelfAttention(nn.Module):
    """(B, T, C) → (B, T, C)。一次投影再拆成 H 头，各算一份 (T, T)。"""

    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(EMB_DIM, EMB_DIM, bias=False)  # weight (C, C)
        self.wk = nn.Linear(EMB_DIM, EMB_DIM, bias=False)
        self.wv = nn.Linear(EMB_DIM, EMB_DIM, bias=False)
        self.wo = nn.Linear(EMB_DIM, EMB_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape  # x: (B, T, C)
        # Linear(C→C) 后再拆头: (B, T, C) → (B, T, H, D) → (B, H, T, D)
        q = self.wq(x).view(batch, seq, HEAD_NUM, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(batch, seq, HEAD_NUM, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(batch, seq, HEAD_NUM, HEAD_DIM).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)  # (B, H, T, T)
        causal = torch.triu(torch.full((seq, seq), float('-inf'), device=x.device), diagonal=1)  # (T, T)
        weights = torch.softmax(scores + causal, dim=-1)  # (B, H, T, T)，末维对 key 做 softmax
        out = weights @ v  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(batch, seq, EMB_DIM)  # (B, T, C)
        return self.wo(out)  # (B, T, C)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(EMB_DIM, 4 * EMB_DIM, bias=False)  # weight (4C, C)
        self.fc2 = nn.Linear(4 * EMB_DIM, EMB_DIM, bias=False)  # weight (C, 4C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, T, 4C) → (B, T, C)
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = CausalSelfAttention()
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual；全程 (B, T, C)
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, EMB_DIM)          # (V, C)
        self.wpe = nn.Embedding(block_size, EMB_DIM)          # (max T, C)
        self.layers = nn.ModuleList([Block() for _ in range(LAYER_NUM)])
        self.lm_head = nn.Linear(EMB_DIM, vocab_size, bias=False)  # weight (V, C)
        for param in self.parameters():
            nn.init.normal_(param, 0.0, 0.08)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids (B, T) → logits (B, T, V)。"""
        seq = token_ids.shape[1]
        pos = torch.arange(seq, device=token_ids.device)  # (T,)
        x = self.wte(token_ids) + self.wpe(pos)  # (B, T, C) + (T, C) 广播
        for layer in self.layers:
            x = layer(x)  # (B, T, C)
        return self.lm_head(rmsnorm(x))  # (B, T, V)


def masked_nll(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), targets/mask (B, T) → 标量；padding 位用 mask 丢掉。"""
    batch, seq, vocab = logits.shape
    nll = F.cross_entropy(logits.reshape(batch * seq, vocab), targets.reshape(-1), reduction='none')  # (B*T,)
    return (nll * mask.reshape(-1)).sum() / mask.sum()


# ---------------------------------------------------------------------------
# 3. AdamW / 训练 / 采样
# ---------------------------------------------------------------------------

def train(
    names: list[str],
    stoi: dict[str, int],
    eos: int,
    block_size: int,
    model: GPT,
    num_steps: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    print(f'num of names: {len(names)}')
    print(f'num params: {sum(p.numel() for p in model.parameters())}')
    print(f'batch size: {batch_size}')
    print('Training...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    model.train()
    for step in range(num_steps):
        start = (step * batch_size) % len(names)
        batch_names = [names[(start + i) % len(names)] for i in range(batch_size)]
        token_ids, targets, mask = make_batch(batch_names, stoi, eos, block_size)  # (B, T)
        logits = model(token_ids)  # (B, T, V)
        loss = masked_nll(logits, targets, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}', end='\r')
    print('\nTraining finished!')


@torch.no_grad()
def inference(
    charlist: list[str],
    eos: int,
    vocab_size: int,
    block_size: int,
    model: GPT,
    num_samples: int,
    temperature: float,
) -> None:
    print('Inference...')
    model.eval()
    for sample_idx in range(num_samples):
        token_ids = [eos]
        output: list[str] = []
        for _ in range(block_size):
            logits = model(torch.tensor([token_ids], dtype=torch.long))  # (1, t, V)，t 逐步变长
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)  # (V,) 只取最后一个位置
            next_token_id = torch.multinomial(probs, 1).item()
            if next_token_id == eos:
                break
            output.append(charlist[next_token_id])
            token_ids.append(next_token_id)
        print(f'sample {sample_idx + 1:2d} output: {"".join(output)}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='PyTorch tiny GPT on makemore names.')
    parser.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '1000')))
    parser.add_argument('--batch', type=int, default=int(os.environ.get('BATCH', str(BATCH_SIZE))))
    parser.add_argument('--lr', type=float, default=float(os.environ.get('LR', '0.001')))
    parser.add_argument('--samples', type=int, default=int(os.environ.get('SAMPLES', '20')))
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args(argv)

    names, stoi, charlist, eos, block_size = load_names()
    vocab_size = eos + 1
    model = GPT(vocab_size, block_size)
    train(names, stoi, eos, block_size, model, args.steps, args.lr, args.batch)
    inference(charlist, eos, vocab_size, block_size, model, args.samples, args.temperature)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
