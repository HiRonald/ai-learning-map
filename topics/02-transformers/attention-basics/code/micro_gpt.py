'''
numpy Tensor 版最小 GPT（无 PyTorch）。

图的节点从标量 Value 变成 Tensor（一块数组上的算子）。
求导仍是链式法则，只是局部导数变成对 ndarray 的操作。

阅读顺序:
  1. Tensor 自动微分
  2. 网络算子
  3. 数据与分词
  4. 超参与 GPT 前向（batch × 序列，因果 mask）
  5. AdamW / 训练 / 采样
'''

from __future__ import annotations

import argparse
import math
import os
import random
import urllib.request
from typing import Callable, Union

import numpy as np

random.seed(42)
np.random.seed(42)

Number = Union[float, int]

NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'


# ---------------------------------------------------------------------------
# 1. Tensor 自动微分
# ---------------------------------------------------------------------------

def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """广播的反向：把梯度求和收回到原始 shape。"""
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    reduce_axes = tuple(
        i for i, (g, s) in enumerate(zip(grad.shape, shape)) if s == 1 and g != 1
    )
    if reduce_axes:
        grad = grad.sum(axis=reduce_axes, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """data / grad 都是 ndarray；每个算子是计算图上的一个节点。"""

    __slots__ = ('data', 'grad', '_children', '_backward')

    data: np.ndarray
    grad: np.ndarray
    _children: tuple[Tensor, ...]
    _backward: Callable[[], None]

    def __init__(
        self,
        data: np.ndarray | Number | list,
        children: tuple[Tensor, ...] = (),
        backward: Callable[[], None] | None = None,
    ) -> None:
        self.data = np.array(data, dtype=np.float64, copy=True, order='C')
        self.grad = np.zeros_like(self.data)
        self._children = children
        self._backward = backward or (lambda: None)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f'Tensor(shape={self.shape})'

    # --- 算术：+ * @ ** ---

    def __add__(self, other: Tensor | Number) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, (self, other))

            def backward() -> None:
                self.grad += _unbroadcast(out.grad, self.shape)
                other.grad += _unbroadcast(out.grad, other.shape)

            out._backward = backward
            return out

        out = Tensor(self.data + other, (self,))

        def backward() -> None:
            self.grad += _unbroadcast(out.grad, self.shape)

        out._backward = backward
        return out

    def __mul__(self, other: Tensor | Number) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, (self, other))

            def backward() -> None:
                self.grad += _unbroadcast(other.data * out.grad, self.shape)
                other.grad += _unbroadcast(self.data * out.grad, other.shape)

            out._backward = backward
            return out

        out = Tensor(self.data * other, (self,))

        def backward() -> None:
            self.grad += _unbroadcast(other * out.grad, self.shape)

        out._backward = backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        # 1D/2D 用 np.dot：macOS Accelerate 的 `@` / np.matmul 会误报 overflow
        out = Tensor(np.dot(self.data, other.data), (self, other))

        def backward() -> None:
            a, b, g = self.data, other.data, out.grad
            a1d, b1d = a.ndim == 1, b.ndim == 1
            ap = a[None, :] if a1d else a
            bp = b[:, None] if b1d else b
            gp = np.reshape(g, (1, 1)) if (a1d and b1d) else (
                g[None, :] if a1d else (g[:, None] if b1d else g)
            )
            da = np.dot(gp, np.swapaxes(bp, -1, -2))
            db = np.dot(np.swapaxes(ap, -1, -2), gp)
            self.grad += da[0] if a1d else da
            other.grad += db[..., 0] if b1d else db

        out._backward = backward
        return out

    def __pow__(self, n: Number) -> Tensor:
        out = Tensor(self.data ** n, (self,))

        def backward() -> None:
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | Number) -> Tensor:
        return self + (-other if isinstance(other, Tensor) else -other)

    def __truediv__(self, other: Tensor | Number) -> Tensor:
        return self * (other ** -1)

    def __radd__(self, other: Number) -> Tensor:
        return self + other

    def __rmul__(self, other: Number) -> Tensor:
        return self * other

    def __rsub__(self, other: Number) -> Tensor:
        return other + (-self)

    def __rtruediv__(self, other: Number) -> Tensor:
        return other * (self ** -1)

    # --- 逐元素激活 ---

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,))

        def backward() -> None:
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), (self,))

        def backward() -> None:
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def backward() -> None:
            self.grad += out.grad * (self.data > 0)

        out._backward = backward
        return out

    # --- 归约 / 形状 / 索引 ---

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def backward() -> None:
            g = out.grad
            if axis is None:
                self.grad += g * np.ones_like(self.data)
                return
            axes = axis if isinstance(axis, tuple) else (axis,)
            if not keepdims:
                for ax in sorted(a if a >= 0 else a + self.data.ndim for a in axes):
                    g = np.expand_dims(g, ax)
            self.grad += g

        out._backward = backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = math.prod(self.data.shape[ax] for ax in axes)
        return self.sum(axis=axis, keepdims=keepdims) / n

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(*shape), (self,))

        def backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = backward
        return out

    def transpose(self, axis0: int = 0, axis1: int = 1) -> Tensor:
        out = Tensor(self.data.swapaxes(axis0, axis1), (self,))

        def backward() -> None:
            self.grad += out.grad.swapaxes(axis0, axis1)

        out._backward = backward
        return out

    @property
    def T(self) -> Tensor:
        if self.data.ndim < 2:
            return self
        return self.transpose(-2, -1)

    def __getitem__(self, idx: int | slice | tuple | np.ndarray) -> Tensor:
        out = Tensor(self.data[idx], (self,))

        def backward() -> None:
            if isinstance(idx, (int, np.integer, np.ndarray)):
                np.add.at(self.grad, idx, out.grad)
            else:
                self.grad[idx] += out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        visited: dict[Tensor, bool] = {}
        topo: list[Tensor] = []

        def dfs(node: Tensor) -> None:
            if visited.get(node, False):
                return
            visited[node] = True
            for child in node._children:
                dfs(child)
            topo.append(node)

        dfs(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# ---------------------------------------------------------------------------
# 2. 网络算子
# ---------------------------------------------------------------------------

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    # 减 max 只为数值稳定；softmax(x - c) = softmax(x)，故 max 不进计算图
    x = x - x.data.max(axis=axis, keepdims=True)
    exps = x.exp()
    return exps / exps.sum(axis=axis, keepdims=True)


def rmsnorm(x: Tensor) -> Tensor:
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x / (ms + 1e-5) ** 0.5


def linear(x: Tensor, w: Tensor) -> Tensor:
    # F.linear: x @ w.T，w 形状 (out, in)。先摊成 2D，避开 3D np.dot 语义
    out_f, in_f = w.shape
    return (x.reshape(-1, in_f) @ w.T).reshape(*x.shape[:-1], out_f)


# ---------------------------------------------------------------------------
# 3. 数据与分词
# ---------------------------------------------------------------------------

def data_dir() -> str:
    env = os.environ.get('ATTN_DATA_DIR')
    if env:
        return env
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def load_names() -> tuple[list[str], dict[str, int], list[str], int, int]:
    """返回 names, stoi, charlist, eos_id, block_size。"""
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
    block_size = max(len(name) for name in names) + 1  # +1 for EOS
    return names, stoi, charlist, eos, block_size


# ---------------------------------------------------------------------------
# 4. 超参与 GPT 前向（batch × 序列，因果 mask）
# ---------------------------------------------------------------------------

EMB_DIM = 16
LAYER_NUM = 1
HEAD_NUM = 4
HEAD_DIM = EMB_DIM // HEAD_NUM
BATCH_SIZE = 32


def randn_matrix(out_features: int, in_features: int, std: float = 0.08) -> Tensor:
    return Tensor(np.random.normal(0.0, std, size=(out_features, in_features)))


def init_params(vocab_size: int, block_size: int) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {
        'wte': randn_matrix(vocab_size, EMB_DIM),
        'wpe': randn_matrix(block_size, EMB_DIM),
        'lm_head': randn_matrix(vocab_size, EMB_DIM),
    }
    for i in range(LAYER_NUM):
        state[f'layer{i}.attn_wq'] = randn_matrix(EMB_DIM, EMB_DIM)
        state[f'layer{i}.attn_wk'] = randn_matrix(EMB_DIM, EMB_DIM)
        state[f'layer{i}.attn_wv'] = randn_matrix(EMB_DIM, EMB_DIM)
        state[f'layer{i}.attn_wo'] = randn_matrix(EMB_DIM, EMB_DIM)
        state[f'layer{i}.mlp_fc1'] = randn_matrix(4 * EMB_DIM, EMB_DIM)
        state[f'layer{i}.mlp_fc2'] = randn_matrix(EMB_DIM, 4 * EMB_DIM)
    return state


def split_heads(x: Tensor) -> Tensor:
    """(B, T, C) → (B, H, T, D)。一次大投影再切开，等价于 H 套独立的 Wq/Wk/Wv。"""
    batch, seq, _ = x.shape
    return x.reshape(batch, seq, HEAD_NUM, HEAD_DIM).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, D) → (B, T, C)，交给 W_o 混回头之间的信息。"""
    batch, _, seq, _ = x.shape
    return x.transpose(1, 2).reshape(batch, seq, EMB_DIM)


def multi_head_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """因果多头。q,k,v: (B, T, C) → (B, T, C)。

    每个头一份 (T, T) 权重（最后一维独立 softmax），互不共享。
    右 padding + 因果 mask 时，真 token 看不到 pad。
    """
    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)  # (B, H, T, D)
    batch, _, seq, _ = qh.shape
    scores = (
        qh.reshape(batch, HEAD_NUM, seq, 1, HEAD_DIM)
        * kh.reshape(batch, HEAD_NUM, 1, seq, HEAD_DIM)
    ).sum(axis=-1) / HEAD_DIM ** 0.5  # (B, H, T, T)
    scores = scores + Tensor(np.triu(np.full((seq, seq), -1e9), k=1))
    weights = softmax(scores, axis=-1)
    out = (
        weights.reshape(batch, HEAD_NUM, seq, seq, 1)
        * vh.reshape(batch, HEAD_NUM, 1, seq, HEAD_DIM)
    ).sum(axis=3)  # (B, H, T, D)
    return merge_heads(out)


def gpt(token_ids: np.ndarray, state: dict[str, Tensor]) -> Tensor:
    """token_ids (B, T) int → logits (B, T, vocab)。时间维一次算完，用因果 mask 防偷看。"""
    seq = token_ids.shape[1]
    x = state['wte'][token_ids] + state['wpe'][np.arange(seq)]
    for li in range(LAYER_NUM):
        h = rmsnorm(x)
        q = linear(h, state[f'layer{li}.attn_wq'])
        k = linear(h, state[f'layer{li}.attn_wk'])
        v = linear(h, state[f'layer{li}.attn_wv'])
        x = x + linear(multi_head_attention(q, k, v), state[f'layer{li}.attn_wo'])
        h = rmsnorm(x)
        x = x + linear(linear(h, state[f'layer{li}.mlp_fc1']).relu(), state[f'layer{li}.mlp_fc2'])
    return linear(rmsnorm(x), state['lm_head'])


def make_batch(
    batch_names: list[str],
    stoi: dict[str, int],
    eos: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """右 padding 到本 batch 最长。x/y/mask 都是 (B, T)；pad 位用 eos，loss 用 mask 丢掉。"""
    seqs = [[eos] + [stoi[ch] for ch in name] + [eos] for name in batch_names]
    lengths = [min(len(seq) - 1, block_size) for seq in seqs]
    seq_len = max(lengths)
    batch = len(seqs)
    x = np.full((batch, seq_len), eos, dtype=np.int64)
    y = np.full((batch, seq_len), eos, dtype=np.int64)
    mask = np.zeros((batch, seq_len), dtype=np.float64)
    for i, seq in enumerate(seqs):
        n = lengths[i]
        x[i, :n] = seq[:n]
        y[i, :n] = seq[1:n + 1]
        mask[i, :n] = 1.0
    return x, y, mask


def masked_nll(logits: Tensor, targets: np.ndarray, mask: np.ndarray) -> Tensor:
    batch, seq, vocab = logits.shape
    probs = softmax(logits.reshape(batch * seq, vocab), axis=-1)
    nll = -probs[np.arange(batch * seq), targets.reshape(-1)].log()
    return (nll.reshape(batch, seq) * mask).sum() / float(mask.sum())


# ---------------------------------------------------------------------------
# 5. AdamW / 训练 / 推理
# ---------------------------------------------------------------------------

def adamw_optimize(
    params: list[Tensor],
    m: list[np.ndarray],
    v: list[np.ndarray],
    learning_rate: float,
    step: int,
) -> None:
    beta1, beta2, eps, weight_decay = 0.9, 0.999, 1e-8, 0.01
    for i, param in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * param.grad
        v[i] = beta2 * v[i] + (1 - beta2) * param.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        param.data -= learning_rate * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param.data)
        param.grad.fill(0.0)


def train(
    names: list[str],
    stoi: dict[str, int],
    eos: int,
    block_size: int,
    state: dict[str, Tensor],
    num_steps: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    params = list(state.values())
    print(f'num of names: {len(names)}')
    print(f'num params: {sum(p.data.size for p in params)}')
    print(f'batch size: {batch_size}')
    print('Training...')
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    for step in range(num_steps):
        start = (step * batch_size) % len(names)
        batch_names = [names[(start + i) % len(names)] for i in range(batch_size)]
        token_ids, targets, mask = make_batch(batch_names, stoi, eos, block_size)
        logits = gpt(token_ids, state)
        loss = masked_nll(logits, targets, mask)
        loss.backward()
        adamw_optimize(params, m, v, learning_rate, step)
        print(f'step {step + 1:4d} / {num_steps:4d} | loss {float(loss.data):.4f}', end='\r')
    print('\nTraining finished!')


def inference(
    charlist: list[str],
    eos: int,
    vocab_size: int,
    block_size: int,
    state: dict[str, Tensor],
    num_samples: int,
    temperature: float,
) -> None:
    print('Inference...')
    for sample_idx in range(num_samples):
        token_ids = [eos]
        output: list[str] = []
        for _ in range(block_size):
            logits = gpt(np.array([token_ids], dtype=np.int64), state)
            probs = softmax(logits[0, -1] / temperature)
            next_token_id = random.choices(range(vocab_size), weights=probs.data.tolist())[0]
            if next_token_id == eos:
                break
            output.append(charlist[next_token_id])
            token_ids.append(next_token_id)
        print(f'sample {sample_idx + 1:2d} output: {"".join(output)}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Train a tiny GPT on makemore names, then sample.')
    parser.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '1000')))
    parser.add_argument('--batch', type=int, default=int(os.environ.get('BATCH', str(BATCH_SIZE))))
    parser.add_argument('--lr', type=float, default=float(os.environ.get('LR', '0.001')))
    parser.add_argument('--samples', type=int, default=int(os.environ.get('SAMPLES', '20')))
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args(argv)

    names, stoi, charlist, eos, block_size = load_names()
    vocab_size = eos + 1
    state = init_params(vocab_size, block_size)
    train(names, stoi, eos, block_size, state, args.steps, args.lr, args.batch)
    inference(charlist, eos, vocab_size, block_size, state, args.samples, args.temperature)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
