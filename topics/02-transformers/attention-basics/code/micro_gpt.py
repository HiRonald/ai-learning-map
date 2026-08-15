'''
numpy Tensor 版最小 GPT（无 PyTorch）。

图的节点从标量 Value 变成 Tensor（一块数组上的算子）。
求导仍是链式法则，只是局部导数变成对 ndarray 的操作。

阅读顺序:
  1. Tensor 自动微分
  2. 网络算子
  3. 数据与分词
  4. 超参与 GPT 前向（单 token + KV cache）
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

    def detach(self) -> Tensor:
        return Tensor(self.data)

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


def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors))

    def backward() -> None:
        gs = np.moveaxis(out.grad, axis, 0)
        for t, g in zip(tensors, gs):
            t.grad += g

    out._backward = backward
    return out


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
    # 与 PyTorch F.linear(x, w) 相同：x @ w.T，w 形状 (out, in)
    return x @ w.T


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
# 4. 超参与 GPT 前向（单 token + KV cache）
# ---------------------------------------------------------------------------

EMB_DIM = 16
LAYER_NUM = 1
HEAD_NUM = 4
HEAD_DIM = EMB_DIM // HEAD_NUM

KVCache = list[list[Tensor]]  # layer -> 历史 K 或 V，每个 shape (emb,)


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


def empty_kv_cache() -> KVCache:
    return [[] for _ in range(LAYER_NUM)]


def gpt(
    token_id: int,
    pos_id: int,
    keys: KVCache,
    values: KVCache,
    state: dict[str, Tensor],
    *,
    detach_cache: bool = False,
) -> Tensor:
    """Pre-norm block；逐步吃 token，K/V 缓存在 keys/values 里。"""
    x = state['wte'][token_id] + state['wpe'][pos_id]  # (emb,)

    for li in range(LAYER_NUM):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state[f'layer{li}.attn_wq'])
        k = linear(x, state[f'layer{li}.attn_wk'])
        v = linear(x, state[f'layer{li}.attn_wv'])
        keys[li].append(k.detach() if detach_cache else k)
        values[li].append(v.detach() if detach_cache else v)

        # (emb,) -> (H, D)；序列 K/V -> (H, T, D)，一次算完所有头
        q_h = q.reshape(HEAD_NUM, HEAD_DIM)
        k_h = stack(keys[li]).reshape(-1, HEAD_NUM, HEAD_DIM).transpose(0, 1)
        v_h = stack(values[li]).reshape(-1, HEAD_NUM, HEAD_DIM).transpose(0, 1)
        attn_logits = (q_h.reshape(HEAD_NUM, 1, HEAD_DIM) * k_h).sum(axis=-1) / HEAD_DIM ** 0.5
        attn_weights = softmax(attn_logits, axis=-1)  # (H, T)
        head_out = (attn_weights.reshape(HEAD_NUM, -1, 1) * v_h).sum(axis=1)  # (H, D)
        x = linear(head_out.reshape(EMB_DIM), state[f'layer{li}.attn_wo'])
        x = x + x_residual

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state[f'layer{li}.mlp_fc1']).relu()
        x = linear(x, state[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    return linear(rmsnorm(x), state['lm_head'])  # (vocab,)


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
) -> None:
    params = list(state.values())
    print(f'num of names: {len(names)}')
    print(f'num params: {sum(p.data.size for p in params)}')
    print('Training...')
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    for step in range(num_steps):
        name = names[step % len(names)]
        tokens = [eos] + [stoi[ch] for ch in name] + [eos]
        n = min(block_size, len(tokens) - 1)
        keys, values = empty_kv_cache(), empty_kv_cache()

        losses: list[Tensor] = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, state)
            losses.append(-softmax(logits)[target_id].log())
        loss = (1 / n) * sum(losses)
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
        keys, values = empty_kv_cache(), empty_kv_cache()
        token_id = eos
        output: list[str] = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values, state, detach_cache=True)
            probs = softmax(logits / temperature)
            next_token_id = random.choices(range(vocab_size), weights=probs.data.tolist())[0]
            if next_token_id == eos:
                break
            output.append(charlist[next_token_id])
            token_id = next_token_id
        print(f'sample {sample_idx + 1:2d} output: {"".join(output)}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Train a tiny GPT on makemore names, then sample.')
    parser.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '1000')))
    parser.add_argument('--lr', type=float, default=float(os.environ.get('LR', '0.001')))
    parser.add_argument('--samples', type=int, default=int(os.environ.get('SAMPLES', '20')))
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args(argv)

    names, stoi, charlist, eos, block_size = load_names()
    vocab_size = eos + 1
    state = init_params(vocab_size, block_size)
    train(names, stoi, eos, block_size, state, args.steps, args.lr)
    inference(charlist, eos, vocab_size, block_size, state, args.samples, args.temperature)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
