'''
字符级 Tiny Shakespeare + 可选 DDP。

对照 Karpathy nanoGPT：
  数据 = tinyshakespeare 切成一条 token 流再随机抽 (B, T) 块（无人名那种 padding）
  配置 = config/train_shakespeare_char.py（--preset shakespeare）
  分布式 = 多进程各一份模型，DDP 在 backward 里 all-reduce 梯度

单卡:    python3 train.py
多进程:  python3 train.py --nproc 2
         # 或: torchrun --standalone --nproc_per_node=2 train.py
'''

from __future__ import annotations

import argparse
import contextlib
import math
import os
import socket
import sys
import time
import urllib.request

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

SHAKESPEARE_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
)

# Karpathy nanoGPT config/train_shakespeare_char.py
SHAKESPEARE = dict(
    n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
    batch_size=64, max_iters=5000, grad_accum=1,
    learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, beta2=0.99,
    eval_interval=250, eval_iters=200, log_interval=10,
)

# 笔记本几分钟能跑完、仍是同一套网络/数据
DEMO = dict(
    n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0,
    batch_size=32, max_iters=400, grad_accum=1,
    learning_rate=1e-3, min_lr=1e-4, warmup_iters=20, beta2=0.99,
    eval_interval=100, eval_iters=20, log_interval=10,
)


# ---------------------------------------------------------------------------
# 1. 数据：下载 → 字符表 → 90/10 长流
# ---------------------------------------------------------------------------

def data_dir() -> str:
    env = os.environ.get('DDP_DATA_DIR')
    if env:
        return env
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def load_shakespeare(master: bool = True, ddp: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[int, str], int]:
    path = os.path.join(data_dir(), 'input.txt')
    if not os.path.exists(path):
        if master:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f'downloading tinyshakespeare → {path}')
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        if ddp:
            dist.barrier()

    text = open(path, encoding='utf-8').read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(encoded))
    return encoded[:n], encoded[n:], itos, len(chars)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从长流里随机切 (B, T) 块。y 是 x 右移 1 位。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 2. 设备 / DDP
# ---------------------------------------------------------------------------

def is_ddp() -> bool:
    return int(os.environ.get('RANK', -1)) != -1


def pick_device(ddp: bool, local_rank: int, requested: str) -> torch.device:
    if requested == 'cpu':
        return torch.device('cpu')
    if ddp:
        # NCCL = NVIDIA；MPS 不跟 DDP 走，多进程演示用 gloo + CPU
        if requested in ('cuda', 'auto') and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device('cuda', local_rank)
        if requested == 'mps':
            print('DDP 不走 MPS，回退 CPU / gloo', file=sys.stderr)
        return torch.device('cpu')
    if requested == 'cuda':
        return torch.device('cuda')
    if requested == 'mps':
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def setup_ddp(device: torch.device) -> tuple[int, int]:
    backend = 'nccl' if device.type == 'cuda' else 'gloo'
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def unwrap(model: torch.nn.Module) -> GPT:
    return model.module if isinstance(model, DDP) else model


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return str(sock.getsockname()[1])


def _ddp_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    raise SystemExit(train(args))


def maybe_launch_ddp(args: argparse.Namespace) -> bool:
    """父进程 spawn 出 nproc 个 worker 后返回 True；已在进程组里则返回 False。"""
    if is_ddp() or args.nproc <= 1:
        return False
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', _free_port())
    mp.spawn(_ddp_worker, args=(args.nproc, args), nprocs=args.nproc, join=True)
    return True


# ---------------------------------------------------------------------------
# 3. 学习率 / 评测 / 采样
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, max_iters: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    if step >= max_iters:
        return min_lr
    decay = (step - warmup) / max(1, max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    splits: dict[str, torch.Tensor],
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss
        out[name] = losses.mean().item()
    model.train()
    return out


def sample_and_print(
    model: GPT,
    itos: dict[int, str],
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> None:
    stoi = {ch: i for i, ch in itos.items()}
    idx = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    text = ''.join(itos[int(t)] for t in out[0].tolist())
    print('\n----- sample -----')
    print(text)
    print('------------------\n')


# ---------------------------------------------------------------------------
# 4. 训练
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> int:
    ddp = is_ddp()
    rank = int(os.environ['RANK']) if ddp else 0
    local_rank = int(os.environ['LOCAL_RANK']) if ddp else 0
    world_size = int(os.environ['WORLD_SIZE']) if ddp else 1
    master = rank == 0

    device = pick_device(ddp, local_rank, args.device)
    if ddp:
        rank, world_size = setup_ddp(device)
        master = rank == 0

    preset = SHAKESPEARE if args.preset == 'shakespeare' else DEMO
    cfg = {**preset}
    if args.steps is not None:
        cfg['max_iters'] = args.steps
        cfg['warmup_iters'] = min(cfg['warmup_iters'], max(1, args.steps // 20))
        cfg['eval_interval'] = min(cfg['eval_interval'], args.steps)
        cfg['eval_iters'] = min(cfg['eval_iters'], max(4, args.steps))
    if args.batch is not None:
        cfg['batch_size'] = args.batch

    torch.manual_seed(1337 + rank)
    train_ids, val_ids, itos, vocab_size = load_shakespeare(master=master, ddp=ddp)
    splits = {'train': train_ids, 'val': val_ids}

    gpt_conf = GPTConfig(
        block_size=cfg['block_size'],
        vocab_size=vocab_size,
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_embd=cfg['n_embd'],
        dropout=cfg['dropout'],
    )
    raw = GPT(gpt_conf).to(device)
    if ddp:
        ddp_kwargs = {'device_ids': [device.index]} if device.type == 'cuda' else {}
        model: torch.nn.Module = DDP(raw, **ddp_kwargs)
    else:
        model = raw

    optimizer = torch.optim.AdamW(
        unwrap(model).parameters(),
        lr=cfg['learning_rate'],
        betas=(0.9, cfg['beta2']),
        weight_decay=0.1,
    )

    tokens_per_iter = cfg['batch_size'] * cfg['block_size'] * cfg['grad_accum'] * world_size
    if master:
        print(f'device: {device}  ddp: {ddp}  world_size: {world_size}  backend: '
              f'{"nccl" if device.type == "cuda" and ddp else ("gloo" if ddp else "-")}')
        print(f'preset: {args.preset}  params: {unwrap(model).n_params()/1e6:.2f}M  '
              f'vocab: {vocab_size}  train chars: {len(train_ids):,}')
        print(f'batch {cfg["batch_size"]} × ctx {cfg["block_size"]} × accum {cfg["grad_accum"]} '
              f'× ranks {world_size} = {tokens_per_iter:,} tokens/iter')
        print('Training...')

    t0 = time.time()
    raw_model = unwrap(model)
    model.train()
    for step in range(cfg['max_iters']):
        lr = get_lr(step, cfg['warmup_iters'], cfg['max_iters'], cfg['learning_rate'], cfg['min_lr'])
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for micro in range(cfg['grad_accum']):
            x, y = get_batch(train_ids, cfg['block_size'], cfg['batch_size'], device)
            # 累积步里只有最后一次让 DDP all-reduce，前面 no_sync 省掉多余通信
            sync = not (ddp and micro < cfg['grad_accum'] - 1)
            ctx = contextlib.nullcontext() if sync else model.no_sync()
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg['grad_accum']
                loss.backward()
            last_loss = loss
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if master and ((step + 1) % cfg['log_interval'] == 0 or step + 1 == cfg['max_iters']):
            dt = time.time() - t0
            t0 = time.time()
            print(f'step {step + 1:5d}/{cfg["max_iters"]}  loss {last_loss.item() * cfg["grad_accum"]:.4f}  '
                  f'lr {lr:.2e}  {dt:.2f}s/{cfg["log_interval"]}it')

        if master and ((step + 1) % cfg['eval_interval'] == 0 or step + 1 == cfg['max_iters']):
            losses = estimate_loss(
                model, splits, cfg['block_size'], cfg['batch_size'], cfg['eval_iters'], device,
            )
            print(f'eval  train {losses["train"]:.4f}  val {losses["val"]:.4f}')

    if master:
        ckpt_path = os.path.join(data_dir(), 'ckpt.pt')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            'model': raw_model.state_dict(),
            'config': gpt_conf.__dict__,
            'itos': itos,
        }, ckpt_path)
        print(f'saved {ckpt_path}')
        sample_and_print(raw_model, itos, device, args.prompt, args.tokens, args.temperature)
        print('Training finished!')

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def sample_only(args: argparse.Namespace) -> int:
    device = pick_device(False, 0, args.device)
    ckpt_path = os.path.join(data_dir(), 'ckpt.pt')
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt['config'])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt['model'])
    sample_and_print(model, ckpt['itos'], device, args.prompt, args.tokens, args.temperature)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='nanoGPT Shakespeare char + optional DDP.')
    parser.add_argument('--preset', choices=('demo', 'shakespeare'),
                        default=os.environ.get('PRESET', 'demo'))
    parser.add_argument('--steps', type=int, default=_env_int('STEPS'))
    parser.add_argument('--batch', type=int, default=_env_int('BATCH'))
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'auto'),
                        choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--prompt', default='ROMEO:\n')
    parser.add_argument('--tokens', type=int, default=int(os.environ.get('TOKENS', '300')))
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--nproc', type=int, default=int(os.environ.get('NPROC', '1')),
                        help='>1 时本进程 spawn DDP workers（已在 torchrun 里则忽略）')
    parser.add_argument('--sample-only', action='store_true')
    args = parser.parse_args(argv)
    if args.sample_only:
        return sample_only(args)
    if maybe_launch_ddp(args):
        return 0
    return train(args)


def _env_int(name: str) -> int | None:
    val = os.environ.get(name)
    return int(val) if val else None


if __name__ == '__main__':
    raise SystemExit(main())
