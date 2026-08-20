'''
字符级 Tiny Shakespeare + 可选 DDP。

对齐 Karpathy nanoGPT：
  数据 = tinyshakespeare → train.bin / val.bin memmap（prepare.py 那套）
  配置 = config/train_shakespeare_char.py（--preset shakespeare）
  训练 = eval 间隔存盘（val 变好才写）、可 --resume、CUDA 上 amp
  分布式 = DDP all-reduce

单卡:    python3 train.py --preset shakespeare
续训:    python3 train.py --preset shakespeare --resume
多进程:  python3 train.py --nproc 2
'''

from __future__ import annotations

import argparse
import math
import os
import pickle
import socket
import sys
import time
import urllib.request
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

SHAKESPEARE_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
)

# karpathy/nanoGPT config/train_shakespeare_char.py + train.py 里没被覆盖的默认值
SHAKESPEARE = dict(
    n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2, bias=False,
    batch_size=64, max_iters=5000, grad_accum=1,
    learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=5000,
    beta1=0.9, beta2=0.99, weight_decay=0.1, grad_clip=1.0, decay_lr=True,
    eval_interval=250, eval_iters=200, log_interval=10,
    always_save_checkpoint=False,
)

DEMO = dict(
    n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0, bias=False,
    batch_size=32, max_iters=400, grad_accum=1,
    learning_rate=1e-3, min_lr=1e-4, warmup_iters=20, lr_decay_iters=400,
    beta1=0.9, beta2=0.99, weight_decay=0.1, grad_clip=1.0, decay_lr=True,
    eval_interval=100, eval_iters=20, log_interval=10,
    always_save_checkpoint=True,
)


# ---------------------------------------------------------------------------
# 1. 数据：input.txt → uint16 memmap（nanoGPT prepare.py）
# ---------------------------------------------------------------------------

def data_dir() -> str:
    env = os.environ.get('DDP_DATA_DIR')
    if env:
        return env
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def ckpt_path() -> str:
    return os.path.join(data_dir(), 'ckpt.pt')


def prepare_shakespeare(master: bool, ddp: bool) -> dict:
    root = data_dir()
    os.makedirs(root, exist_ok=True)
    text_path = os.path.join(root, 'input.txt')
    train_bin = os.path.join(root, 'train.bin')
    val_bin = os.path.join(root, 'val.bin')
    meta_path = os.path.join(root, 'meta.pkl')

    if not os.path.exists(text_path):
        if master:
            print(f'downloading tinyshakespeare → {text_path}')
            urllib.request.urlretrieve(SHAKESPEARE_URL, text_path)
        if ddp:
            dist.barrier()

    if master and (not os.path.exists(train_bin) or not os.path.exists(meta_path)):
        text = open(text_path, encoding='utf-8').read()
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        n = int(0.9 * len(text))
        train_ids = np.array([stoi[c] for c in text[:n]], dtype=np.uint16)
        val_ids = np.array([stoi[c] for c in text[n:]], dtype=np.uint16)
        train_ids.tofile(train_bin)
        val_ids.tofile(val_bin)
        with open(meta_path, 'wb') as f:
            pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)
        print(f'train {len(train_ids):,}  val {len(val_ids):,}  vocab {len(chars)}')
    if ddp:
        dist.barrier()

    with open(meta_path, 'rb') as f:
        return pickle.load(f)


def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """每次重建 memmap，避开 numpy 的泄漏。"""
    path = os.path.join(data_dir(), 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
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
    if is_ddp() or args.nproc <= 1:
        return False
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', _free_port())
    mp.spawn(_ddp_worker, args=(args.nproc, args), nprocs=args.nproc, join=True)
    return True


def amp_setup(device: torch.device):
    """CUDA：bf16 优先，否则 fp16+GradScaler。MPS/CPU：fp32。"""
    if device.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            dtype, ptdtype = 'bfloat16', torch.bfloat16
        else:
            dtype, ptdtype = 'float16', torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ctx: object = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        return dtype, ptdtype, ctx, scaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=False)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    return 'float32', torch.float32, nullcontext(), scaler


# ---------------------------------------------------------------------------
# 3. 学习率 / 评测 / 采样 / 存盘
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: dict) -> float:
    if not cfg['decay_lr']:
        return cfg['learning_rate']
    warmup, decay_iters = cfg['warmup_iters'], cfg['lr_decay_iters']
    lr, min_lr = cfg['learning_rate'], cfg['min_lr']
    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    if step > decay_iters:
        return min_lr
    decay = (step - warmup) / max(1, decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    ctx: object,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ('train', 'val'):
        losses = torch.zeros(cfg['eval_iters'])
        for i in range(cfg['eval_iters']):
            x, y = get_batch(split, cfg['block_size'], cfg['batch_size'], device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def strip_compile_prefix(state_dict: dict) -> dict:
    prefix = '_orig_mod.'
    for key in list(state_dict):
        if key.startswith(prefix):
            state_dict[key[len(prefix):]] = state_dict.pop(key)
    return state_dict


def save_checkpoint(
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    model_args: dict,
    cfg: dict,
    iter_num: int,
    best_val_loss: float,
    itos: dict,
) -> None:
    path = ckpt_path()
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': cfg,
        'itos': itos,
    }, path)
    print(f'saving checkpoint to {path}')


def sample_and_print(
    model: GPT,
    itos: dict[int, str],
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = 200,
) -> None:
    stoi = {ch: i for i, ch in itos.items()}
    idx = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = ''.join(itos[int(t)] for t in out[0].tolist())
    print('\n----- sample -----')
    print(text)
    print('------------------\n')


def load_ckpt(device: torch.device) -> dict:
    path = ckpt_path()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# 4. 训练（循环结构跟 nanoGPT train.py）
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
    _apply_cli(cfg, args)

    # 原版：accum 能被 world_size 整除就平分到每卡。shakespeare 默认 accum=1，2 卡时保持 1（全局 batch 翻倍）
    if ddp and cfg['grad_accum'] % world_size == 0 and cfg['grad_accum'] >= world_size:
        cfg['grad_accum'] //= world_size

    dtype, _, ctx, scaler = amp_setup(device)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    torch.manual_seed(1337 + rank)

    meta = prepare_shakespeare(master=master, ddp=ddp)
    itos = meta['itos']
    vocab_size = meta['vocab_size']

    model_args = dict(
        n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embd=cfg['n_embd'],
        block_size=cfg['block_size'], bias=cfg['bias'], vocab_size=vocab_size,
        dropout=cfg['dropout'],
    )
    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None

    if args.resume:
        if not os.path.exists(ckpt_path()):
            print(f'no checkpoint at {ckpt_path()}', file=sys.stderr)
            if ddp:
                dist.destroy_process_group()
            return 1
        checkpoint = load_ckpt(device)
        if 'model_args' not in checkpoint or 'optimizer' not in checkpoint:
            print('旧 ckpt 缺 model_args/optimizer，无法续训，请重新跑一遍', file=sys.stderr)
            if ddp:
                dist.destroy_process_group()
            return 1
        for key in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'):
            model_args[key] = checkpoint['model_args'][key]
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        if 'itos' in checkpoint:
            itos = checkpoint['itos']

    gpt_conf = GPTConfig(**model_args)
    raw = GPT(gpt_conf)
    if args.resume:
        raw.load_state_dict(strip_compile_prefix(checkpoint['model']))
    raw.to(device)

    optimizer = raw.configure_optimizers(
        cfg['weight_decay'], cfg['learning_rate'], (cfg['beta1'], cfg['beta2']), device_type,
    )
    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    if args.compile:
        print('compiling the model...')
        raw = torch.compile(raw)

    model: torch.nn.Module = raw
    if ddp:
        ddp_kwargs = {'device_ids': [device.index]} if device.type == 'cuda' else {}
        model = DDP(raw, **ddp_kwargs)
    raw_model = unwrap(model)

    tokens_per_iter = cfg['batch_size'] * cfg['block_size'] * cfg['grad_accum'] * world_size
    if master:
        print(f'device: {device}  dtype: {dtype}  ddp: {ddp}  world_size: {world_size}')
        print(f'preset: {args.preset}  params: {raw_model.n_params()/1e6:.2f}M  '
              f'vocab: {vocab_size}  resume: {args.resume} iter {iter_num}')
        print(f'batch {cfg["batch_size"]} × ctx {cfg["block_size"]} × accum {cfg["grad_accum"]} '
              f'× ranks {world_size} = {tokens_per_iter:,} tokens/iter')
        print('Training...')

    x, y = get_batch('train', cfg['block_size'], cfg['batch_size'], device)
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    model.train()

    while True:
        lr = get_lr(iter_num, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr

        if iter_num % cfg['eval_interval'] == 0 and master:
            losses = estimate_loss(model, cfg, device, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            improved = losses['val'] < best_val_loss
            if improved:
                best_val_loss = losses['val']
            if iter_num > 0 and (improved or cfg['always_save_checkpoint']):
                save_checkpoint(
                    raw_model, optimizer, model_args, cfg, iter_num, best_val_loss, itos,
                )

        for micro in range(cfg['grad_accum']):
            if ddp:
                model.require_backward_grad_sync = (micro == cfg['grad_accum'] - 1)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg['grad_accum']
            x, y = get_batch('train', cfg['block_size'], cfg['batch_size'], device)
            scaler.scale(loss).backward()
        if cfg['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if master and iter_num % cfg['log_interval'] == 0:
            lossf = loss.item() * cfg['grad_accum']
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(cfg['batch_size'] * cfg['grad_accum'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f'step {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, '
                  f'{dt * 1000:.0f}ms, mfu {running_mfu * 100:.2f}%')

        iter_num += 1
        local_iter_num += 1
        if iter_num > cfg['max_iters']:
            break

    if master:
        sample_and_print(
            raw_model, itos, device, args.prompt, args.tokens, args.temperature, args.top_k,
        )
        print('Training finished!')

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def sample_only(args: argparse.Namespace) -> int:
    device = pick_device(False, 0, args.device)
    ckpt = load_ckpt(device)
    model_args = ckpt.get('model_args') or ckpt['config']
    keep = {k: model_args[k] for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout')
            if k in model_args}
    model = GPT(GPTConfig(**keep)).to(device)
    model.load_state_dict(strip_compile_prefix(ckpt['model']))
    if 'itos' in ckpt:
        itos = ckpt['itos']
    else:
        with open(os.path.join(data_dir(), 'meta.pkl'), 'rb') as f:
            itos = pickle.load(f)['itos']
    sample_and_print(model, itos, device, args.prompt, args.tokens, args.temperature, args.top_k)
    return 0


def _apply_cli(cfg: dict, args: argparse.Namespace) -> None:
    mapping = (
        ('batch', 'batch_size'),
        ('lr', 'learning_rate'),
        ('min_lr', 'min_lr'),
        ('warmup', 'warmup_iters'),
        ('lr_decay_iters', 'lr_decay_iters'),
        ('n_layer', 'n_layer'),
        ('n_head', 'n_head'),
        ('n_embd', 'n_embd'),
        ('block_size', 'block_size'),
        ('dropout', 'dropout'),
        ('eval_interval', 'eval_interval'),
        ('eval_iters', 'eval_iters'),
        ('log_interval', 'log_interval'),
        ('grad_accum', 'grad_accum'),
    )
    for attr, key in mapping:
        val = getattr(args, attr)
        if val is not None:
            cfg[key] = val
    if args.steps is not None:
        cfg['max_iters'] = args.steps
        if args.lr_decay_iters is None:
            cfg['lr_decay_iters'] = args.steps
        if args.warmup is None:
            cfg['warmup_iters'] = min(cfg['warmup_iters'], max(1, args.steps // 20))
        if args.eval_interval is None:
            cfg['eval_interval'] = min(cfg['eval_interval'], max(1, args.steps))
        if args.eval_iters is None:
            cfg['eval_iters'] = min(cfg['eval_iters'], max(4, args.steps))
        if args.log_interval is None:
            cfg['log_interval'] = min(cfg['log_interval'], max(1, args.steps))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='nanoGPT Shakespeare char + optional DDP.')
    parser.add_argument('--preset', choices=('demo', 'shakespeare'),
                        default=os.environ.get('PRESET', 'demo'))
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'auto'),
                        choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--nproc', type=int, default=int(os.environ.get('NPROC', '1')))
    parser.add_argument('--resume', action='store_true', default=_env_flag('RESUME'))
    parser.add_argument('--compile', action='store_true', default=_env_flag('COMPILE'))
    parser.add_argument('--steps', type=int, default=_env_int('STEPS'))
    parser.add_argument('--batch', type=int, default=_env_int('BATCH'))
    parser.add_argument('--lr', type=float, default=_env_float('LR'))
    parser.add_argument('--min-lr', dest='min_lr', type=float, default=_env_float('MIN_LR'))
    parser.add_argument('--warmup', type=int, default=_env_int('WARMUP'))
    parser.add_argument('--lr-decay-iters', dest='lr_decay_iters', type=int, default=_env_int('LR_DECAY_ITERS'))
    parser.add_argument('--n-layer', dest='n_layer', type=int, default=_env_int('N_LAYER'))
    parser.add_argument('--n-head', dest='n_head', type=int, default=_env_int('N_HEAD'))
    parser.add_argument('--n-embd', dest='n_embd', type=int, default=_env_int('N_EMBD'))
    parser.add_argument('--block-size', dest='block_size', type=int, default=_env_int('BLOCK_SIZE'))
    parser.add_argument('--dropout', type=float, default=_env_float('DROPOUT'))
    parser.add_argument('--eval-interval', dest='eval_interval', type=int, default=_env_int('EVAL_INTERVAL'))
    parser.add_argument('--eval-iters', dest='eval_iters', type=int, default=_env_int('EVAL_ITERS'))
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=_env_int('LOG_INTERVAL'))
    parser.add_argument('--grad-accum', dest='grad_accum', type=int, default=_env_int('GRAD_ACCUM'))
    parser.add_argument('--prompt', default=os.environ.get('PROMPT', 'ROMEO:\n'))
    parser.add_argument('--tokens', type=int, default=int(os.environ.get('TOKENS', '300')))
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('TEMPERATURE', '0.8')))
    parser.add_argument('--top-k', dest='top_k', type=int, default=int(os.environ.get('TOP_K', '200')))
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


def _env_float(name: str) -> float | None:
    val = os.environ.get(name)
    return float(val) if val else None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '').lower() in ('1', 'true', 'yes')


if __name__ == '__main__':
    raise SystemExit(main())
