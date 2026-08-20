'''从 ckpt.pt 采样。改参数用 sample.sh，或直接：

  python3 sample.py --prompt "HAMLET:" --tokens 500 --temperature 0.8 --samples 3
'''

from __future__ import annotations

import argparse
import os

import torch

from model import GPT, GPTConfig
from train import (
    ckpt_path,
    data_dir,
    load_ckpt,
    pick_device,
    sample_and_print,
    strip_compile_prefix,
)


def load_model(device: torch.device) -> tuple[GPT, dict[int, str]]:
    import pickle

    path = ckpt_path()
    if not os.path.exists(path):
        raise SystemExit(f'找不到 {path}，先训练再采样')
    ckpt = load_ckpt(device)
    model_args = ckpt.get('model_args') or ckpt['config']
    keep = {
        k: model_args[k]
        for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout')
        if k in model_args
    }
    model = GPT(GPTConfig(**keep)).to(device)
    model.load_state_dict(strip_compile_prefix(ckpt['model']))
    if 'itos' in ckpt:
        itos = ckpt['itos']
    else:
        with open(os.path.join(data_dir(), 'meta.pkl'), 'rb') as f:
            itos = pickle.load(f)['itos']
    return model, itos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='nanoGPT Shakespeare 采样')
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'auto'),
                        choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--prompt', default=os.environ.get('PROMPT', 'ROMEO:\n'))
    parser.add_argument('--tokens', type=int, default=int(os.environ.get('TOKENS', '500')))
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('TEMPERATURE', '0.8')))
    parser.add_argument('--top-k', dest='top_k', type=int, default=int(os.environ.get('TOP_K', '200')))
    parser.add_argument('--samples', type=int, default=int(os.environ.get('SAMPLES', '3')))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '1337')))
    args = parser.parse_args(argv)

    device = pick_device(False, 0, args.device)
    torch.manual_seed(args.seed)
    model, itos = load_model(device)
    print(f'device: {device}  ckpt: {ckpt_path()}  samples: {args.samples}')
    for i in range(args.samples):
        print(f'===== sample {i + 1}/{args.samples} =====')
        sample_and_print(
            model, itos, device, args.prompt, args.tokens, args.temperature, args.top_k,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
