#!/usr/bin/env python3
"""
train_score_dermamnist.py

Standalone VESDE / NCSN++ score-model training script for DermaMNIST,
built to be dropped straight into your Score_Based_Lyapunov repo on
Colab (same folder as `models/ncsnpp.py`, `models/utils.py`, etc. --
the ones debug.py already imports successfully).

Why standalone: it does NOT depend on main.py/run_lib.py/sde_lib.py or
on TensorFlow / TFDS at all -- only on the model definition
(models/ncsnpp.py + friends) plus plain PyTorch + medmnist. That's the
minimal dependency set that is confirmed to exist in your repo, so this
should run without you needing to hunt down or rewrite the TF data
pipeline.

It trains with the standard (unregularized) denoising score matching
loss for VE-SDE -- i.e. plain USBM / vanilla score-based training, no
QCSBM/BSP-style Jacobian regularizer. That's the right choice here:
you just need *a* score model to plug into the Lyapunov Neural-ODE
purifier and test your method, not a research comparison of
regularizers on this dataset.

Budget knobs (all CLI flags, defaults chosen to be Colab-T4-cheap):
  --image_size   MUST be a power of 2 (NCSN++'s FIR resampling requires it).
                 224 is NOT a power of 2, so it cannot be used here even
                 though it's the "big" MedMNIST size you use for the
                 classifier. Use 64 for a cheap smoke test, or 256 to stay
                 close to the 224 scale. The classifier itself is unaffected
                 (resnet18 + adaptive pooling handles 224 fine) -- only the
                 score network needs this.
  --nf, --ch_mult, --num_res_blocks   control network size/cost.
  --n_iters, --batch_size             control wall-clock cost directly.

Usage (smoke test, ~10-15 min on a T4):
  python train_score_dermamnist.py \
      --image_size 64 --nf 32 --ch_mult 1,2,2,2 --num_res_blocks 1 \
      --attn_resolutions 16 --batch_size 32 --n_iters 3000 \
      --snapshot_freq 1000 --log_freq 50 \
      --out_dir /content/drive/MyDrive/lyap_derma_smoke

Usage (a bit more serious, still cheap, ~1-2h on a T4, closer to 224 scale):
  python train_score_dermamnist.py \
      --image_size 256 --nf 64 --ch_mult 1,1,2,2,2,2,2 --num_res_blocks 1 \
      --attn_resolutions 16 --batch_size 8 --n_iters 15000 \
      --snapshot_freq 2000 --log_freq 50 \
      --out_dir /content/drive/MyDrive/lyap_derma_small
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from ml_collections import ConfigDict
from medmnist import DermaMNIST
from torch.utils.data import DataLoader

# --- these three imports assume the standard score_sde_pytorch layout
# --- that debug.py already showed works in your repo (models/ncsnpp.py,
# --- models/utils.py, models/layers.py, models/layerspp.py,
# --- models/normalization.py, models/up_or_down_sampling.py).
import models  # noqa: F401  (registers submodules on import, as in debug.py)
from models import ncsnpp  # noqa: F401  (registers 'ncsnpp' in the model registry)
from models import utils as mutils


def build_config(args):
    # NCSN++'s FIR up/downsampling halves resolution at each level and needs
    # everything to divide out cleanly. 224 is NOT a power of 2 (224 = 32*7),
    # so a chain of >2 halvings lands on a non-integer/odd size and either
    # crashes or silently mismatches skip connections. 224 is fine for the
    # classifier (resnet18 + adaptive avgpool doesn't care), but for the
    # score network itself, resize to the nearest power of 2 instead --
    # 256 if you want to stay close to "big" MedMNIST, 64 for a cheap
    # smoke test. This script resizes DermaMNIST accordingly regardless of
    # what --image_size you pass, as long as it's a power of 2.
    n_levels = len(args.ch_mult.split(','))
    if args.image_size & (args.image_size - 1) != 0:
        raise ValueError(
            f'--image_size {args.image_size} is not a power of 2, which NCSN++\'s '
            'FIR up/downsampling requires. Use e.g. 64 (smoke test) or 256 '
            '(closer to the 224 "big" MedMNIST scale) instead. The classifier '
            'can stay at 224 -- only the score model needs a power-of-2 input.'
        )
    if 2 ** n_levels > args.image_size:
        raise ValueError(
            f'--ch_mult has {n_levels} levels, which downsamples below 1x1 for '
            f'image_size={args.image_size}. Reduce the number of comma-separated '
            f'entries in --ch_mult (e.g. "1,2,2,2" for image_size=64).'
        )

    config = ConfigDict()

    # ---- training ----
    config.training = training = ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.n_iters = args.n_iters
    training.batch_size = args.batch_size
    training.snapshot_freq = args.snapshot_freq
    training.log_freq = args.log_freq
    training.reduce_mean = True

    # ---- data ----
    config.data = data = ConfigDict()
    data.image_size = args.image_size
    data.num_channels = 3
    data.centered = True  # we map to [-1, 1] below, matches "centered" convention

    # ---- model (NCSN++ / VE-SDE hyperparams, Yang Song naming) ----
    config.model = model = ConfigDict()
    model.name = 'ncsnpp'
    model.sigma_min = args.sigma_min
    model.sigma_max = args.sigma_max
    model.num_scales = args.num_scales
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = args.nf
    model.ch_mult = tuple(int(c) for c in args.ch_mult.split(','))
    model.num_res_blocks = args.num_res_blocks
    model.attn_resolutions = tuple(int(a) for a in args.attn_resolutions.split(',')) if args.attn_resolutions else ()
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.0
    model.embedding_type = 'fourier'
    model.fourier_scale = 16
    model.conv_size = 3
    model.dropout = 0.0

    # ---- optimizer ----
    config.optim = optim_cfg = ConfigDict()
    optim_cfg.weight_decay = 0
    optim_cfg.optimizer = 'Adam'
    optim_cfg.lr = args.lr
    optim_cfg.beta1 = 0.9
    optim_cfg.eps = 1e-8
    optim_cfg.warmup = 500
    optim_cfg.grad_clip = 1.0

    config.seed = args.seed
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return config


def get_dermamnist_loader(image_size, batch_size, root, num_workers=2):
    def to_minus_one_one(x):
        return x * 2.0 - 1.0

    tf_list = [T.ToTensor()]
    if image_size != 224:
        tf_list = [T.Resize(image_size)] + tf_list
    tf_list.append(T.Lambda(to_minus_one_one))
    transform = T.Compose(tf_list)

    train_ds = DermaMNIST(root=root, split='train', size=224, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader


def get_sigmas(config):
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales)
    )
    return torch.tensor(sigmas, dtype=torch.float32)


def optimization_manager(config):
    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                     warmup=config.optim.warmup, grad_clip=config.optim.grad_clip):
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()
    return optimize_fn


def main(args):
    config = build_config(args)
    device = config.device
    print('Device:', device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    score_model = mutils.create_model(config)
    score_model = score_model.to(device)
    ema_params = [p.clone().detach() for p in score_model.parameters()]
    ema_rate = config.model.ema_rate

    optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr,
                            betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                            weight_decay=config.optim.weight_decay)
    optimize_fn = optimization_manager(config)

    sigmas = get_sigmas(config).to(device)

    start_step = 0
    resume_path = ckpt_dir / 'checkpoint_latest.pth'
    if resume_path.is_file():
        print('Resuming from', resume_path)
        state = torch.load(resume_path, map_location=device, weights_only=False)
        score_model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        ema_params = [p.to(device) for p in state['ema_params']]
        start_step = state['step']

    train_loader = get_dermamnist_loader(config.data.image_size, config.training.batch_size,
                                          root=args.data_root, num_workers=args.num_workers)

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite(train_loader)

    print(f'Starting training from step {start_step} to {config.training.n_iters}')
    t0 = time.time()
    score_model.train()
    for step in range(start_step, config.training.n_iters):
        x, _ = next(data_iter)
        x = x.to(device).float()

        # ---- standard VE-SDE denoising score matching loss (unregularized) ----
        labels = torch.randint(0, len(sigmas), (x.shape[0],), device=device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], 1, 1, 1)
        noise = torch.randn_like(x) * used_sigmas
        perturbed_x = x + noise
        target = -noise / (used_sigmas ** 2)

        score_fn = mutils.get_score_fn(config, score_model, train=True, continuous=config.training.continuous)
        t = labels.float() / (len(sigmas) - 1)
        score = score_fn(perturbed_x, t)

        losses = torch.square(score - target) * used_sigmas ** 2
        loss = torch.mean(torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        optimizer.zero_grad()
        loss.backward()
        optimize_fn(optimizer, score_model.parameters(), step + 1)

        with torch.no_grad():
            for p_ema, p in zip(ema_params, score_model.parameters()):
                p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)

        if (step + 1) % config.training.log_freq == 0:
            elapsed = time.time() - t0
            print(f'step {step+1}/{config.training.n_iters}  loss={loss.item():.3f}  '
                  f'elapsed={elapsed:.0f}s  ({elapsed/(step+1-start_step):.2f}s/it)')

        if (step + 1) % config.training.snapshot_freq == 0 or (step + 1) == config.training.n_iters:
            state = {
                'model': score_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema_params': [p.cpu() for p in ema_params],
                'step': step + 1,
                'config': config.to_dict(),
            }
            torch.save(state, ckpt_dir / f'checkpoint_{step+1}.pth')
            torch.save(state, resume_path)
            print('Saved checkpoint at step', step + 1)

    # Also dump a plain state_dict checkpoint using EMA weights -- this is
    # the format global_1_.py / evaluate.py's load_score_model() expects
    # (a bare model.state_dict(), no wrapper dict), so you don't have to
    # write any glue code on the evaluation side.
    ema_model = mutils.create_model(config).to(device)
    ema_model.load_state_dict(score_model.state_dict())
    sd = ema_model.state_dict()
    for (k, _), p_ema in zip(sd.items(), ema_params):
        sd[k] = p_ema
    ema_model.load_state_dict(sd)
    torch.save(ema_model.state_dict(), out_dir / 'score_dermamnist_final.pth')
    print('Saved final EMA-weights checkpoint to', out_dir / 'score_dermamnist_final.pth')
    print('Also saved the config used, so evaluate.py/global_1_.py can rebuild the same architecture:')
    import json
    with open(out_dir / 'score_dermamnist_config.json', 'w') as f:
        json.dump(config.to_dict(), f, default=str, indent=2)
    print(out_dir / 'score_dermamnist_config.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image_size', type=int, default=64,
                    help='must be a power of 2 (64, 128, 256, ...) -- see module docstring')
    p.add_argument('--nf', type=int, default=64)
    p.add_argument('--ch_mult', type=str, default='1,1,2,2,2,2,2',
                    help='comma-separated, one multiplier per downsampling level; '
                         'needs len(ch_mult) == log2(image_size)-1 roughly -- see notes below')
    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--attn_resolutions', type=str, default='16')
    p.add_argument('--sigma_min', type=float, default=0.01)
    p.add_argument('--sigma_max', type=float, default=50.0)
    p.add_argument('--num_scales', type=int, default=232)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_iters', type=int, default=15000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--snapshot_freq', type=int, default=2000)
    p.add_argument('--log_freq', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--out_dir', type=str, required=True)
    args = p.parse_args()
    main(args)
