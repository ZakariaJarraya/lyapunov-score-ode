#!/usr/bin/env python3
"""
plot_latency_vs_tf_sigma.py

Checks empirically whether the purifier's cost depends only on the ratio
t_f/sigma (your paper's claim), or on t_f and sigma independently. Reports
BOTH wall-clock latency AND the raw NFE (number of vector-field evaluations)
-- NFE is the theoretically-grounded quantity (no CPU/GPU scheduling noise),
latency is what you'd actually report as "inference time".

Produces a 2-panel plot per metric:
  (A) t_f/sigma held CONSTANT (t_f scaled together with sigma) -> flat.
  (B) sigma held FIXED, t_f varied alone -> should move with t_f.

IMPORTANT: this only produces a meaningful (B) curve if NeuralODE uses an
ADAPTIVE solver (dopri5). With the fixed-step 'rk4' method and a 2-point
time grid [0, T] (evaluate.py's original default), torchdiffeq takes
exactly ONE step regardless of T -- so NFE (and latency) is *always* flat,
even for curve (B), which is a null result, not evidence for anything. Make
sure your patched evaluate.py's NeuralODE defaults to method='dopri5'
before trusting this script's curve (B).

Reuses evaluate.py's own loading code (load_classifier, load_score_model,
ScoreVectorField, NeuralODE, PurifiedClassifier) so the purifier measured
here is byte-for-byte identical to the one evaluate.py uses for attacks.

Usage:
  python plot_latency_vs_tf_sigma.py \
      --dataset dermamnist \
      --classifier_ckpt derma_resnet18_ckpt.pth \
      --score_ckpt workdir_64/checkpoints-meta/checkpoint.pth \
      --score_config configs.score_configs.ve.dermamnist_ncsnpp_continuous_64 \
      --classifier_input_size 224 \
      --ode_method dopri5 \
      --ratio 2e-7 \
      --sigma_values 0.02,0.05,0.1,0.2,0.4 \
      --fixed_sigma 0.1 \
      --tf_values 1e-8,2e-8,4e-8,8e-8,1.6e-7 \
      --batch_size 8 --n_batches 5 --n_samples 40 \
      --out latency_vs_tf_sigma.pdf
"""
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate import (load_classifier, load_score_model, ScoreVectorField,
                       NeuralODE, PurifiedClassifier, DEVICE)
from models.classifier.classifier import get_classifier, adapt_model
from my_datasets.datasets import get_dataset


class _NFECountingVF(torch.nn.Module):
    """Wraps a vector field module and counts how many times it's called --
    that count IS the NFE torchdiffeq actually used for one odeint call."""
    def __init__(self, vf):
        super().__init__()
        self.vf = vf
        self.calls = 0

    def forward(self, t, x):
        self.calls += 1
        return self.vf(t, x)


def build_model(classifier, score_model, sigma, ode_T, t_noise, denormalize, normalize,
                 ode_method, classifier_input_size):
    vf = ScoreVectorField(score_model, t_noise).to(DEVICE)
    counting_vf = _NFECountingVF(vf)
    ode = NeuralODE(counting_vf, ode_T, method=ode_method).to(DEVICE)
    model = PurifiedClassifier(denormalize, ode, normalize, classifier, sigma,
                                classifier_input_size=classifier_input_size).to(DEVICE)
    model.eval()
    return model, counting_vf


def measure(model, counting_vf, loader, n_batches):
    model.eval()
    times, nfes = [], []
    with torch.no_grad():
        it = iter(loader)
        for i in range(n_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, _ = next(it)
            x = x.to(DEVICE)
            counting_vf.calls = 0
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(x, mode='classify')
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            nfes.append(counting_vf.calls)
    return float(np.mean(times)), float(np.std(times)), float(np.mean(nfes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='dermamnist')
    p.add_argument('--classifier', type=str, default='resnet18')
    p.add_argument('--classifier_ckpt', type=str, required=True)
    p.add_argument('--score_ckpt', type=str, required=True)
    p.add_argument('--score_config', type=str, required=True)
    p.add_argument('--classifier_input_size', type=int, default=224,
                    help='resolution the classifier was trained at; purified output is '
                         'resized to this before classification (must match PurifiedClassifier patch)')
    p.add_argument('--ode_method', type=str, default='dopri5',
                    help="'dopri5' (adaptive, required for curve B to mean anything) or 'rk4' (fixed, always flat)")
    p.add_argument('--t_noise', type=float, default=1e-7)
    p.add_argument('--ratio', type=float, required=True,
                    help='t_f / sigma ratio to hold constant for curve (A)')
    p.add_argument('--sigma_values', type=str, required=True,
                    help='comma-separated sigma values for curve (A); t_f = ratio * sigma each time')
    p.add_argument('--fixed_sigma', type=float, required=True,
                    help='sigma held fixed for curve (B)')
    p.add_argument('--tf_values', type=str, required=True,
                    help='comma-separated t_f values for curve (B), sigma fixed at --fixed_sigma')
    p.add_argument('--fixed_tf', type=float, default=None,
                    help='t_f held fixed for curve (C): sigma varied ALONE, t_f constant. '
                         'Needed to separate "sigma damps the t_f effect" from "sigma has its '
                         'own independent effect" -- without this curve you cannot tell them apart '
                         'from curve (A) alone. Omit to skip curve (C).')
    p.add_argument('--sigma_values_C', type=str, default=None,
                    help='comma-separated sigma values for curve (C); defaults to --sigma_values if omitted')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_samples', type=int, default=64)
    p.add_argument('--n_batches', type=int, default=5)
    p.add_argument('--out', type=str, default='latency_vs_tf_sigma.pdf')
    args = p.parse_args()

    print('Device:', DEVICE)
    if args.ode_method == 'rk4':
        print("WARNING: --ode_method rk4 uses a single fixed step regardless of t_f -- "
              "curve (B) will be flat by construction, not because the claim is false. "
              "Use --ode_method dopri5 to actually test it.")

    base_model = get_classifier(args.classifier).to(DEVICE)
    classifier = adapt_model(base_model, args.dataset)
    classifier = load_classifier(classifier, args.classifier_ckpt)

    score_model, cfg = load_score_model(args.score_ckpt, args.score_config, DEVICE)
    score_image_size = cfg.data.image_size

    # Load at the SCORE MODEL's resolution -- same fix as evaluate.py needs
    # (feeding it the dataset's native size desyncs NCSN++'s internal
    # module indexing if it differs from cfg.data.image_size).
    dataset, normalize, denormalize = get_dataset(args.dataset, train=False, image_size=score_image_size)
    normalize = normalize.to(DEVICE)
    denormalize = denormalize.to(DEVICE)
    if args.n_samples < len(dataset):
        idx = np.random.choice(len(dataset), args.n_samples, replace=False)
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sigma_values = [float(s) for s in args.sigma_values.split(',')]
    tf_values = [float(s) for s in args.tf_values.split(',')]

    # ---- Curve A: t_f/sigma held constant ----
    latencies_A, nfes_A = [], []
    for sigma in sigma_values:
        t_f = args.ratio * sigma
        model, cvf = build_model(classifier, score_model, sigma, t_f, args.t_noise,
                                  denormalize, normalize, args.ode_method, args.classifier_input_size)
        mean_t, std_t, mean_nfe = measure(model, cvf, loader, args.n_batches)
        print(f'[A] sigma={sigma:.4g}  t_f={t_f:.4g}  t_f/sigma={t_f/sigma:.4g}  '
              f'latency={mean_t*1000:.2f}+/-{std_t*1000:.2f} ms/batch  NFE={mean_nfe:.1f}')
        latencies_A.append(mean_t * 1000)
        nfes_A.append(mean_nfe)

    # ---- Curve B: sigma fixed, t_f varied alone ----
    latencies_B, nfes_B = [], []
    for t_f in tf_values:
        model, cvf = build_model(classifier, score_model, args.fixed_sigma, t_f, args.t_noise,
                                  denormalize, normalize, args.ode_method, args.classifier_input_size)
        mean_t, std_t, mean_nfe = measure(model, cvf, loader, args.n_batches)
        print(f'[B] sigma={args.fixed_sigma:.4g}  t_f={t_f:.4g}  t_f/sigma={t_f/args.fixed_sigma:.4g}  '
              f'latency={mean_t*1000:.2f}+/-{std_t*1000:.2f} ms/batch  NFE={mean_nfe:.1f}')
        latencies_B.append(mean_t * 1000)
        nfes_B.append(mean_nfe)

    # ---- Curve C (optional): t_f fixed, sigma varied ALONE ----
    # This is what actually separates "sigma damps the t_f effect" (your
    # claim) from "sigma has its own independent effect on cost that has
    # nothing to do with t_f". Curve A alone cannot distinguish these,
    # because in A both t_f and sigma move together.
    sigma_values_C, latencies_C, nfes_C = None, [], []
    if args.fixed_tf is not None:
        sigma_values_C = [float(s) for s in (args.sigma_values_C or args.sigma_values).split(',')]
        for sigma in sigma_values_C:
            model, cvf = build_model(classifier, score_model, sigma, args.fixed_tf, args.t_noise,
                                      denormalize, normalize, args.ode_method, args.classifier_input_size)
            mean_t, std_t, mean_nfe = measure(model, cvf, loader, args.n_batches)
            print(f'[C] sigma={sigma:.4g}  t_f={args.fixed_tf:.4g}  t_f/sigma={args.fixed_tf/sigma:.4g}  '
                  f'latency={mean_t*1000:.2f}+/-{std_t*1000:.2f} ms/batch  NFE={mean_nfe:.1f}')
            latencies_C.append(mean_t * 1000)
            nfes_C.append(mean_nfe)

    ncols = 3 if args.fixed_tf is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(5.3 * ncols, 8.5))

    axes[0, 0].plot(sigma_values, latencies_A, marker='o')
    axes[0, 0].set_xlabel(r'$\sigma$  (with $t_f = %.3g \cdot \sigma$ held fixed)' % args.ratio)
    axes[0, 0].set_ylabel('Latency (ms/batch)')
    axes[0, 0].set_title(r'(A) ratio $t_f/\sigma$ constant')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(tf_values, latencies_B, marker='o', color='tab:orange')
    axes[0, 1].set_xlabel(r'$t_f$  (with $\sigma = %.3g$ fixed)' % args.fixed_sigma)
    axes[0, 1].set_ylabel('Latency (ms/batch)')
    axes[0, 1].set_title(r'(B) $t_f$ alone varies')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(sigma_values, nfes_A, marker='s', color='tab:green')
    axes[1, 0].set_xlabel(r'$\sigma$  (with $t_f = %.3g \cdot \sigma$ held fixed)' % args.ratio)
    axes[1, 0].set_ylabel('NFE')
    axes[1, 0].set_title(r'(A) NFE vs $\sigma$, ratio constant')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(tf_values, nfes_B, marker='s', color='tab:red')
    axes[1, 1].set_xlabel(r'$t_f$  (with $\sigma = %.3g$ fixed)' % args.fixed_sigma)
    axes[1, 1].set_ylabel('NFE')
    axes[1, 1].set_title(r'(B) NFE vs $t_f$, $\sigma$ fixed')
    axes[1, 1].grid(alpha=0.3)

    if args.fixed_tf is not None:
        axes[0, 2].plot(sigma_values_C, latencies_C, marker='o', color='tab:purple')
        axes[0, 2].set_xlabel(r'$\sigma$  (with $t_f = %.3g$ fixed)' % args.fixed_tf)
        axes[0, 2].set_ylabel('Latency (ms/batch)')
        axes[0, 2].set_title(r'(C) $\sigma$ alone varies, $t_f$ fixed')
        axes[0, 2].grid(alpha=0.3)

        axes[1, 2].plot(sigma_values_C, nfes_C, marker='s', color='tab:brown')
        axes[1, 2].set_xlabel(r'$\sigma$  (with $t_f = %.3g$ fixed)' % args.fixed_tf)
        axes[1, 2].set_ylabel('NFE')
        axes[1, 2].set_title(r'(C) NFE vs $\sigma$, $t_f$ fixed')
        axes[1, 2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out)
    print('Saved plot to', args.out)
    print()
    print('How to read this once curve (C) is included:')
    print('  - if (C) is flat (sigma alone does not change cost) and (A) grows slower than (B)')
    print('    for a comparable t_f range -> genuine support for a t_f/sigma coupling.')
    print('  - if (C) moves on its own -> sigma has an independent effect on cost, and (A)\'s')
    print('    slower growth vs (B) may just be that independent effect, not a ratio effect.')


if __name__ == '__main__':
    main()