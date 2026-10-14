import sys
import os
from pathlib import Path

# Fix the path before importing run_lib
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))
    
import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import importlib

from models.classifier.classifier import *
from my_datasets.datasets import get_dataset

from torchdiffeq import odeint_adjoint as odeint
import torchattacks
from autoattack import AutoAttack


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# ----------- SONG SCORE MODEL LOADING (UNCHANGED) ----------
# =========================================================

def register_song_models():
    import models.score_models.ddpm
    import models.score_models.ncsnv2
    import models.score_models.ncsnpp


def safe_load_state_dict(model, ckpt_dict, strict=True):
    if isinstance(ckpt_dict, dict) and 'model' in ckpt_dict:
        ckpt_dict = ckpt_dict['model']

    new_sd = {}
    for k, v in ckpt_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=strict)
    return True


def load_score_model(score_ckpt,
                     config_module="configs.score_configs.ve.cifar10_ncsnpp_continuous",
                     device="cpu"):

    register_song_models()
    import models.score_models.utils as mutils

    cfg_mod = importlib.import_module(config_module)
    cfg = cfg_mod.get_config()

    cfg.device = torch.device(device)

    score_model = mutils.create_model(cfg)
    # 🔥 ajouter ça
    if isinstance(score_model, torch.nn.DataParallel):
        score_model = score_model.module
    score_model = score_model.to(device)
    score_model.eval()

    if not os.path.exists(score_ckpt):
        raise FileNotFoundError(f"Score checkpoint not found: {score_ckpt}")

    ckpt = torch.load(score_ckpt, weights_only=False, map_location=device)

    ok = safe_load_state_dict(score_model, ckpt, strict=False)
    if not ok:
        raise RuntimeError("Failed to load score model checkpoint.")

    print("Loaded score model from:", score_ckpt)
    return score_model, cfg


# =========================================================
# ---------------- CLASSIFIER LOADING ----------------------
# =========================================================

def load_classifier(model, path):
    ckpt = torch.load(path, map_location=DEVICE)

    # --- cas DiffPure / checkpoints structurés ---
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "net" in ckpt:
            ckpt = ckpt["net"]
        elif "model" in ckpt:          # <-- AJOUT : format de train_cifar_classifier.py
            ckpt = ckpt["model"]

    # --- remove "module." (DataParallel) ---
    state_dict = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded classifier:", path)
    return model


# =========================================================
# ---------------- ODE + SCORE WRAPPER ---------------------
# =========================================================

class ScoreVectorField(nn.Module):
    def __init__(self, score_model, t_noise=0.2):
        super().__init__()
        self.score_model = score_model
        self.t_noise = t_noise

    def forward(self, t, x):
        t_vec = torch.ones(x.shape[0], device=x.device) * self.t_noise
        return self.score_model(x, t_vec)


class NeuralODE(nn.Module):
    def __init__(self, vf, T=1.0, method='dopri5', rtol=1e-3, atol=1e-3):
        super().__init__()
        self.vf = vf
        self.T = T
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        t = torch.tensor([0.0, self.T], device=x.device)
        out = odeint(self.vf, x, t, method=self.method, rtol=self.rtol, atol=self.atol)
        return out[-1]
##


class PurifiedClassifier(nn.Module):
    def __init__(self, denormalize, purifier, normalize, classifier, sigma,
                 classifier_input_size=None):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.sigma = sigma
        self.denormalize = denormalize
        self.normalize = normalize
        # If the score model runs at a different resolution than the
        # classifier was trained at (e.g. score model at 64px, classifier
        # at 224px), resize the purified output before classifying.
        # None/unset -> no resize (same resolution end to end).
        self.classifier_input_size = classifier_input_size

    def add_noise(self, x):
        return x + torch.randn_like(x) * self.sigma

    def purify(self, x):
        x = self.add_noise(x)
        x = self.purifier(x)
        return x

    def classify(self, x):
        if self.classifier_input_size is not None and x.shape[-1] != self.classifier_input_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.classifier_input_size, self.classifier_input_size),
                mode='bilinear', align_corners=False
            )
        return self.classifier(x)

    def forward(self, x, mode="classify"):
        if mode == "purify":
            return self.purify(x)
        elif mode == "classify":
            x = self.purify(x)
            return self.classify(x)
        else:
            raise ValueError("mode must be 'purify' or 'classify'")


'''
class PurifiedClassifier(nn.Module):
    def __init__(self, denormalize, purifier, normalize, classifier, sigma):
        super().__init__()
        self.denormalize = denormalize
        self.purifier = purifier
        self.normalize = normalize
        self.classifier = classifier
        self.sigma = sigma

    def forward(self, x):
        # normalized → [0,1]
        #x = self.denormalize(x)
        #x = (x-0.5) * 2
        # purification (score model domain)
        noise = torch.randn_like(x) * self.sigma
        x = x + noise
        x = self.purifier(x)
        # safety clamp
        #x = x.clamp(0.0, 1.0)
        #x = (x+1)/2

        # back to classifier domain
        #x = self.normalize(x)

        return self.classifier(x)

'''

class BPDA_EOT_Attack():
    def __init__(self, model, adv_eps=8/255, eot_defense_reps=5, eot_attack_reps=5):
        self.model = model

        self.config = {
            'eot_defense_ave': 'logits',
            'eot_attack_ave': 'loss',
            'eot_defense_reps': eot_defense_reps,
            'eot_attack_reps': eot_attack_reps,
            'adv_steps': 20,
            'adv_norm': 'l_inf',
            'adv_eps': adv_eps,
            'adv_eta': 2/255,
        }

        self.criterion = nn.CrossEntropyLoss()

    def purify(self, x):
        return self.model(x, mode='purify')

    def predict(self, x, y):
        logits = self.model(x, mode='classify')
        loss = self.criterion(logits, y)
        pred = logits.argmax(1)
        correct = (pred == y)
        return correct, loss

    def purify_and_predict(self, x, y, reps, requires_grad=True):
        x_rep = x.repeat(reps, 1, 1, 1)

        # 🔥 BPDA trick: detach purification
        x_pur = self.purify(x_rep).detach().clone()
        x_pur.requires_grad_(True)

        logits = self.model.classify(x_pur)

        loss = self.criterion(logits, y.repeat(reps))

        if requires_grad:
            grad = torch.autograd.grad(loss, x_pur)[0]
            grad = grad.view(reps, *x.shape).mean(0)
            return grad
        else:
            pred = logits.argmax(1)
            pred = pred.view(reps, -1).mode(0)[0]
            return pred

    def attack(self, x, y):
        x_adv = x.clone()

        for _ in range(self.config['adv_steps']):
            grad = self.purify_and_predict(
                x_adv, y,
                reps=self.config['eot_attack_reps'],
                requires_grad=True
            )

            # PGD step
            x_adv = x_adv + self.config['adv_eta'] * grad.sign()

            # projection
            x_adv = torch.max(torch.min(x_adv, x + self.config['adv_eps']),
                              x - self.config['adv_eps'])
            #x_adv = x_adv.clamp(0, 1).detach()
            x_adv = x_adv.detach()

        return x_adv


class BPDA_EOT_Attack():
    def __init__(self, model, adv_eps=8/255, eot_attack_reps=10):
        self.model = model

        self.config = {
            'adv_steps': 20,
            'adv_eps': adv_eps,
            'adv_eta': 2/255,
            'eot_attack_reps': eot_attack_reps,
        }

        self.criterion = nn.CrossEntropyLoss()

    def purify(self, x):
        return self.model(x, mode='purify')

    def classify(self, x):
        return self.model(x, mode='classify')

    def attack(self, x, y):
        x_adv = x.clone().detach()

        for step in range(self.config['adv_steps']):
            grad = torch.zeros_like(x_adv)

            # -----------------------------
            # EOT loop
            # -----------------------------
            for _ in range(self.config['eot_attack_reps']):

                # 1. Purify (stochastic)
                x_pur = self.purify(x_adv).detach()

                # 2. Compute gradient THROUGH classifier ONLY
                x_pur.requires_grad_(True)
                logits = self.classify(x_pur)
                loss = self.criterion(logits, y)

                grad_clf = torch.autograd.grad(loss, x_pur)[0]

                # 3. PGD step in purified space
                x_pur_adv = x_pur + self.config['adv_eta'] * grad_clf.sign()

                # 4. BPDA gradient approximation (CRUCIAL LINE)
                grad += (x_pur_adv.detach() - x_pur)

            # Average EOT gradients
            grad = grad / self.config['eot_attack_reps']

            # -----------------------------
            # PGD update in ORIGINAL space
            # -----------------------------
            x_adv = x_adv + self.config['adv_eta'] * grad.sign()

            # Projection (Linf)
            x_adv = torch.max(torch.min(x_adv, x + self.config['adv_eps']),
                              x - self.config['adv_eps'])

            x_adv = x_adv.clamp(0, 1).detach()

        return x_adv 


class BPDA_EOT_Attack():
    def __init__(self, model, adv_eps=8/255, eot_attack_reps=10):
        self.model = model

        self.config = {
            'adv_steps': 20,
            'adv_eps': adv_eps,
            'adv_eta': 2/255,
            'eot_attack_reps': eot_attack_reps,

            # 🔥 NEW (inner attack strength)
            'inner_steps': 5,
            'inner_eta': 2/255,
        }

        self.criterion = nn.CrossEntropyLoss()

    def purify(self, x):
        return self.model(x, mode='purify')

    def classify(self, x):
        return self.model(x, mode='classify')

    def inner_pgd(self, x, y):
        """
        PGD in purified space (replaces 1-step FGSM)
        """
        x_adv = x.clone().detach()

        for _ in range(self.config['inner_steps']):
            x_adv.requires_grad_(True)

            logits = self.classify(x_adv)
            loss = self.criterion(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            x_adv = x_adv + self.config['inner_eta'] * grad.sign()
            x_adv = x_adv.clamp(0, 1).detach()

        return x_adv

    def attack(self, x, y):
        x_adv = x.clone().detach()

        for step in range(self.config['adv_steps']):
            grad = torch.zeros_like(x_adv)

            # -----------------------------
            # EOT loop
            # -----------------------------
            for _ in range(self.config['eot_attack_reps']):

                # 1. Purify
                x_pur = self.purify(x_adv).detach()

                # 2. Strong inner attack (🔥 replaces FGSM)
                x_pur_adv = self.inner_pgd(x_pur, y)

                # 3. BPDA gradient
                grad += (x_pur_adv.detach() - x_pur)

            # Average
            grad = grad / self.config['eot_attack_reps']

            # -----------------------------
            # PGD update in ORIGINAL space
            # -----------------------------
            x_adv = x_adv + self.config['adv_eta'] * grad.sign()

            # Projection
            x_adv = torch.max(torch.min(x_adv, x + self.config['adv_eps']),
                              x - self.config['adv_eps'])

            x_adv = x_adv.clamp(0, 1).detach()

        return x_adv

import torch.nn.functional as F

class STAdvAttack:
    def __init__(self, model, eps=0.05, steps=20, lr=0.01):
        self.model = model
        self.eps = eps          # max spatial displacement
        self.steps = steps
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def _flow_warp(self, x, flow):
        B, C, H, W = x.shape

        # base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )

        grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        # apply flow
        flow = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        new_grid = grid + flow

        return F.grid_sample(x, new_grid, align_corners=True)

    def attack(self, x, y):
        B, C, H, W = x.shape

        # flow field (dx, dy)
        flow = torch.zeros(B, 2, H, W, device=x.device, requires_grad=True)

        optimizer = torch.optim.Adam([flow], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad()

            # constrain flow
            flow_clamped = torch.clamp(flow, -self.eps, self.eps)

            # warp image
            x_adv = self._flow_warp(x, flow_clamped)

            logits = self.model(x_adv, mode="classify")
            loss = self.criterion(logits, y)

            (-loss).backward()  # maximize loss
            optimizer.step()

        flow_clamped = torch.clamp(flow, -self.eps, self.eps)
        x_adv = self._flow_warp(x, flow_clamped).detach()

        return x_adv
# =========================================================
# ---------------- BPDA WRAPPER ----------------------------
# =========================================================

class BPDAWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# =========================================================
# ---------------- EVALUATION ------------------------------
# =========================================================

def evaluate(model, loader):
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def measure_latency(model, loader, n_batches=20):
    model.eval()
    start = time.time()

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(DEVICE)
            _ = model(x)

    end = time.time()
    return (end - start) / n_batches


# =========================================================
# ---------------- ATTACKS --------------------------------
# =========================================================

def get_attack(name, model, eps, num_classes=10):

    if name == "pgd":
        return torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=1)

    elif name == "eot_pgd":
        return torchattacks.EOTPGD(
            model,
            eps=eps,
            alpha=eps/4,
            steps=10,
            eot_iter=20
        )

    elif name == "autoattack":
        adversary = AutoAttack(model, norm='Linf', eps=eps, version='rand')
        # AutoAttack's targeted sub-attacks (APGD-T, FAB-T) default to 9
        # target classes, which only makes sense for >=10-class datasets
        # (CIFAR-10 etc). DermaMNIST has 7 classes -> at most 6 "other"
        # classes exist to target. Without this, APGD-T crashes with
        # "IndexError: index -8 is out of bounds" once it tries the 8th/9th
        # target class that doesn't exist.
        n_target = max(1, min(9, num_classes - 1))
        if hasattr(adversary, 'apgd_targeted'):
            adversary.apgd_targeted.n_target_classes = n_target
        if hasattr(adversary, 'fab'):
            adversary.fab.n_target_classes = n_target
        return adversary
    elif name == "bpda_eot":
        return BPDA_EOT_Attack(model, adv_eps=8/255, eot_attack_reps=10)
    elif name == "stadv":
        return STAdvAttack(model, eps=0.05, steps=20, lr=0.01)    
    else:
        raise ValueError(f"Unknown attack {name}")


# =========================================================
# ---------------- MAIN -----------------------------------
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    # Dataset / model
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--classifier", type=str, default="resnet18")

    # Checkpoints
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--score_ckpt", type=str, required=True)
    parser.add_argument("--score_config", type=str,
                        default="configs.score_configs.ve.cifar10_ncsnpp_continuous")

    # ODE params
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--ode_T", type=float, default=0.000000015)
    parser.add_argument("--t_noise", type=float, default=0.0000001)

    # Attack
    parser.add_argument("--attack", type=str, default="autoattack")
    parser.add_argument("--eps", type=float, default=8/255)

    # Eval
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()

    print("Device:", DEVICE)

    # -------------------------
    # Dataset
    # -------------------------

    # -------------------------
    # Score model (moved up so we know its image_size before loading data)
    # -------------------------
    score_model, cfg = load_score_model(
        args.score_ckpt,
        args.score_config,
        DEVICE
    )
    score_image_size = cfg.data.image_size

    # -------------------------
    # Dataset -- loaded at the SCORE MODEL's resolution, not the dataset's
    # native size. MedMNIST's native "big" render is 224, but NCSN++ was
    # built for cfg.data.image_size (e.g. 64) -- feeding it 224px images
    # desyncs its internal module indexing (AttnBlockpp TypeError).
    # -------------------------
    dataset, normalize, denormalize = get_dataset(args.dataset, train=False, image_size=score_image_size)
    normalize = normalize.to(DEVICE)
    denormalize = denormalize.to(DEVICE)
    if args.n_samples is not None and args.n_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.n_samples, replace=False)
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------
    # Classifier
    # -------------------------
    base_model = get_classifier(args.classifier).to(DEVICE)
    classifier = base_model
    classifier = adapt_model(base_model, args.dataset)
    classifier = load_classifier(classifier, args.classifier_ckpt)



    # -------------------------
    # ODE purifier
    # -------------------------
    vf = ScoreVectorField(score_model, args.t_noise).to(DEVICE)
    ode = NeuralODE(vf, args.ode_T).to(DEVICE)

        # 224 = resolution native/d'entrainement du classifieur (train_cifar_classifier.py).
    # Si tu entraines un jour le classifieur a une autre taille, remplace 224 ici.
        # Native training resolution of the classifier: 32x32 for cifar10/mnist
    # (train_cifar_classifier.py's CIFAR-style stem), 224x224 for the
    # MedMNIST datasets. Resizing to the WRONG one breaks classifiers with
    # a fixed-size (non-adaptive) final pooling layer -- e.g. a CIFAR
    # WideResNet fed 224px input won't collapse to a single spatial
    # location, and argmax(1) then returns extra spatial dims instead of
    # per-sample class predictions.
    if args.dataset.lower() in ('cifar10', 'mnist'):
        classifier_native_size = 32
    else:
        classifier_native_size = 224

    model = PurifiedClassifier(denormalize, ode, normalize, classifier, args.sigma,
                                classifier_input_size=classifier_native_size).to(DEVICE)
    model.eval()

    # -------------------------
    # Clean accuracy
    # -------------------------
    clean_acc = evaluate(model, loader)
    print(f"\nClean accuracy: {clean_acc:.4f}")

    # -------------------------
    # Attack
    # -------------------------
    # Number of classes for the current dataset -- needed by AutoAttack's
    # targeted sub-attacks (see get_attack() patch above).
    if args.dataset.lower() in ('cifar10', 'mnist'):
        num_classes = 10
    else:
        from medmnist import INFO
        num_classes = len(INFO[args.dataset.lower()]['label'])

    attack = get_attack(args.attack, model, args.eps, num_classes=num_classes)

    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if args.attack == "bpda_eot":
          x_adv = attack.attack(x, y)
        elif args.attack == "autoattack":
            x_adv = attack.run_standard_evaluation(x, y, bs=x.size(0))
        elif args.attack == "stadv":
            x_adv = attack.attack(x, y)    
        else:
            x_adv = attack(x, y)
        with torch.no_grad():
          logits = model(x_adv, mode="classify")
          pred = logits.argmax(1)

          correct += (pred == y).sum().item()
          total += y.size(0)

    robust_acc = correct / total
    print(f"Robust accuracy ({args.attack}): {robust_acc:.4f}")

    # -------------------------
    # Latency
    # -------------------------
    latency = measure_latency(model, loader)
    print(f"Latency (sec/batch): {latency:.4f}")


if __name__ == "__main__":
    main()