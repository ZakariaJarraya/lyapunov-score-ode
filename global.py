#!/usr/bin/env python3
"""
global_ode_dopri5.py

Composed classifier: NeuralODE(score_model) followed by classifier.
Neural ODE uses torchdiffeq with DOPRI5 (dopri5). FGSM is implemented as a white-box
attack optionally targeting the composed model.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchdiffeq import odeint_adjoint as odeint
#from torchdiffeq import odeint_adjoint as odeint
import importlib
from medmnist import DermaMNIST, PneumoniaMNIST
from torchvision.datasets import MNIST
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# ----------------- Defaults -----------------
DEFAULT_SCORE_CKPT = "mnist/checkpoints-meta/checkpoint.pth"
DEFAULT_CLASSIFIER_CKPT = "mnist_resnet18_ckpt.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Helpers -----------------
def register_song_models():
    # ensure model modules are imported so registry is populated
    import models.ddpm  # noqa: F401
    import models.ncsnv2  # noqa: F401
    import models.ncsnpp  # noqa: F401

def strip_module_prefix(state_dict):
    # Remove 'module.' prefixes from DataParallel checkpoints
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_sd[new_k] = v
    return new_sd

def safe_load_state_dict(model, ckpt_dict, strict=True):
    """
    ckpt_dict can be {'model': state_dict} or a raw state_dict.
    Will attempt to load and automatically strip 'module.' prefixes.
    """
    if ckpt_dict is None:
        return False
    sd = ckpt_dict
    # If it's a dict with a 'model' key, extract it
    if isinstance(ckpt_dict, dict) and 'model' in ckpt_dict:
        sd = ckpt_dict['model']
    # If keys have module prefix, strip and try again
    try:
        model.load_state_dict(sd, strict=strict)
        return True
    except Exception as e:
        # try stripping module.
        try:
            sd2 = strip_module_prefix(sd)
            model.load_state_dict(sd2, strict=strict)
            return True
        except Exception as e2:
            # final fallback: try non-strict
            try:
                model.load_state_dict(sd2, strict=False)
                return True
            except Exception:
                print("Failed to load state dict (strict and non-strict) :", e2)
                return False

# ----------------- Model loaders -----------------
import os
from medmnist import INFO


def tensor_to_rgb01(x):
    """
    x: tensor in [-1, 1], shape [3,H,W]
    returns numpy RGB in [0,1], shape [H,W,3]
    """
    x = x#(x + 1.0) / 2.0
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).cpu().numpy()


def tensor_to_rgb01(x, eps=1e-6):
    """
    x: tensor, shape [3,H,W], arbitrary range
    returns numpy RGB in [0,1], shape [H,W,3]
    """
    x = x.detach()
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min + eps)
    x = x.clamp(0.0, 1.0)
    return x.permute(1, 2, 0).cpu().numpy()    



import matplotlib.colors as mcolors


def hsv_residual(img_adv, img_ode, v_scale=5.0):
    """
    Compute residual in HSV space using matplotlib (version-safe).

    Inputs:
      img_adv, img_ode: numpy RGB in [0,1], shape [H,W,3]

    Output:
      numpy RGB in [0,1]
    """

    # RGB → HSV
    adv_hsv = mcolors.rgb_to_hsv(img_adv)
    ode_hsv = mcolors.rgb_to_hsv(img_ode)

    # Difference in V channel
    v_res = np.abs(ode_hsv[..., 2] - adv_hsv[..., 2])
    v_res = np.clip(v_res * v_scale, 0.0, 1.0)

    # Reconstruct HSV residual image
    hsv_res = np.zeros_like(adv_hsv)
    hsv_res[..., 0] = adv_hsv[..., 0]  # hue
    hsv_res[..., 1] = adv_hsv[..., 1]  # saturation
    hsv_res[..., 2] = v_res            # amplified value difference

    # HSV → RGB
    rgb_res = mcolors.hsv_to_rgb(hsv_res)
    return rgb_res



def visualize_denoising(
    classifier,
    composed,
    test_loader,
    device,
    fgsm_eps=0.03,
    n_samples=6,
    v_scale=5.0
):
    """
    Rows:
      1) clean images
      2) adversarial images
      3) ODE-denoised adversarial images
      4) HSV-V residual (adv vs denoised)
    """

    classifier.eval()
    composed.eval()

    xb, yb = next(iter(test_loader))
    xb = xb.to(device)
    yb = yb.to(device)

    # Randomly select samples
    idx = torch.randperm(xb.size(0))[:n_samples]
    xb = xb[idx]
    yb = yb[idx]

    # Generate adversarial examples (attack plain classifier)
    x_adv = pgd_attack(
        classifier,
        xb,
        yb,
        eps=fgsm_eps,
        alpha=fgsm_eps / 4,
        n_steps=10,
        random_start=True
    )

    # Denoise via Neural ODE
    with torch.no_grad():
        x_ode = composed.neural_ode(x_adv)

    # Convert to numpy for plotting
    clean_imgs = [tensor_to_rgb01(x) for x in xb]
    adv_imgs   = [tensor_to_rgb01(x) for x in x_adv]
    ode_imgs   = [tensor_to_rgb01(x) for x in x_ode]
    res_imgs   = [
        hsv_residual(adv_imgs[i], ode_imgs[i], v_scale=v_scale)
        for i in range(n_samples)
    ]

    # ---- Plot ----
    fig, axes = plt.subplots(4, n_samples, figsize=(2.2*n_samples, 8))

    row_titles = [
        "Clean",
        "Adversarial",
        "ODE Denoised",
        "HSV-V Residual"
    ]

    for r, imgs in enumerate([clean_imgs, adv_imgs, ode_imgs, res_imgs]):
        for c in range(n_samples):
            axes[r, c].imshow(imgs[c])
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(row_titles[r], fontsize=12)

    plt.tight_layout()
    plt.savefig('dermadenoise.pdf')
    plt.show()






def build_classifier(dataset, device):
    # Number of classes
    if dataset in ['mnist', 'cifar10']:
        num_classes = 10
    else:
        info = INFO[dataset]
        num_classes = len(info['label'])

    model = resnet18(weights=None)

    if dataset in ['mnist', 'cifar10']:
        # CIFAR-style stem
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    else:
        # ImageNet-style stem (e.g. 224x224 / 256x256 medical images)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def load_classifier(dataset, ckpt_path=None, device="cpu"):
    model = build_classifier(dataset, device=device)
    if ckpt_path is None:
        print("No classifier checkpoint provided; classifier is randomly initialized.")
        return model
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ok = safe_load_state_dict(model, ckpt, strict=False)
    if not ok:
        raise RuntimeError("Failed to load classifier checkpoint correctly.")
    print("Loaded classifier from:", ckpt_path)
    model.eval()
    return model

def load_score_model(score_ckpt, config_module="configs.ve.cifar10_ncsnpp_continuous", device="cpu"):
    # Register models first
    register_song_models()
    import models.utils as mutils
    # load config module and config
    cfg_mod = importlib.import_module(config_module)
    cfg = cfg_mod.get_config()
    # set config.device properly as torch.device
    try:
        cfg.device = torch.device(device)
    except Exception:
        # if config.device typed differently, convert in-place
        cfg.device = torch.device(device)
    # create model
    score_model = mutils.create_model(cfg)
    score_model = score_model.to(device)
    score_model.eval()
    # load checkpoint
    if not os.path.exists(score_ckpt):
        raise FileNotFoundError(f"Score checkpoint not found: {score_ckpt}")
    ckpt = torch.load(score_ckpt, weights_only=False, map_location=device)
    ok = safe_load_state_dict(score_model, ckpt, strict=False)
    if not ok:
        raise RuntimeError("Failed to load score model checkpoint.")
    print("Loaded score model from:", score_ckpt)
    return score_model, cfg

# ----------------- Score -> VectorField wrapper and Neural ODE -----------------
class ScoreVectorField(nn.Module):
    """
    Wrap Song score model into a vector field for ODE. Handles domain conversions.
    - classifier works in [-1,1]
    - score model trained in [0,1] when config.data.centered=False (common)
    Conversion:
      x_in = (x + 1)/2   # [-1,1] -> [0,1]
      v_in = score_fn(x_in, t_noise)   # derivative in x_in units
      v = 0.5 * v_in                     # derivative in x units
    """
    def __init__(self, score_model, cfg=None, t_noise=0.2, continuous=True, device="cpu"):
        super().__init__()
        self.score_model = score_model
        self.cfg = cfg
        self.t_noise = float(t_noise)
        self.device = device
        # Try to get the repo's score function (preferred)
        try:
            import models.utils as mutils
            # pass cfg.training.continuous if present
            self.score_fn = mutils.get_score_fn(None, self.score_model, train=False, continuous=continuous)
            self.uses_repo_score_fn = True
        except Exception:
            self.score_fn = None
            self.uses_repo_score_fn = False

    def forward(self, t_ode, x):
        # x: [-1,1]
        x_in = x
        #x_in = x_in.clamp(0.0, 1.0)
        b = x_in.shape[0]
        t_vec = torch.ones(b, device=x.device) * float(self.t_noise)
        # Query score model (no grad here if used for inference; but for FGSM through ODE we need grad)
        # We DO NOT use torch.no_grad so gradients flow during FGSM on composed model.
        if self.uses_repo_score_fn:
            v_in = self.score_fn(x_in, t_vec)
        else:
            # fallback: try calling model directly
            try:
                v_in = self.score_model(x_in, t_vec)
            except Exception:
                v_in = self.score_model(x_in, t_vec)
        # convert derivative units
        v = v_in * 1. #0.5
        return v

class NeuralODE(nn.Module):
    def __init__(self, vf: nn.Module, T_ode=0.1, rtol=1e-3, atol=1e-3, method='dopri5'):
        super().__init__()
        self.vf = vf
        self.T = float(T_ode)
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, x0):
        # Integrate from 0 -> T
        t = torch.tensor([0.0, self.T], device=x0.device, dtype=x0.dtype)
        x_t = odeint(self.vf, x0, t, rtol=self.rtol, atol=self.atol, method=self.method)
        xT = x_t[-1] 
        return xT

class ComposedClassifier(nn.Module):
    def __init__(self, neural_ode: NeuralODE, classifier: nn.Module):
        super().__init__()
        self.neural_ode = neural_ode
        self.classifier = classifier

    def forward(self, x):
        x_trans = self.neural_ode(x)
        return self.classifier(x_trans)

# ----------------- Attacks -----------------
def fgsm_attack(model, x, y, eps=0.03):
    """FGSM attack on model (white-box). Expects model(x) in [-1,1] domain."""
    model.train()  # ensure grads flow
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y.squeeze().long())
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + eps * grad.sign()
    x_adv = x_adv.clamp(-1.0, 1.0)
    model.eval()
    return x_adv.detach()


def pgd_attack(
    model,
    x,
    y,
    eps=0.03,
    alpha=0.007,
    n_steps=10,
    random_start=True
):
    """PGD attack on model (white-box). Expects model(x) in [-1,1] domain."""
    model.train()  # ensure grads flow
    y = y.squeeze().long()
    # initialize adversarial example
    x_adv = x.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(-1.0, 1.0)

    for _ in range(n_steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad = x_adv.grad.detach()

        # gradient ascent step
        x_adv = x_adv + alpha * grad.sign()

        # project back to epsilon-ball around x
        x_adv = torch.max(
            torch.min(x_adv, x + eps),
            x - eps
        )

        # clamp to valid input range
        x_adv = x_adv.clamp(-1.0, 1.0)

        x_adv = x_adv.detach()

    model.eval()
    return x_adv


# ----------------- Evaluation -----------------
def evaluate(
    classifier,
    composed=None,
    test_loader=None,
    device="cpu",
    fgsm_eps=0.03,
    attack_target="plain",
):
    """
    attack_target:
        'plain'    -> attack classifier alone (baseline)
        'composed' -> attack composed model (white-box)

    Returns a dict with:
        - clean accuracy of classifier
        - clean accuracy of composed model
        - adversarial accuracy of classifier
        - adversarial accuracy of composed model
    """

    classifier.eval()
    if composed is not None:
        composed.eval()

    total = 0
    correct_cls = 0
    correct_comp = 0
    correct_adv_cls = 0
    correct_adv_comp = 0

    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.squeeze().long().to(device)

        batch_size = xb.size(0)
        total += batch_size

        # -------------------------
        # Clean accuracy (classifier)
        # -------------------------
        with torch.no_grad():
            logits = classifier(xb)
            preds = logits.argmax(dim=1)
            correct_cls += (preds == yb).sum().item()

        # -------------------------
        # Clean accuracy (composed)
        # -------------------------
        if composed is not None:
            with torch.no_grad():
                logits_c = composed(xb)
                preds_c = logits_c.argmax(dim=1)
                correct_comp += (preds_c == yb).sum().item()
        else:
            correct_comp += (preds == yb).sum().item()

        # -------------------------
        # Generate adversarial examples
        # -------------------------
        if attack_target == "plain":
            x_adv = pgd_attack(
                classifier,
                xb,
                yb,
                eps=0.031,
                alpha=0.007,
                n_steps=10,
                random_start=True,
            )
        elif attack_target == "composed" and composed is not None:
            x_adv = pgd_attack(
                composed,
                xb,
                yb,
                eps=0.031,
                alpha=0.007,
                n_steps=10,
                random_start=True,
            )
        else:
            x_adv = pgd_attack(
                classifier,
                xb,
                yb,
                eps=0.031,
                alpha=0.007,
                n_steps=10,
                random_start=True,
            )

        # -------------------------
        # Adversarial accuracy (classifier)
        # -------------------------
        with torch.no_grad():
            logits_adv = classifier(x_adv)
            preds_adv = logits_adv.argmax(dim=1)
            correct_adv_cls += (preds_adv == yb).sum().item()

        # -------------------------
        # Adversarial accuracy (composed)
        # -------------------------
        if composed is not None:
            with torch.no_grad():
                logits_adv_c = composed(x_adv)
                preds_adv_c = logits_adv_c.argmax(dim=1)
                correct_adv_comp += (preds_adv_c == yb).sum().item()
        else:
            correct_adv_comp += (preds_adv == yb).sum().item()

        # limit evaluation for speed
        if total >= 500:
            break

    return {
        "total": total,
        "clean_cls_acc": correct_cls / total,
        "clean_comp_acc": correct_comp / total,
        "adv_cls_acc": correct_adv_cls / total,
        "adv_comp_acc": correct_adv_comp / total, 
      }



def get_dataloaders(
    dataset,
    batch_size,
    num_workers=4,
    root="./datasets"
):
    """
    Supported datasets:
      - 'cifar10'
      - 'mnist'
      - 'pneumonia'   (expects ImageFolder structure)
      - 'dermamnist'  (from MedMNIST)
    All datasets are mapped to [-1,1] and 3x32x32.
    """

    # ---- Common transforms ----
    def to_minus_one_one(x):
        return x #* 2.0 - 1.0

    rgb_32 = [
        T.Resize(32),
        T.ToTensor(),
        to_minus_one_one
    ]

    rgb_224 = [
        T.Resize(224),
        T.ToTensor(),
        to_minus_one_one
    ]

    rgb_32_aug = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        to_minus_one_one
    ]

    gray_to_rgb_32 = [
        T.Resize(32),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        to_minus_one_one
    ]

    gray_to_rgb_224 = [
        T.Resize(224),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        to_minus_one_one
    ]

    # ---- Dataset selection ----
    dataset = dataset.lower()

    if dataset == "cifar10":
        train_transform = T.Compose(rgb_32_aug)
        test_transform  = T.Compose(rgb_32)

        train_ds = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )

    elif dataset == "mnist":
        train_transform = T.Compose(gray_to_rgb_32)
        test_transform  = T.Compose(gray_to_rgb_32)

        train_ds = MNIST(
            root=root, train=True, download=True, transform=train_transform
        )
        test_ds = MNIST(
            root=root, train=False, download=True, transform=test_transform
        )

    elif dataset == "dermamnist":
        train_transform = T.Compose(rgb_224)
        test_transform  = T.Compose(rgb_224)

        train_ds = DermaMNIST(
            root=root, split="train", size=224, download=True, transform=train_transform
        )
        test_ds = DermaMNIST(
            root=root, split="test", size=224, download=True, transform=test_transform
        )

    elif dataset == "pneumoniamnist":
        # Expected structure:
        # data/pneumonia/train/NORMAL, PNEUMONIA
        # data/pneumonia/test/NORMAL, PNEUMONIA
        train_transform = T.Compose(gray_to_rgb_224)
        test_transform  = T.Compose(gray_to_rgb_224)

        train_ds = PneumoniaMNIST(
            root=root, split="train", size=224, download=True, transform=train_transform
        )
        test_ds = PneumoniaMNIST(
            root=root, split="test", size=224, download=True, transform=test_transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ---- Loaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader



# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--score_ckpt", type=str, default=DEFAULT_SCORE_CKPT)
    parser.add_argument("--score_config", type=str, default="configs.ve.cifar10_ncsnpp_continuous")
    parser.add_argument("--classifier_ckpt", type=str, default=DEFAULT_CLASSIFIER_CKPT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fgsm_eps", type=float, default=0.15)
    parser.add_argument("--t_noise", type=float, default=0.25)
    parser.add_argument("--ode_T", type=float, default=1.)
    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--attack_target", type=str, choices=["plain", "composed"], default="plain")
    args = parser.parse_args()

    device = DEVICE
    print("Using device:", device)

    # Data loader (matches classifier training -> [-1,1])
    test_loader = get_dataloaders(
    args.dataset,
    args.batch_size,
    num_workers=4,
    root="./datasets")
    # Load classifier
    classifier = load_classifier(args.dataset, args.classifier_ckpt, device=device)
    classifier.eval()

    # Load score model
    score_model = None
    cfg = None
    if args.score_ckpt and os.path.exists(args.score_ckpt):
        score_model, cfg = load_score_model(args.score_ckpt, config_module=args.score_config, device=device)
    else:
        print("Score checkpoint not found; running without ODE.")
    # Build composed classifier (if score model loaded)
    composed = None
    if score_model is not None:
        vf = ScoreVectorField(score_model, cfg, t_noise=args.t_noise, continuous=True, device=device).to(device)
        neural_ode = NeuralODE(vf, T_ode=args.ode_T, method=args.ode_method).to(device)
        composed = ComposedClassifier(neural_ode, classifier).to(device)
        print("Composed classifier ready (NeuralODE -> classifier).")
    # Evaluate
    print("Running evaluation. FGSM eps:", args.fgsm_eps, "attack_target:", args.attack_target)
    #results = evaluate(classifier, composed=composed, test_loader=test_loader, device=device)
    #print("Results:")
    #for k, v in results.items():
        #print(f"  {k}: {v}")
        # ---- Visualize denoising effect ----
    if composed is not None:
        print("Visualizing denoising effect on adversarial examples...")
        visualize_denoising(
            classifier=classifier,
            composed=composed,
            test_loader=test_loader,
            device=device,
            fgsm_eps=args.fgsm_eps,
            n_samples=4,
            v_scale=5
        )
    

if __name__ == "__main__":
    main()

 
    
