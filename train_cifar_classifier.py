#!/usr/bin/env python3
# train_cifar_classifier.py
# Usage examples:
#  Fast (train only final layer):
#    python train_cifar_classifier.py --mode head --epochs 30 --batch-size 256 --out ckpt_head.pth
#  Full fine-tune:
#    python train_cifar_classifier.py --mode full --epochs 100 --batch-size 256 --out ckpt_full.pth

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import time
import os
from medmnist import INFO

def build_model(device, dataset):
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
        # ImageNet-style stem (for 256x256 medical images)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from medmnist import DermaMNIST, PneumoniaMNIST
from torchvision.datasets import ImageFolder

def get_dataloaders(
    dataset,
    batch_size,
    num_workers=4,
    root="./data"
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
        train_transform = T.Compose([T.ToTensor(),to_minus_one_one])
        test_transform  = T.Compose([T.ToTensor(),to_minus_one_one])

        train_ds = DermaMNIST(
            root=root, split="train", size=224, download=True, transform=train_transform
        )
        test_ds = DermaMNIST(
            root=root, split="test", size=224, download=True, transform=test_transform
        )

    elif dataset == "pneumoniamnist":
        train_transform = T.Compose([T.Grayscale(num_output_channels=3),T.ToTensor(),to_minus_one_one])
        test_transform  = T.Compose([T.Grayscale(num_output_channels=3),T.ToTensor(),to_minus_one_one])

        train_ds = PneumoniaMNIST(
            root=root,
            split="train",
            size = 224,
            download=True,
            transform=train_transform
        )

        test_ds = PneumoniaMNIST(
            root=root,
            split="test",
            size = 224,
            download=True,
            transform=test_transform
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

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        yb = yb.squeeze().long()
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return correct / total

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = build_model(device, args.dataset)
    if Path(args.out).is_file():
      checkpoint = torch.load(args.out)
      model.load_state_dict(checkpoint['model'])

    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, args.num_workers)

    if args.mode == "head":
        # freeze everything except last linear layer
        for name, p in model.named_parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        # optimizer: Adam on head
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.mode == "full":
        # fine-tune entire network
        for p in model.parameters():
            p.requires_grad = True
        # standard CIFAR setting: SGD + momentum
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # cosine or step LR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError("Unknown mode")

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            yb = yb.squeeze().long()
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        if args.mode == "full":
            scheduler.step()
        else:
            scheduler.step()

        train_loss = epoch_loss / n
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        # save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model': model.state_dict(), 'acc': best_acc, 'epoch': epoch}, args.out)
            print("Saved checkpoint:", args.out)

    elapsed = time.time() - start_time
    print("Training finished. Best acc:", best_acc, "Elapsed (s):", int(elapsed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head", "full"], default="full",
                        help="head: train only final layer (fast). full: fine-tune entire network")
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="cifar10_resnet18_ckpt.pth")
    args = parser.parse_args()
    train(args)

