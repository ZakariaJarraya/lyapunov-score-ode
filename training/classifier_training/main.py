import sys
import os
from pathlib import Path

# Fix the path before importing run_lib
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

    
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision

import importlib
from training import train






# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  choices=['resnet18', 'resnet50', 'vlt','wideresnet'])
    parser.add_argument("--dataset", type=str, choices=['tiny_imagenet','cifar10','mnist','dermamnist','pneumoniamnist', 'imagenet'], default='cifar10')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epoch", type=int, default=100)
    args = parser.parse_args()


    assert isinstance(args.n_epoch, int)
    assert isinstance(args.batch_size, int)
    train(args)



if __name__ == "__main__":
    main()
