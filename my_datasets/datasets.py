from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import tensorflow as tf
from torchvision import datasets, transforms
import medmnist
from medmnist import INFO
import os
import torchvision.transforms as T


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    INFO = None


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class _SqueezeLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset whose per-sample label comes back as a (1,)-shaped
    array/tensor (MedMNIST's convention) so it presents as a plain scalar
    instead -- avoids CrossEntropyLoss's "multi-target not supported" and
    silently-wrong (pred == y) broadcasting once batched."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = torch.as_tensor(y).reshape(-1)[0].long()
        return x, y


class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))

    def forward(self, x):
        return x * self.std + self.mean


def get_dataset(name='cifar10', data_dir='./data', train=True, image_size=None):
    """
    image_size: if given, a final torchvision Resize is applied on top of
    whatever native resolution the underlying dataset ships at. For MedMNIST
    datasets this lets you keep downloading the "big" native size=224 render
    (as intended) while still handing the score model a power-of-2 input
    (e.g. image_size=64 for a smoke test, 256 for a bigger run) -- NCSN++'s
    FIR up/downsampling requires a power of 2, and 224 is not one.
    If image_size is None, no resize is applied (old behavior).
    """
    name = name.lower()
    split = 'train' if train else 'test'

    def build_transforms(native_size, mean, std, target_size=None):
        tfms = [transforms.ToTensor()]
        if target_size is not None and target_size != native_size:
            tfms.append(transforms.Resize(target_size))
        return transforms.Compose(tfms)

    # ---------------- CIFAR10 ----------------
    if name == 'cifar10':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        tfm = build_transforms(32, mean, std, target_size=image_size)
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)

    # ---------------- MNIST ----------------
    elif name == 'mnist':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        resize_target = image_size if image_size is not None else 32
        tfm = transforms.Compose([
            transforms.Resize(resize_target),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm)

    # ---------------- MedMNIST ----------------
    elif name in ['pneumoniamnist', 'dermamnist']:
        if INFO is None:
            raise ImportError("medmnist not installed")

        info = INFO[name]
        DataClass = getattr(medmnist, info['python_class'])

        n_channels = info['n_channels']
        mean = tuple([0.5] * n_channels)
        std = tuple([0.5] * n_channels)

        # Always download the native "big" 224 render (intentional), and
        # resize down/up to whatever the score model needs on top of that.
        tfm = build_transforms(224, mean, std, target_size=image_size)

        dataset = DataClass(root=data_dir, split=split, transform=tfm, download=True, size=224)
        if n_channels == 1:
            # score model / classifier both expect 3 channels
            base_tfm = dataset.transform
            dataset.transform = transforms.Compose([base_tfm, transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

        # MedMNIST labels come back shaped (1,) per sample instead of a
        # scalar -- batched by DataLoader that becomes (B,1), which breaks
        # nn.CrossEntropyLoss ("multi-target not supported") and silently
        # corrupts accuracy counts (pred==y broadcasts (B,) against (B,1)
        # into a (B,B) matrix, e.g. the "Clean accuracy: 8.74" you just saw).
        # train_cifar_classifier.py works around this itself with
        # yb.squeeze().long(); evaluate.py does not, so we fix it once here.
        dataset = _SqueezeLabelDataset(dataset)

    # ---------------- ImageNet ----------------
    elif name == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        resize_target = image_size if image_size is not None else 256
        tfm = transforms.Compose([
            transforms.Resize(resize_target),
            transforms.CenterCrop(resize_target if image_size is not None else 224),
            transforms.ToTensor(),
        ])
        path = os.path.join(data_dir, 'train' if train else 'val')
        dataset = datasets.ImageFolder(root=path, transform=tfm)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    normalize = Normalize(mean, std)
    denormalize = Denormalize(mean, std)

    return dataset, normalize, denormalize


def get_dataloaders(dataset_name, batch_size, num_workers=2, root="./my_datasets", image_size=None):
    train_ds, normalize, denormalize = get_dataset(dataset_name, root, train=True, image_size=image_size)
    test_ds, _, _ = get_dataset(dataset_name, root, train=False, image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return train_loader, test_loader, normalize, denormalize

