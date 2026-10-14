# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """


import sys
from pathlib import Path

# MUST HAPPEN BEFORE 'from models...'
current_file = Path(__file__).resolve()
grandparent_dir = current_file.parent.parent.parent # Check: training/score_training/run_lib.py -> training/ -> root/
sys.path.append(str(grandparent_dir))

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
#import tensorflow_gan as tfgan
import logging

# FIX: import the SAME sde_lib module that models/score_models/utils.py and
# training/score_training/losses.py use (`from training.score_training import
# sde_lib`). The old `import sde_lib` here resolved to a DIFFERENT module
# object (whatever stray sde_lib.py sits on sys.path via grandparent_dir),
# so `isinstance(sde, sde_lib.VESDE)` inside get_score_fn() failed even
# though the SDE really was a VESDE -- same class name, different module.
from training.score_training import sde_lib

from models.score_models import ddpm, ncsnv2, ncsnpp
from training.score_training import losses
from models.score_models import utils as mutils
from models.score_models.ema import ExponentialMovingAverage
from my_datasets import datasets
from torch.utils.data import DataLoader
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS


def _build_sde(config):
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  return sde, sampling_eps


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # FIX: use config.data.dataset instead of a hardcoded 'mnist', and load
  # via a real DataLoader (get_dataset() returns a raw torch Dataset, not a
  # DataLoader -- iterating it directly, as before, yields one un-batched
  # sample at a time, which is why you were seeing
  # "batch shape: torch.Size([3, 32, 32])" with no batch dimension).
  # image_size is threaded through so MedMNIST's native 224 download gets
  # resized to whatever the score model's config expects (e.g. 64 or 256 --
  # NCSN++ needs a power of 2, 224 itself is not one).
  dataset_name = config.data.dataset
  image_size = config.data.image_size

  train_ds, _, _ = datasets.get_dataset(dataset_name, train=True, image_size=image_size)
  eval_ds, _, _ = datasets.get_dataset(dataset_name, train=False, image_size=image_size)

  train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
  eval_loader = DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=False,
                           num_workers=2, pin_memory=True, drop_last=True)

  train_iter = iter(train_loader)
  eval_iter = iter(eval_loader)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  sde, sampling_eps = _build_sde(config)

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)


  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    try:
      batch = next(train_iter)[0]
    except StopIteration:
      # Recreate the iterator when the DataLoader is exhausted
      train_iter = iter(train_loader)
      batch = next(train_iter)[0]

    batch = batch.to(config.device).float()
    batch = scaler(batch)
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      try:
        eval_batch = next(eval_iter)[0]
      except StopIteration:
        eval_iter = iter(eval_loader)
        eval_batch = next(eval_iter)[0]
      eval_batch = eval_batch.to(config.device).float()
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
      logging.info("Saved checkpoint_%d.pth" % (save_step,))


# NOTE: `evaluate()` (FID/IS/likelihood evaluation) is not restored here.
# In the file you sent me, the entire function body was accidentally left
# inside an unterminated triple-quoted string (a stray `'''` right after
# train()'s checkpoint-save line), so it was never actually executable --
# calling run_lib.evaluate(...) would have raised NameError, not run TF-GAN
# code. Given your compute budget, the FID/IS/likelihood pipeline (which
# still depends on TensorFlow + tensorflow-gan + an Inception model) is a
# separate, expensive piece of work -- worth doing later once you have a
# score model you're happy with from Asym/NAsym/qualitative checks, not
# before. Tell me when you're ready and I'll rebuild it properly rather
# than just un-indenting the dead string.
