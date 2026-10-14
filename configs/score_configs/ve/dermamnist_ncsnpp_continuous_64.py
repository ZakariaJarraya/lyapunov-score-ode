# Drop this into your repo at:
#   configs/score_configs/ve/dermamnist_ncsnpp_continuous_64.py
#
# Matches exactly the training run launched with:
#   python train_score_dermamnist.py --image_size 64 --nf 32 \
#       --ch_mult 1,2,2,2 --num_res_blocks 1 --attn_resolutions 16 \
#       --batch_size 32 --n_iters 3000 --out_dir .../lyap_derma_smoke
#
# Use it as --score_config configs.score_configs.ve.dermamnist_ncsnpp_continuous_64
# in evaluate.py / global_1_.py, together with
# --score_ckpt .../lyap_derma_smoke/score_dermamnist_final.pth

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 32
    training.n_iters = 50000
    training.likelihood_weighting = False
    training.log_freq = 50
    training.eval_freq = 100
    training.snapshot_freq = 500
    training.snapshot_freq_for_preemption = 200
    training.snapshot_sampling = False

    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 500
    optim.grad_clip = 1.0

    config.data = data = ml_collections.ConfigDict()
    data.image_size = 64
    data.num_channels = 3
    data.centered = True
    data.dataset = 'DermaMNIST'

    config.model = model = ml_collections.ConfigDict()
    model.name = 'ncsnpp'
    model.sigma_min = 0.01
    model.sigma_max = 50.0
    model.num_scales = 232
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 32
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 1
    model.attn_resolutions = (16,)
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

    config.seed = 0
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return config