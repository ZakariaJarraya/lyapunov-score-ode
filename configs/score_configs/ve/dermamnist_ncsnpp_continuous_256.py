# Drop this into your repo at:
#   configs/score_configs/ve/dermamnist_ncsnpp_continuous_256.py
#
# Matches exactly the training run launched with:
#   python train_score_dermamnist.py --image_size 256 --nf 64 \
#       --ch_mult 1,1,2,2,2,2,2 --num_res_blocks 1 --attn_resolutions 16 \
#       --batch_size 8 --n_iters 15000 --out_dir .../lyap_derma_small
#
# Use it as --score_config configs.score_configs.ve.dermamnist_ncsnpp_continuous_256
# in evaluate.py / global_1_.py, together with
# --score_ckpt .../lyap_derma_small/score_dermamnist_final.pth

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 8
    training.n_iters = 15000

    config.data = data = ml_collections.ConfigDict()
    data.image_size = 256
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
    model.nf = 64
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
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
