import pytest
import torch as th
from guided_diffusion.unet import UNetModel, ResBlock
from guided_diffusion import dist_util, logger
from torchinfo import summary
from guided_diffusion.mobileTrans import MobileViT, STEM, STAGE1, STAGE2, STAGE3, STAGE4
from guided_diffusion.module import *
import torch.nn as nn
from scripts.LIDCLoader import load_LIDC
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from guided_diffusion.elunet import ELUnet

model_cfg = {
    "s": {
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 8,
        "layers": [2, 3, 4]
    },
}


class Test_Unet:

    def test_unet(self):
        print('\n')
        # features_list = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        # expansion = 8
        # d_list = [144, 192, 240]
        # transformer_depth = [2, 3, 4]

        # mobilleVIT_model = MobileViT(img_size=224,
        #                              input_channels=3,
        #                              features_list=features_list,
        #                              d_list=d_list,
        #                              transformer_depth=transformer_depth,
        #                              expansion=expansion,
        #                              output_channels=2)

        eluNet = ELUnet(in_channels=3, out_channels=2, n=16)

        input = th.randn((1, 3, 256, 256))

        t = th.tensor([500] * 1, device='cpu')

        temp_output = eluNet(input, t)
        print(temp_output.shape)

        total_params = sum(param.numel()
                           for param in eluNet.parameters())
        print("Total parameter ", total_params)

        t = 5
        assert t == 5
