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
        features_list = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        expansion = 8
        d_list = [144, 192, 240]
        transformer_depth = [2, 3, 4]

        mobilleVIT_model = MobileViT(img_size=224,
                                     input_channels=3,
                                     features_list=features_list,
                                     d_list=d_list,
                                     transformer_depth=transformer_depth,
                                     expansion=expansion,
                                     output_channels=2)

        input = th.randn((1, 3, 512, 512))

        t = th.tensor([500] * 1, device='cpu')

        temp_output = mobilleVIT_model(input, t)
        print(temp_output.shape)

        total_params = sum(param.numel()
                           for param in mobilleVIT_model.parameters())
        print("Total parameter ", total_params)

        # time_embed_dim = features_list[0] * 4
        # time_embed = nn.Sequential(
        #     linear(features_list[0], time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )
        # emb = time_embed(timestep_embedding(t, features_list[0]))

        # stem = STEM(input_channels=5,
        #             middle_channel=features_list[0], output_channels=features_list[1], expand_ratio=expansion, emb_channels=time_embed_dim)

        # stem = STAGE1(input_channels=features_list[1],
        #               middle_channel=features_list[2], output_channels=features_list[3], expand_ratio=expansion, emb_channels=time_embed_dim)

        # stem = STAGE2(input_channels=features_list[3],
        #               middle_channel=features_list[4],
        #               output_channels=features_list[5],
        #               expand_ratio=expansion,
        #               emb_channels=time_embed_dim,
        #               d_model=d_list[0],
        #               layers=transformer_depth[0],)

        # stem = STAGE3(input_channels=features_list[5],
        #                      middle_channel=features_list[6],
        #                      output_channels=features_list[7],
        #                      expand_ratio=expansion,
        #                      emb_channels=time_embed_dim,
        #                      d_model=d_list[1],
        #                      layers=transformer_depth[1],)

        # stem = STAGE4(input_channels=features_list[7],
        #               middle_channel1=features_list[8],
        #               middle_channel2=features_list[9],
        #               output_channels=features_list[10],
        #               expand_ratio=expansion,
        #               emb_channels=time_embed_dim,
        #               d_model=d_list[2],
        #               layers=transformer_depth[2],)
        # temp_output = stem(input, emb)
        # print(temp_output.shape)

        t = 5
        assert t == 5
