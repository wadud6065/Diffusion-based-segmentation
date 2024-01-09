import math
import torch
import torch.nn as nn
from torchsummary import summary
from .module import InvertedResidual, MobileVitBlock

model_cfg = {
    "s": {
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 8,
        "layers": [2, 3, 4]
    },
}


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0,
                                             end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class STEM(nn.Module):
    def __init__(self, input_channels, middle_channel, output_channels, expand_ratio, emb_channels):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channel = middle_channel
        self.output_channels = output_channels
        self.expand_ratio = expand_ratio
        self.emb_channels = emb_channels

        self.conv_layer = nn.Conv2d(
            in_channels=input_channels, out_channels=middle_channel, kernel_size=3, stride=2, padding=1)
        self.Inverted = InvertedResidual(
            in_channels=middle_channel, out_channels=output_channels, emb_channels=emb_channels, stride=1, expand_ratio=expand_ratio)

    def forward(self, x, timeEmbed):
        x = self.conv_layer(x)
        x = self.Inverted(x, timeEmbed)
        return x


class STAGE1(nn.Module):
    def __init__(self, input_channels, middle_channel, output_channels, expand_ratio, emb_channels):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channel = middle_channel
        self.output_channels = output_channels
        self.expand_ratio = expand_ratio
        self.emb_channels = emb_channels

        self.Inverted1 = InvertedResidual(
            in_channels=input_channels, out_channels=middle_channel, emb_channels=emb_channels, stride=2, expand_ratio=expand_ratio)
        self.Inverted2 = InvertedResidual(
            in_channels=middle_channel, out_channels=middle_channel, emb_channels=emb_channels, stride=1, expand_ratio=expand_ratio)
        self.Inverted3 = InvertedResidual(
            in_channels=middle_channel, out_channels=output_channels, emb_channels=emb_channels, stride=1, expand_ratio=expand_ratio)

    def forward(self, x, timeEmbed):
        x = self.Inverted1(x, timeEmbed)
        x = self.Inverted2(x, timeEmbed)
        x = self.Inverted3(x, timeEmbed)
        return x


class STAGE2(nn.Module):
    def __init__(self, input_channels, middle_channel, output_channels, expand_ratio, emb_channels, d_model, layers):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channel = middle_channel
        self.output_channels = output_channels
        self.expand_ratio = expand_ratio
        self.emb_channels = emb_channels
        self.d_model = d_model
        self.layers = layers

        self.Inverted1 = InvertedResidual(
            in_channels=input_channels, out_channels=middle_channel, emb_channels=emb_channels, stride=2, expand_ratio=expand_ratio)

        self.MobileVitBlock1 = MobileVitBlock(in_channels=middle_channel, out_channels=output_channels, d_model=d_model,
                                              layers=layers, mlp_dim=d_model*2)

    def forward(self, x, timeEmbed):
        x = self.Inverted1(x, timeEmbed)
        x = self.MobileVitBlock1(x)
        return x


class STAGE3(nn.Module):
    def __init__(self, input_channels, middle_channel, output_channels, expand_ratio, emb_channels, d_model, layers):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channel = middle_channel
        self.output_channels = output_channels
        self.expand_ratio = expand_ratio
        self.emb_channels = emb_channels
        self.d_model = d_model
        self.layers = layers

        self.Inverted1 = InvertedResidual(
            in_channels=input_channels, out_channels=middle_channel, emb_channels=emb_channels, stride=2, expand_ratio=expand_ratio)

        self.MobileVitBlock1 = MobileVitBlock(in_channels=middle_channel, out_channels=output_channels, d_model=d_model,
                                              layers=layers, mlp_dim=d_model*4)

    def forward(self, x, timeEmbed):
        x = self.Inverted1(x, timeEmbed)
        x = self.MobileVitBlock1(x)
        return x


class STAGE4(nn.Module):
    def __init__(self, input_channels, middle_channel1, middle_channel2, output_channels, expand_ratio, emb_channels, d_model, layers):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channel1 = middle_channel1
        self.middle_channel2 = middle_channel2
        self.output_channels = output_channels
        self.expand_ratio = expand_ratio
        self.emb_channels = emb_channels
        self.d_model = d_model
        self.layers = layers

        self.Inverted1 = InvertedResidual(
            in_channels=input_channels, out_channels=middle_channel1, emb_channels=emb_channels, stride=2, expand_ratio=expand_ratio)

        self.MobileVitBlock1 = MobileVitBlock(in_channels=middle_channel1, out_channels=middle_channel2, d_model=d_model,
                                              layers=layers, mlp_dim=d_model*4)

        self.conv = nn.Conv2d(
            in_channels=middle_channel2, out_channels=output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, timeEmbed):
        x = self.Inverted1(x, timeEmbed)
        x = self.MobileVitBlock1(x)
        x = self.conv(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, img_size, input_channels, features_list, d_list, transformer_depth, expansion, output_channels=2):
        super(MobileViT, self).__init__()
        self.init_feature = features_list[0]
        self.emb_channels = self.init_feature*4

        self.stem = STEM(input_channels=input_channels,
                         middle_channel=features_list[0],
                         output_channels=features_list[1],
                         expand_ratio=expansion,
                         emb_channels=self.emb_channels)

        self.stage1 = STAGE1(input_channels=features_list[1],
                             middle_channel=features_list[2],
                             output_channels=features_list[3],
                             expand_ratio=expansion,
                             emb_channels=self.emb_channels)

        self.stage2 = STAGE2(input_channels=features_list[3],
                             middle_channel=features_list[4],
                             output_channels=features_list[5],
                             expand_ratio=expansion,
                             emb_channels=self.emb_channels,
                             d_model=d_list[0],
                             layers=transformer_depth[0],)

        self.stage3 = STAGE3(input_channels=features_list[5],
                             middle_channel=features_list[6],
                             output_channels=features_list[7],
                             expand_ratio=expansion,
                             emb_channels=self.emb_channels,
                             d_model=d_list[1],
                             layers=transformer_depth[1],)

        self.stage4 = STAGE4(input_channels=features_list[7],
                             middle_channel1=features_list[8],
                             middle_channel2=features_list[9],
                             output_channels=features_list[10],
                             expand_ratio=expansion,
                             emb_channels=self.emb_channels,
                             d_model=d_list[2],
                             layers=transformer_depth[2],)

        self.dconv1 = nn.ConvTranspose2d(640, 192, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(320, 128, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(224, 96, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(160, 48, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(80, output_channels, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(192)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.batch_norm4 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(True)

        self.time_embed = nn.Sequential(
            nn.Linear(features_list[0], self.emb_channels),
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels),
        )

    def forward(self, x, time):
        emb = self.time_embed(timestep_embedding(
            timesteps=time, dim=self.init_feature))
        # Stem
        x1 = self.stem(x, emb)  # [4, 32, 112, 112]

        # stage1
        x2 = self.stage1(x1, emb)  # [4, 64, 56, 56]

        # Stage2
        x3 = self.stage2(x2, emb)  # [4, 96, 28, 28]

        # Stage3
        x4 = self.stage3(x3, emb)  # [4, 128, 16, 16]
        x4 = nn.functional.interpolate(
            x4, size=(16, 16), mode='bilinear', align_corners=True)

        # Stage4
        x5 = self.stage4(x4, emb)  # [4, 640, 8, 8]

        # Decoder
        # 4, 384,  16, 16
        d1_ = self.batch_norm(self.dconv1(self.relu(x5)))
        d1 = torch.cat((d1_, x4), 1)

        # 2, 384, 28, 28
        d2_ = self.batch_norm1(self.dconv2(self.relu(d1)))
        d2_ = nn.functional.interpolate(d2_, size=(
            28, 28), mode='bilinear', align_corners=True)
        d2 = torch.cat((d2_, x3), 1)

        # 2, 192, 56, 56
        d3_ = self.batch_norm2(self.dconv3(self.relu(d2)))
        d3_ = nn.functional.interpolate(d3_, size=(
            56, 56), mode='bilinear', align_corners=True)
        # 2, 288, 56, 56
        d3 = torch.cat((d3_, x2), 1)

        # 2, 96, 112, 112
        d4_ = self.batch_norm3(self.dconv4(self.relu(d3)))
        d4_ = nn.functional.interpolate(d4_, size=(
            112, 112), mode='bilinear', align_corners=True)
        d4 = torch.cat((d4_, x1), 1)

        d5 = self.dconv5(self.relu(d4))

        return d5

# ==================================================================================================


def MobileViT_S(img_size=256, num_classes=1):
    cfg_s = model_cfg["s"]
    model_s = MobileViT(img_size, cfg_s["features"], cfg_s["d"],
                        cfg_s["layers"], cfg_s["expansion_ratio"], num_classes)
    return model_s
