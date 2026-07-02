# model/Step_A.py
from __future__ import annotations
import math
import os
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Utils -------------------- #
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -------------------- LSKNet Components -------------------- #
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=stride,
                              padding=(patch[0] // 2, patch[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


# -------------------- FiLM Gate -------------------- #
class SpatialFiLMGate(nn.Module):
    def __init__(self, c4: int, c2: int, c3: int, r: int = 4, lam: float = 0.1):
        super().__init__()
        self.lam = float(lam)
        h2 = max(1, c2 // int(r))
        h3 = max(1, c3 // int(r))

        self.conv2 = nn.Sequential(
            nn.Conv2d(c4, h2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h2, 2 * c2, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c4, h3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h3, 2 * c3, 1)
        )
        # Identity-safe init
        for m in [self.conv2[-1], self.conv3[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x4, x2, x3):
        params2 = self.conv2(x4)
        params3 = self.conv3(x4)

        gamma2, beta2 = params2.chunk(2, dim=1)
        gamma3, beta3 = params3.chunk(2, dim=1)

        # 动态上采样对齐
        gamma2 = F.interpolate(gamma2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        beta2 = F.interpolate(beta2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        gamma3 = F.interpolate(gamma3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        beta3 = F.interpolate(beta3, size=x3.shape[2:], mode='bilinear', align_corners=False)

        y2 = x2 * (1 + self.lam * torch.tanh(gamma2)) + beta2
        y3 = x3 * (1 + self.lam * torch.tanh(gamma3)) + beta3
        return y2, y3


# -------------------- Main Backbone -------------------- #
class LSKNet_Backbone(nn.Module):
    def __init__(self,
                 in_chans=3,
                 # [修改] 默认使用 LSKNet-S 配置
                 embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 # [修改] 默认使用 LSKNet-S 深度
                 depths=[2, 2, 4, 2],
                 num_stages=4,
                 film_r: int = 4,
                 film_lambda: float = 0.1,
                 # [新增] 预训练权重路径
                 pretrained: str = "/mnt/volume3/home/jp/DLSNet/lsk_s_backbone-e9d2e551.pth"):
        super().__init__()

        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i]
            )
            block = nn.ModuleList([
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # FiLM 门控 (不在预训练权重中)
        self.film_gate = SpatialFiLMGate(c4=embed_dims[3], c2=embed_dims[1], c3=embed_dims[2], r=film_r,
                                         lam=film_lambda)

        # 初始化后立即尝试加载权重
        self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained_path):
        if not os.path.exists(pretrained_path):
            print(f" Warning: Pretrained weight not found at {pretrained_path}")
            print("   Training from scratch (random init)!")
            return

        print(f"Loading LSKNet-S weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # 处理 checkpoint 格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # [关键] 权重 Key 映射逻辑
        # 官方权重通常是: backbone.stages.0.embed.proj...
        # 我们的模型是: patch_embed1.proj...
        new_state_dict = {}
        for k, v in state_dict.items():
            # 1. 去掉 backbone. 前缀
            if k.startswith('backbone.'):
                k = k.replace('backbone.', '')

            # 2. 映射 stages.i -> patch_embed/block/norm
            # 官方: stages.0.embed -> 我们的: patch_embed1
            # 官方: stages.0.blocks.0 -> 我们的: block1.0
            # 官方: stages.0.norm -> 我们的: norm1

            if k.startswith('stages.'):
                parts = k.split('.')
                stage_idx = int(parts[1])  # 0, 1, 2, 3
                module_idx = int(stage_idx) + 1  # 1, 2, 3, 4

                module_type = parts[2]  # embed, blocks, norm

                if module_type == 'embed':
                    # stages.0.embed.proj.weight -> patch_embed1.proj.weight
                    new_k = f"patch_embed{module_idx}." + ".".join(parts[3:])
                elif module_type == 'blocks':
                    # stages.0.blocks.0.norm1.weight -> block1.0.norm1.weight
                    new_k = f"block{module_idx}." + ".".join(parts[3:])
                elif module_type == 'norm':
                    # stages.0.norm.weight -> norm1.weight
                    new_k = f"norm{module_idx}." + ".".join(parts[3:])
                else:
                    new_k = k  # Fallback
            else:
                new_k = k

            new_state_dict[new_k] = v

        # 加载权重 (strict=False 忽略我们自己加的 film_gate)
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Weights Loaded. Missing keys (should be FiLM only): {len(msg.missing_keys)}")
        # print(f"   Unexpected keys: {msg.unexpected_keys}")

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        x1, x2, x3, x4 = outs
        x2_mod, x3_mod = self.film_gate(x4, x2, x3)

        return {
            "x1": x1,
            "x2": x2_mod,
            "x3": x3_mod,
            "x4": x4
        }


if __name__ == "__main__":
    # Test LSKNet-S Config
    model = LSKNet_Backbone(embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2])
    x = torch.randn(1, 3, 1024, 1024)
    res = model(x)

    print("x1 shape", res['x1'].shape)
    print("x3 shape should be 320:", res['x3'].shape)
    print("x2 shape should be 1024:", res['x2'].shape)
    print("x4 shape should be 1024:", res['x4'].shape)