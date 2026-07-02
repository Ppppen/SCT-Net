import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    [保留模块] 条形池化
    针对道路细长结构的拓扑增强。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2:]
        x1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return x * self.sigmoid(out)


class SelectiveFeatureFusion(nn.Module):
    """
    [核心升级] SK-Style 选择性特征融合

    Paper Story:
    "Addressing the spatial misalignment caused by geometric sampling in Step C.
    We introduce a Selective Fusion mechanism that dynamically recalibrates
    the contribution of local details (x1) and long-range context (F_ctx)
    per channel, based on global descriptor."

    解决痛点：Step C 采样回来的特征可能有噪点。这个模块让网络自动判断：
    "对于这个通道，我是该信原来的特征，还是信采样回来的特征？"
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.dim = out_channels

        # 1. 特征对齐
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()  # ReLU -> GELU
        )
        # 对 ctx 加一个 Depthwise 卷积进行"微调/去噪"
        self.conv_ctx = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),  # ReLU -> GELU
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),  # DW微调
            nn.BatchNorm2d(out_channels),
            nn.GELU()   # ReLU -> GELU
        )

        # 2. 全局描述子
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction),
            nn.GELU(),  # ReLU -> GELU
            nn.Linear(out_channels // reduction, out_channels * 2),  # 输出两个分支的权重
            nn.Softmax(dim=1)  # 使得两个分支权重之和为 1
        )

    def forward(self, x1, f_ctx):
        # 对齐
        feat_x1 = self.conv_x1(x1)
        feat_ctx = self.conv_ctx(f_ctx)

        # 融合基底：两者相加
        feat_sum = feat_x1 + feat_ctx

        # 计算选择权重
        b, c, _, _ = feat_sum.size()
        s = self.avg_pool(feat_sum).view(b, c)
        attn = self.fc(s).view(b, 2, c, 1, 1)  # [B, 2, C, 1, 1]

        # 动态选择：w1 * x1 + w2 * ctx
        # 这里的 attn[:, 0] 是局部特征的权重，attn[:, 1] 是上下文特征的权重
        out = feat_x1 * attn[:, 0] + feat_ctx * attn[:, 1]

        return out


class EdgeStream(nn.Module):
    """
    [辅助流] 边缘强化流
    专门提取高频信息，用于最后修整上采样的模糊边界。
    """

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),  # ReLU -> GELU
            # Laplacian 算子风格的卷积 (差分)
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()  # 输出边缘 Attention Map
        )

    def forward(self, x):
        return self.conv(x)


class StepD_UltimateDecoder(nn.Module):
    """
    Step D: 最终完整版
    集成了 StripPooling (拓扑), SK-Fusion (融合), EdgeStream (边缘)
    """

    def __init__(self,
                 in_channels_ctx: int = 128,
                 in_channels_x1: int = 64,
                 hidden_dim: int = 128,
                 num_classes: int = 1):
        super().__init__()

        # 1. 智能融合 (解决 Step C 采样噪声问题)
        self.sk_fusion = SelectiveFeatureFusion(
            in_channels=in_channels_ctx,  # 假设 x1 和 ctx 通道在内部对齐
            out_channels=hidden_dim
        )
        # 注意：这里需要重新映射 x1 通道
        self.x1_proj = nn.Conv2d(in_channels_x1, in_channels_ctx, 1)

        # 2. 条形池化 (增强连通性 - Dice友好)
        self.strip_pool = StripPooling(hidden_dim)

        # 3. 边缘流 (增强边界 - BCE友好)
        # 利用 x1 (细节最丰富) 来提取边缘
        self.edge_stream = EdgeStream(in_channels_x1, hidden_dim)

        # 4. 级联上采样
        # 1/4 -> 1/2
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # ReLU -> GELU
        )
        # 1/2 -> 1/1
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # ReLU -> GELU
        )

        # 5. 输出头
        self.head = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x1, f_ctx):
        # x1: [B, 64, H, W]
        # f_ctx: [B, 128, H, W]

        # --- Stage 1: 选择性融合 ---
        # 先把 x1 映射到和 f_ctx 一样的维度
        x1_mapped = self.x1_proj(x1)
        # SK-Fusion: 自动去噪，自动选择信赖 x1 还是 f_ctx
        feat = self.sk_fusion(x1_mapped, f_ctx)  # [B, 128, H, W]

        # --- Stage 2: 拓扑增强 ---
        feat = self.strip_pool(feat)

        # --- Stage 3: 边缘注入 ---
        # 计算边缘权重图
        edge_map = self.edge_stream(x1)  # 基于原始 x1 计算
        # 显式强化边缘区域的特征
        feat = feat + feat * edge_map

        # --- Stage 4: 上采样 ---
        feat = self.up1(feat)
        feat = self.up2(feat)

        # --- Stage 5: 输出 ---
        logits = self.head(feat)

        return logits


if __name__ == "__main__":
    B = 2
    x1 = torch.randn(B, 64, 256, 256)
    ctx = torch.randn(B, 128, 256, 256)

    model = StepD_UltimateDecoder()
    out = model(x1, ctx)

    print("-" * 30)
    print(f"Decoder Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print("-" * 30)