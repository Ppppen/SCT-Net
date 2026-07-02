import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FeatureSmoothModule(nn.Module):
    """
    [核心组件] 抗锯齿与特征混合模块
    作用：消除 grid_sample 的插值噪声，并进行非线性特征混合。
    结构：标准的 Residual Block，比 1x1 Conv 强，比 Inverted Residual 稳。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()  
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class DirectionalSamplerStepC(nn.Module):
    """
    Step C: 向量化方向引导采样模块 (最终定稿版)
    专注：几何引导(Geometry Guided) + 信号去噪(Denoising)
    """

    def __init__(self,
                 in_channels: int = 128,  # Step B 输出特征维度
                 out_channels: int = 128,  # 输出维度
                 num_steps: int = 4,  # 单侧采样步数
                 base_step: float = 2.0,  # 基础步长 (1/4尺度下2px)
                 step_scales: List[float] = [0.5, 1.0, 1.5, 2.0]):
        super().__init__()

        self.base_step = base_step
        self.scales = step_scales
        self.num_branches = 1 + 2 * num_steps

        # 1. 并行分支特征校正 (Group Conv)
        # 必须保留：因为采样回来的特征发生了空间错位，需要重新校准
        self.branch_convs = nn.Sequential(
            nn.Conv2d(
                in_channels * self.num_branches,
                in_channels * self.num_branches,
                kernel_size=3, padding=1,
                groups=self.num_branches,
                bias=False
            ),
            nn.BatchNorm2d(in_channels * self.num_branches),
            nn.GELU()  # ReLU -> GELU
        )

        # 2. 分支注意力 (Local Context Attention)
        # 必须保留：简单的 3x3 卷积足以判断局部置信度
        self.attention_gen = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels // 2, 3, padding=1),
            nn.GELU(),  # ReLU -> GELU
            nn.Conv2d(in_channels // 2, self.num_branches, 1)
        )

        # 3. 融合投影 (Linear Projection)
        self.fusion_project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()  # ReLU -> GELU
        )

        # 4. 后处理去噪 (Non-linear Smoothing)
        # 必须保留：这是保证特征图纯净的关键
        self.post_refine = FeatureSmoothModule(out_channels)

    def _create_vectorized_grid(self, theta, width, H, W):
        """构建超级采样网格"""
        B = theta.shape[0]
        device = theta.device

        # 基础网格
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)

        eps = 1e-6
        scale_x = 2.0 / (W - 1 + eps)
        scale_y = 2.0 / (H - 1 + eps)
        
        # scale_x = 2.0 / (W - 1)
        # scale_y = 2.0 / (H - 1)

        # 偏移量计算 (鲁棒逻辑)
        # 0.5 系数用于抑制 Width 预测误差
        step_len_map = self.base_step * (1.0 + 0.5 * width)

        delta_x = step_len_map * theta[:, 0:1, :, :] * scale_x
        delta_y = step_len_map * theta[:, 1:2, :, :] * scale_y

        offsets_list = []
        # 原点
        offsets_list.append(torch.zeros(B, 1, H, W, 2, device=device))

        # 多尺度分支
        for s in self.scales:
            # 正方向
            d_pos = torch.stack([delta_x * s, delta_y * s], dim=-1)
            offsets_list.append(d_pos)
            # 负方向
            d_neg = torch.stack([-delta_x * s, -delta_y * s], dim=-1)
            offsets_list.append(d_neg)

        all_offsets = torch.cat(offsets_list, dim=1)
        return base_grid.unsqueeze(1) + all_offsets

    def forward(self, feature, theta, width):
        B, C, H, W = feature.shape

        # 1. 几何计算

        grid = self._create_vectorized_grid(theta, width, H, W)

        # 2. 向量化采样
        # 1. 把 K 个分身看作是 Batch 的一部分，或者独立的图片
        # feature: [B, C, H, W] -> [B, 1, C, H, W] -> [B, K, C, H, W] -> [B*K, C, H, W]
        x_expanded = feature.unsqueeze(1).expand(-1, self.num_branches, -1, -1, -1).reshape(B * self.num_branches, C, H,
                                                                                            W)

        # 2. 把坐标网格也拉平，跟 feature 对应上
        # grid: [B, K, H, W, 2] -> [B*K, H, W, 2]
        grid_flatten = grid.view(B * self.num_branches, H, W, 2)

        sampled_feats = F.grid_sample(x_expanded, grid_flatten, mode='bilinear', padding_mode='border',
                                      align_corners=False)

        # 3. 特征校正
        # 1. 变回来：把 K 和 C 拼在一起
        # [B*K, C, H, W] -> [B, K*C, H, W]
        sampled_feats = sampled_feats.view(B, self.num_branches * C, H, W)
        # 2. 卷积处理
        refined_feats = self.branch_convs(sampled_feats)
        # 3. 再拆开
        refined_feats = refined_feats.view(B, self.num_branches, C, H, W)

        # 4. 注意力加权
        att_input = torch.cat([feature, theta, width], dim=1)
        att_logits = self.attention_gen(att_input)
        att_weights = F.softmax(att_logits, dim=1).unsqueeze(2)

        fused_context = (refined_feats * att_weights).sum(dim=1)

        # 5. 投影与平滑
        out = self.fusion_project(fused_context)
        out = self.post_refine(out)

        return out


if __name__ == "__main__":
    # 模拟测试
    B, C, H, W = 2, 128, 256, 256
    feat = torch.randn(B, C, H, W)
    theta = torch.randn(B, 2, H, W)
    width = torch.abs(torch.randn(B, 1, H, W))

    step_c = DirectionalSamplerStepC(in_channels=128, out_channels=128)
    out = step_c(feat, theta, width)

    print("-" * 30)
    print(f"Step C Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in step_c.parameters()) / 1e6:.2f} M")
    print("-" * 30)