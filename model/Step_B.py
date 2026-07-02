import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiScaleFeatureProcessingModule(nn.Module):
    """
    模块1：多尺度特征处理模块
    负责对来自Step A的不同尺度特征进行降维和初步处理
    """

    def __init__(self, hidden_dim: int, x2_channels: int, x3_channels: int):
        super().__init__()
        # 处理 x2 (1/8 尺度)
        self.branch_x2 = nn.Sequential(
            nn.Conv2d(x2_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),  # ReLU -> GELU
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # ReLU -> GELU
        )
        # 处理 x3 (1/16 尺度)
        self.branch_x3 = nn.Sequential(
            nn.Conv2d(x3_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),  # ReLU -> GELU
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # ReLU -> GELU
        )

    def forward(self, x2, x3):
        return self.branch_x2(x2), self.branch_x3(x3)


class ContinuousDirectionAwareFusion(nn.Module):
    """
    模块2：连续方向感知融合模块
    Paper Story: "在特征融合阶段显式引入隐式方向场的一致性先验，
    解决不同尺度特征因感受野不同导致的方向错位问题。"
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 轻量级方向特征提取器
        self.direction_extractor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, 1)  # 临时预测方向 [cos, sin]
        )

        # 学习型注意力权重
        self.attention_net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),  # 输入：Concat([dir_x2, dir_x3])
            nn.GELU(),  # ReLU -> GELU
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # ReLU -> GELU
        )

    def forward(self, x2_feat: torch.Tensor, x3_feat: torch.Tensor) -> torch.Tensor:
        # 1. 空间对齐：将 x3 (1/16) 上采样到 x2 (1/8)
        x3_up = F.interpolate(x3_feat, size=x2_feat.shape[2:], mode='bilinear', align_corners=False)

        # 2. 提取潜在方向场 (Latent Direction Field)
        dir_x2 = self.direction_extractor(x2_feat)
        dir_x3 = self.direction_extractor(x3_up)

        # 归一化为单位向量 (Unit Vector)
        dir_x2 = F.normalize(dir_x2, p=2, dim=1)
        dir_x3 = F.normalize(dir_x3, p=2, dim=1)

        # 3. 计算一致性注意力 (Consistency Attention)
        # 物理项: 余弦相似度
        cosine_sim = F.cosine_similarity(dir_x2, dir_x3, dim=1).unsqueeze(1)

        # 学习项: 神经网络判断
        dir_cat = torch.cat([dir_x2, dir_x3], dim=1)
        learned_att = self.attention_net(dir_cat)

        # 混合注意力
        final_att = (cosine_sim + learned_att) / 2.0

        # 4. 加权融合
        # 思想：方向一致性高的地方，特征置信度高，予以保留；否则抑制。
        features_cat = torch.cat([x2_feat, x3_up], dim=1)
        weighted_features = features_cat * final_att
        fused_features = self.fusion_conv(weighted_features)

        return fused_features


class DirectionGuidedSmoothing(nn.Module):
    """
    模块3 (核心创新)：方向引导的各向异性平滑 (DGS)
    Paper Story: "Differentiable Anisotropic Diffusion Module.
    模拟热扩散过程，扩散系数由局部方向一致性动态决定。
    沿着道路方向进行特征平滑（连通断点），垂直道路方向阻断扩散（保护边缘）。"
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        # 可学习的温度系数，控制扩散的“敏锐度”
        # 初始化为 10.0，让 Softmax 比较尖锐，区分度更高
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, theta, width):
        """
        Args:
            theta: [B, 2, H, W] 归一化的方向场
            width: [B, 1, H, W] 宽度场
        """
        B, _, H, W = theta.shape

        # 1. 提取邻域特征 (Unfold)
        # 将每个像素的 3x3 邻居提取出来
        # [B, 2, H, W] -> [B, 2*9, H*W] -> [B, 2, 9, H, W]
        theta_unfold = F.unfold(theta, kernel_size=self.kernel_size, padding=self.pad)
        theta_unfold = theta_unfold.view(B, 2, -1, H, W)

        width_unfold = F.unfold(width, kernel_size=self.kernel_size, padding=self.pad)
        width_unfold = width_unfold.view(B, 1, -1, H, W)

        # 2. 提取中心像素
        # 在 3x3 窗口中，索引 4 是中心点
        center_idx = (self.kernel_size ** 2) // 2
        center_theta = theta.unsqueeze(2)  # [B, 2, 1, H, W]

        # 3. 计算扩散核 (Diffusion Kernel)
        # 计算中心像素与周围像素的方向余弦相似度
        # Sim = cos(θ_center, θ_neighbor)
        # 维度变化: [B, 2, 1, H, W] * [B, 2, 9, H, W] -> sum(dim=1) -> [B, 1, 9, H, W]
        similarity = (center_theta * theta_unfold).sum(dim=1, keepdim=True)

        # 4. 计算动态权重 (Softmax)
        # 这里的 weights 就是各向异性扩散方程中的 "Conduction Coefficient"
        # 方向一致 -> sim大 -> weight大 -> 强平滑
        # 方向垂直 -> sim小 -> weight小 -> 不平滑
        weights = F.softmax(similarity * self.temperature, dim=2)  # [B, 1, 9, H, W]

        # 5. 执行平滑 (Weighted Average)
        # Width Smoothing
        width_smoothed = (width_unfold * weights).sum(dim=2)  # [B, 1, H, W]

        # Orientation Smoothing (向量平均)
        theta_smoothed = (theta_unfold * weights).sum(dim=2)  # [B, 2, H, W]
        # 平滑后的向量模长可能小于1，必须重新归一化
        theta_smoothed = F.normalize(theta_smoothed, p=2, dim=1, eps=1e-4)

        # 6. 残差融合 (Residual Connection)
        # 保留原始预测的低频信息，叠加平滑后的高频修正
        # 0.5/0.5 是经验值，也可以改为可学习的门控
        width_final = 0.5 * width + 0.5 * width_smoothed
        # theta_final = F.normalize(0.5 * theta + 0.5 * theta_smoothed, p=2, dim=1)
        theta_final = F.normalize(0.5 * theta + 0.5 * theta_smoothed, p=2, dim=1,eps=1e-4)

        return theta_final, width_final


class StepB_ComplexGeometry(nn.Module):
    """
    Step B 最终完整版：高效几何提取器
    输出: 1/4 尺度 (256x256 @ 1024 input)
    """

    def __init__(self,
                 x2_channels: int = 64,
                 x3_channels: int = 128,
                 hidden_dim: int = 128):
        super().__init__()

        # 1. 特征预处理
        self.feature_processor = MultiScaleFeatureProcessingModule(hidden_dim, x2_channels, x3_channels)

        # 2. 方向感知融合 (在 1/8 尺度进行，兼顾感受野与效率)
        self.direction_fusion = ContinuousDirectionAwareFusion(hidden_dim)

        # 3. 上采样到目标 1/4 尺度
        self.up_to_quarter = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # ReLU -> GELU
        )

        # 4. 原始预测头 (Raw Prediction Heads)
        self.head_theta = nn.Conv2d(hidden_dim, 2, 1)  # 输出 sin, cos
        self.head_width = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.GELU()  # ReLU -> GELU (注意：最后的输出限制依然由 forward 逻辑决定，或者可以加一个 ReLU 保证非负)
        )
        # 注意：head_width 如果需要非负，通常保留一个 ReLU 或者 Softplus。
        # 这里我把 Sequential 里的 ReLU 换成了 GELU，但在 forward 里可能需要额外的约束。
        # 为了稳健，我在 forward 里加上了 width_raw = F.relu(width_raw) 或者是绝对值。
        # 让我们检查一下 head_width 的最后一层。原始是 ReLU 保证非负。GELU 有负值。
        # 修正：width 必须非负。我们可以用 F.softplus 或者 F.relu。
        # 为了保持一致性，我们在 forward 里做处理，这里用 GELU 没问题，只是激活特征。

        # 5. 几何精炼模块 (Refinement Module)
        self.geometry_consistency = DirectionGuidedSmoothing(kernel_size=3)

    def forward(self, x2, x3):
        # x2: [B, C, H/8, W/8]
        # x3: [B, C, H/16, W/16]

        # 1. 预处理
        x2_proc, x3_proc = self.feature_processor(x2, x3)

        # 2. 融合 (隐式对齐 + 方向感知)
        fused_feat_1_8 = self.direction_fusion(x2_proc, x3_proc)

        # 3. 上采样到 1/4
        fused_feat_1_4 = self.up_to_quarter(fused_feat_1_8)

        # 4. 原始预测 (Raw)
        theta_raw = self.head_theta(fused_feat_1_4)
        theta_raw = F.normalize(theta_raw, p=2, dim=1, eps=1e-6)  # 强制单位向量

        width_raw = self.head_width(fused_feat_1_4)

        # [关键修正] 限制 width 的范围，防止 NaN
        # 1. softplus 保证非负
        width_raw = F.softplus(width_raw)
        # 2. clamp 保证不溢出 (限制在 64 像素以内，约等于 1/4 图上的 256 像素原图宽度)
        width_raw = torch.clamp(width_raw, max=64.0)

        # 5. 动态扩散平滑 (Refined)
        theta_refined, width_refined = self.geometry_consistency(theta_raw, width_raw)

        # [再次保险] 经过 DGS 后再次 clamp，确保万无一失
        # theta_refined = F.normalize(theta_refined, p=2, dim=1)
        theta_refined = F.normalize(theta_refined, p=2, dim=1, eps=1e-4)

        width_refined = torch.clamp(width_refined, min=0.0, max=64.0)

        # 返回:
        # theta_refined: [B, 2, H/4, W/4] -> 供 Step C 采样使用
        # width_refined: [B, 1, H/4, W/4] -> 供 Step C 确定步长
        # fused_feat_1_4: [B, 128, H/4, W/4] -> Step C 的特征底板
        return theta_refined, width_refined, fused_feat_1_4


# ==========================================
# 简单的维度测试
# ==========================================
if __name__ == "__main__":
    B = 2
    # 模拟输入 (原图 1024x1024)
    x2 = torch.randn(B, 64, 64, 64)  # 1/8
    x3 = torch.randn(B, 128, 32, 32)  # 1/16

    model = StepB_ComplexGeometry(x2_channels=64, x3_channels=128, hidden_dim=128)

    # 打印参数量，看看是否足够轻量
    params = sum(p.numel() for p in model.parameters())
    print(f"Step B Params: {params / 1e6:.2f} M")

    theta, width, feat = model(x2, x3)

    print("-" * 30)
    print(f"Output Theta: {theta.shape}")  # [2, 2, 256, 256]
    print(f"Output Width: {width.shape}")  # [2, 1, 256, 256]
    print(f"Output Feat : {feat.shape}")  # [2, 128, 256, 256]
    print("-" * 30)

    # 验证 DGS 模块是否保持了归一化
    norm = torch.norm(theta, p=2, dim=1)
    print(f"Theta Norm (Expect ~1.0): Min={norm.min().item():.4f}, Max={norm.max().item():.4f}")