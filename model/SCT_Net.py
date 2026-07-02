import torch
import torch.nn as nn

# 假设你的目录结构是 model/Step_A.py, model/Step_B.py 等
# 如果不是，请根据实际情况修改 import 路径
from model.Step_A import LSKNet_Backbone
from model.Step_B import StepB_ComplexGeometry
from model.Step_C import DirectionalSamplerStepC
from model.Step_D import StepD_UltimateDecoder


class SCT_Net(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1,
                 embed_dims=[64, 128, 320, 512],  # 对应 Step A 的配置
                 hidden_dim=128):  # 内部处理通道数 (Step B/C/D 统一)
        super(SCT_Net, self).__init__()

        print("______________________________")

        # ====================================================
        # 1. Step A: Backbone (LSKNet + Spatial FiLM)
        # ====================================================
        # 输出: x1(1/4), x2(1/8), x3(1/16)
        self.backbone = LSKNet_Backbone(
            in_chans=in_chans,
            embed_dims=embed_dims
        )

        # ====================================================
        # 2. Step B: Geometry Learning (1/4 Scale)
        # ====================================================
        # 输入: x2(1/8), x3(1/16)
        # 输出: theta, width, feat_b (均为 1/4 尺度)
        self.step_b = StepB_ComplexGeometry(
            x2_channels=embed_dims[1],  # 128
            x3_channels=embed_dims[2],  # 256
            hidden_dim=hidden_dim  # 128
        )

        # ====================================================
        # 3. Step C: Directional Sampling (1/4 Scale)
        # ====================================================
        # 输入: feat_b, theta, width
        # 输出: feat_c (1/4 尺度, 包含长距离上下文)
        self.step_c = DirectionalSamplerStepC(
            in_channels=hidden_dim,  # 128
            out_channels=hidden_dim,  # 128
            num_steps=4,
            step_scales=[0.5, 1.0, 1.5, 2.0],  # 采样步数
            base_step=2.0  # 基础步长
        )

        # ====================================================
        # 4. Step D: Ultimate Decoder (1/1 Scale)
        # ====================================================
        # 输入: x1 (细节, 1/4), feat_c (上下文, 1/4)
        # 输出: logits (1/1)
        self.step_d = StepD_UltimateDecoder(
            in_channels_ctx=hidden_dim,  # 128
            in_channels_x1=embed_dims[0],  # 64
            hidden_dim=hidden_dim,  # 128
            num_classes=num_classes
        )

    def forward(self, x):
        """
        x: [B, 3, H, W] (原始输入图像)
        """

        # --- Step A: 特征提取 ---
        # feats = {'x1': [B,64,H/4,W/4], 'x2':..., 'x3':...}
        feats = self.backbone(x)

        # --- Step B: 几何预测 (输出对齐到 1/4 尺度) ---
        # theta: [B, 2, H/4, W/4]
        # width: [B, 1, H/4, W/4]
        # feat_b: [B, 128, H/4, W/4]
        theta, width, feat_b = self.step_b(feats['x2'], feats['x3'])

        # --- Step C: 向量化采样 ---
        # feat_c: [B, 128, H/4, W/4]
        feat_c = self.step_c(feat_b, theta, width)

        # --- Step D: 解码与上采样 ---
        # 融合 x1 (细节) 和 feat_c (上下文)
        # logits: [B, 1, H, W]
        logits = self.step_d(feats['x1'], feat_c)

        # 返回字典，匹配 utils/loss.py 中 DLSNetTotalLoss 的输入要求
        return {
            'out': logits,  # 用于计算 Dice + BCE Loss
            'theta': theta,  # 用于计算 Step B 几何 Loss
            'width': width  # 用于计算 Step B 几何 Loss
        }


if __name__ == "__main__":
    # 简单的冒烟测试 (Smoke Test)
    model = SCT_Net()

    # 模拟输入 (1024x1024)
    x = torch.randn(2, 3, 1024, 1024)

    # 前向传播
    outputs = model(x)

    print("-" * 30)
    print("DLSNet Output Check:")
    print(f"Seg Logits : {outputs['out'].shape}")  # 应为 [2, 1, 1024, 1024]
    print(f"Pred Theta : {outputs['theta'].shape}")  # 应为 [2, 2, 256, 256]
    print(f"Pred Width : {outputs['width'].shape}")  # 应为 [2, 1, 256, 256]
    print("-" * 30)

    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")