import torch
import torch.nn as nn
import torch.nn.functional as F


# [移除] from utils.loss_B import StepBLossWithRegularization

class DiceBCELoss(nn.Module):
    """
    Step D
    组合 Binary Cross Entropy (针对像素精度) + Dice Loss (针对拓扑连通性)
    """

    def __init__(self, bce_weight=0.3, dice_weight=0.7, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target):
        """
        pred_logits: [B, 1, H, W] (Step D 的原始输出)
        target:      [B, 1, H, W] (0/1 真值掩码)
        """

        bce = self.bce_loss(pred_logits, target)

        pred_probs = torch.sigmoid(pred_logits)

        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # Dice Coefficient = 2 * Intersection / Union
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce + self.dice_weight * dice_loss



class SCT_NetTotalLoss(nn.Module):
    def __init__(self,
                 weight_step_d=1.0):
        super().__init__()

        self.loss_d_fn = DiceBCELoss()

        # self.weight_b = weight_step_b # 不再需要存储
        self.weight_d = weight_step_d

        self.loss_stats = {}

    def forward(self, model_outputs, batch_targets):
        """
        model_outputs: 字典，包含 {'out': logits, ...}
        batch_targets: 字典，包含 {'gt_mask'}
        """

        loss_d = self.loss_d_fn(
            pred_logits=model_outputs['out'],
            target=batch_targets['gt_mask']
        )

        total_loss = self.weight_d * loss_d

        self.loss_stats = {
            'total': total_loss.item(),
            'seg_loss': loss_d.item(),
        }

        return total_loss

    def get_stats(self):
        return self.loss_stats