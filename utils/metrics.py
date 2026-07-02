import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = None

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].long() + pre_image[mask].long()
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros((self.num_class, self.num_class), device=gt_image.device)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        if self.confusion_matrix is not None:
            self.confusion_matrix.zero_()

    def Mean_Intersection_over_Union(self):
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        return torch.mean(iou).item()

    def Get_Binary_Metrics(self):
        """
        返回二分类指标，同时附带 mIoU
        """
        if self.confusion_matrix is None:
            return {"IoU": 0.0, "mIoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}

        # 1. 计算二分类（道路类 Class 1）指标
        TP = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[0, 1]
        FN = self.confusion_matrix[1, 0]

        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        road_iou = TP / (TP + FP + FN + 1e-10)

        # 2. 计算 mIoU (背景 IoU + 道路 IoU) / 2
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou_per_class = intersection / (union + 1e-10)
        mIoU = torch.mean(iou_per_class)

        return {
            "IoU": road_iou.item(),  # Road IoU
            "mIoU": mIoU.item(),  # Mean IoU
            "F1": f1.item(),
            "Precision": precision.item(),
            "Recall": recall.item()
        }