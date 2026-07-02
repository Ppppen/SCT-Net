import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from timm.scheduler import CosineLRScheduler

# 导入自定义模块
from model.SCT_Net import SCT_Net
from dataloader.dataloader import RoadGeometryDataLoader
from utils.loss import SCT_NetTotalLoss
from utils.metrics import Evaluator
from utils.saver import Saver

# 开启 cudnn benchmark
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="DLSNet Training (Implicit Geometry Version)")

    # --- 数据路径 ---
    parser.add_argument('--dataset', type=str, default='CHIN6_CUG', choices=['deepglobe', 'spacenet', 'cityscale','CHIN6_CUG'])

    parser.add_argument('--train_img_dir', type=str, default='')
    parser.add_argument('--train_gt_dir', type=str, default='')

    parser.add_argument('--val_img_dir', type=str, default='')
    parser.add_argument('--val_gt_dir', type=str, default='')

    # --- 训练超参 ---
    parser.add_argument('--epochs', type=int, default=200, help='总训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='预热轮数')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)

    # 优化器
    parser.add_argument('--lr', type=float, default=1e-4, help='初始峰值学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 损失权重
    parser.add_argument('--loss_weight_d', type=float, default=1.0)

    # 系统
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--checkname', type=str, default='')
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

        # 1. Saver
        self.saver = Saver(args)
        print(f" Experiment Dir: {self.saver.get_log_dir()}")

        # 2. Dataloader
        # [修改] 移除了 npy_root 参数
        self.train_loader = RoadGeometryDataLoader(
            image_dir=args.train_img_dir,
            gt_dir=args.train_gt_dir,
            batch_size=args.batch_size,
            image_size=(512, 512),
            augmentation=True,
            dataset_type=args.dataset,
            num_workers=args.num_workers
        ).dataloader

        if args.val_img_dir:
            self.val_loader = RoadGeometryDataLoader(
                image_dir=args.val_img_dir,
                gt_dir=args.val_gt_dir,
                batch_size=1,
                image_size=(512, 512),
                augmentation=False,
                shuffle=False,
                dataset_type=args.dataset
            ).dataloader
        else:
            self.val_loader = None

        # 3. Model
        print("Building DLSNet Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DLSNet(in_chans=3, num_classes=1).to(self.device)

        if len(args.gpu_ids.split(',')) > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 4. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 5. Scheduler
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=args.epochs,
            lr_min=args.min_lr,
            warmup_t=args.warmup_epochs,
            warmup_lr_init=args.min_lr,
            cycle_limit=1,
            t_in_epochs=True
        )

        # 6. Loss & Metrics
        self.criterion = DLSNetTotalLoss(
            # weight_step_b=0.0,
            weight_step_d=args.loss_weight_d
        ).to(self.device)

        print(f" [Info] Training Mode: Implicit Geometry (No DWLoss).")

        self.evaluator = Evaluator(num_class=2)
        self.scaler = GradScaler()
        self.best_iou = 0.0
        self.start_epoch = 1

        # 7. Resume
        if args.resume:
            self._resume_checkpoint(args.resume)

    def _resume_checkpoint(self, resume_path):
        if not os.path.isfile(resume_path):
            print(f"Checkpoint not found: {resume_path}")
            return
        print(f" Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.best_iou = checkpoint.get('best_iou', 0.0)
        self.start_epoch = checkpoint['epoch'] + 1
        print(f" Resumed at Epoch {self.start_epoch}, Best IoU: {self.best_iou:.4f}")

    def training(self, epoch):
        self.model.train()
        self.evaluator.reset()


        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Train]")

        # [修改] 仅记录 total 和 seg loss
        loss_rec = {'total': 0.0, 'seg': 0.0}

        for i, sample in enumerate(tbar):
            img = sample['image'].to(self.device)

            # [修改] 此时 sample 中已经没有 theta/width 等数据了，只取 gt_mask
            targets = {
                'gt_mask': sample['gt_mask'].to(self.device)
            }

            self.optimizer.zero_grad()

            with autocast():
                preds = self.model(img)
                loss = self.criterion(preds, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            stats = self.criterion.get_stats()
            loss_rec['total'] += stats['total']
            loss_rec['seg'] += stats['seg_loss']

            # [修改] 移除了 geo_loss 的累加

            current_lr = self.optimizer.param_groups[0]['lr']

            # [修改] 进度条不再显示 wb
            tbar.set_postfix(loss=f"{stats['total']:.3f}", lr=f"{current_lr:.6f}")

        self.scheduler.step(epoch)
        print(f" -> Train Avg Loss: {loss_rec['total'] / len(self.train_loader):.4f}")

    def validation(self, epoch):
        if self.val_loader is None:
            self._save(epoch, is_best=False)
            return

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Val]")

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                img = sample['image'].to(self.device)
                target_mask = sample['gt_mask'].to(self.device)

                preds = self.model(img)
                logits = preds['out']
                pred_mask = (logits > 0).float()

                self.evaluator.add_batch(target_mask.squeeze(1), pred_mask.squeeze(1))

        metrics = self.evaluator.Get_Binary_Metrics()
        curr_iou = metrics['IoU']
        curr_miou = metrics['mIoU']
        curr_f1 = metrics['F1']

        print(f" -> Val Results: Road IoU: {curr_iou:.4f} | mIoU: {curr_miou:.4f} | F1: {curr_f1:.4f}")

        is_best = curr_iou > self.best_iou
        if is_best:
            self.best_iou = curr_iou
            print(f" New Best IoU: {self.best_iou:.4f}")

        self._save(epoch, is_best)

    def _save(self, epoch, is_best):
        self.saver.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if isinstance(self.model,
                                                                       torch.nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
        }, is_best, epoch, self.args.save_interval)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    trainer = Trainer(args)
    print(f" Start Training: {args.epochs} Epochs with timm Scheduler")

    for epoch in range(trainer.start_epoch, args.epochs + 1):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == "__main__":
    main()