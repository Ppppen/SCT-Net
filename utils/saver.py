import os
import shutil
import torch
from datetime import datetime
import glob


class Saver(object):
    def __init__(self, args):
        self.args = args

        # 1. 自动生成带时间戳的唯一实验目录
        # 格式: run/deepglobe/DLSNet_20231207_120000
        # 这样每次训练都不会覆盖之前的实验
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(
            'run',
            args.dataset,
            f"{args.checkname}_{run_id}"
        )

        # 创建目录
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # 保存参数配置到txt，方便以后查阅这次训练用了什么参数
        self.save_experiment_config()

    def save_checkpoint(self, state, is_best, epoch, save_interval=20):
        """
        保存模型权重
        Args:
            state: 模型参数字典 (包含 model, optimizer, epoch, best_iou 等)
            is_best: bool, 是否是当前最优模型
            epoch: 当前轮数
            save_interval: 固定保存间隔 (如 20)
        """
        # 1. 始终保存为 'last.pth' (覆盖式，用于断点续训)
        # 无论跑到哪一轮，这个文件永远是最新的
        last_filename = os.path.join(self.experiment_dir, 'checkpoint_last.pth')
        torch.save(state, last_filename)

        # 2. 如果是 Best，拷贝一份命名为 'model_best.pth'
        # 这个文件永远保留验证集 IoU 最高的那个版本
        if is_best:
            best_filename = os.path.join(self.experiment_dir, 'model_best.pth')
            shutil.copyfile(last_filename, best_filename)

        # 3. 每隔 save_interval 保存一份历史记录 (如 epoch_20.pth, epoch_40.pth)
        # 这些文件不会被覆盖，用于后期分析模型训练过程
        if epoch % save_interval == 0:
            interval_filename = os.path.join(self.experiment_dir, f'checkpoint_epoch_{epoch}.pth')
            shutil.copyfile(last_filename, interval_filename)

    def save_experiment_config(self):
        """将 args 参数保存为文本文件"""
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        with open(logfile, 'w') as f:
            for key, val in vars(self.args).items():
                f.write(f"{key}: {val}\n")

    def get_log_dir(self):
        """返回实验目录路径，用于 Tensorboard"""
        return self.experiment_dir