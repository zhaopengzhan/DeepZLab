import argparse
import glob
import os
import random
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from timm.scheduler import CosineLRScheduler
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from dataloader.Chesapeake import Chesapeake
from utils.losses import sigmoid_focal_loss, dice_loss
from utils.misc import Wrapper, adjust_checkpoint


parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

warnings.filterwarnings("ignore")
# 设置环境变量 CPL_LOG 为 'nul'，屏蔽所有日志输出
os.environ['CPL_LOG'] = 'nul'
# 设置 PYTHONWARNINGS 环境变量为 ignore
os.environ['PYTHONWARNINGS'] = 'ignore'


def init_all():
    args.seed = 1234
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # train
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.end_epoch = 1000
    args.num_classes = 17
    args.mobile_name = 'Doftv3'
    args.in_channels = 4

    # save
    job_name = f'{args.mobile_name}_small2_Chesapeake_ny'
    args.save_path = rf'run/weights1/checkpoint_{job_name}.pth'
    args.pretrain_path = rf'run/weights1/checkpoint_{job_name}.pth'
    args.log_path = rf'run/logs/{job_name}'
    args.fig_path = rf'run/figures/{job_name}'

    # optimizer
    args.lr = 1e-3

    # dataset
    args.batch_size = 12
    args.num_workers = 3
    args.num_chips_per_tile = 12
    args.image_size = 64
    args.stride = 32
    args.image_re_path = r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-train_tiles\*_naip-new.tif'

    # val
    args.val_dir = r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles'

    # other
    args.epoch_print_result = 1

    #
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.fig_path, exist_ok=True)

    print('args', args)
    pass


def build_dataloader():
    train_image_list = glob.glob(args.image_re_path)
    train_label_lr_list = [filename.replace('_naip-new.tif', '_nlcd.tif') for filename in train_image_list]
    train_label_hr_list = [filename.replace('_naip-new.tif', '_lc.tif') for filename in train_image_list]

    # test_image_list = glob.glob(
    #     r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles\*_naip-new.tif')
    # test_label_lr_list = [filename.replace('_naip-new.tif', '_nlcd.tif') for filename in test_image_list]
    # test_label_hr_list = [filename.replace('_naip-new.tif', '_lc.tif') for filename in test_image_list]

    train_ds = Chesapeake(train_image_list, train_label_lr_list, train_label_hr_list,
                          chip_size=args.image_size, num_chips_per_tile=args.num_chips_per_tile)
    # test_ds = Chesapeake(test_image_list, test_label_lr_list, test_label_hr_list,
    #                      num_chips_per_tile=30 * 22, stride=224)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, pin_memory=False)
    # test_dl = DataLoader(test_ds, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True)
    test_dl = None

    print('dataloader finish')
    return train_dl, test_dl


def get_model():
    model = models.build_models(args.mobile_name,
                                in_channels=args.in_channels,
                                num_classes=args.num_classes,
                                img_size=args.image_size,
                                )


    return model


def build_trainer():
    from trainer.chesapeake.train_L2HNet import train_one_epoch
    from trainer.chesapeake.val_L2HNet import val_one_epoch
    return train_one_epoch, val_one_epoch

def build_trainer1():
    from trainer.chesapeake.train import train_one_epoch
    from trainer.chesapeake.val import val_one_epoch
    return train_one_epoch, val_one_epoch

def main():
    init_all()
    writer = SummaryWriter(args.log_path)

    # model
    model = get_model().to(args.device)
    # summary(model, (4, 224, 224))
    macs, params = get_model_complexity_info(model, (args.in_channels, args.image_size, args.image_size), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # data
    train_dl, test_dl = build_dataloader()

    # loss
    loss_wrapper = Wrapper()
    # [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    # t_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 4.0, 2.0, 1.0, 1.5, 1.5, 1.5, 2.0, 1.0, 1.0, 1.5, 1.5]) \
    #     .to(args.device)
    # TODO: 检查 ignore_index
    # ce_loss = nn.CrossEntropyLoss(ignore_index=0, weight=t_weights).to(args.device)
    ce_loss = nn.CrossEntropyLoss(ignore_index=0).to(args.device)
    loss_wrapper.register('ce_loss', ce_loss)
    loss_wrapper.register('focal_loss', sigmoid_focal_loss)
    loss_wrapper.register('dice_loss', dice_loss)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4,
                                  betas=(0.9, 0.999),
                                  amsgrad=False, )

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=0.99,
    #                             weight_decay=1e-4,
    #                             nesterov=True)

    best_pre1 = 0
    start_epoch = 0
    if args.pretrain_path:
        if os.path.isfile(args.pretrain_path):
            print("=> loading checkpoint '{}'".format(args.pretrain_path))
            checkpoint = torch.load(args.pretrain_path, weights_only=False)

            start_epoch = checkpoint['epoch']
            best_pre1 = checkpoint['best_pre1']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> have loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrain_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain_path))

            # weight_path = r'F:\Projects2\SpiderNet\run\weights\checkpoint_ConViT_cls11_ESA_GLC10_label.pth'
            # checkpoint = torch.load(weight_path, weights_only=False)
            # updated_checkpoint = adjust_checkpoint(checkpoint['state_dict'], model.state_dict())
            # model.load_state_dict(updated_checkpoint,strict=False)
            # print("=> load checkpoint found at '{}'".format(weight_path))

    print(f'train start, device {args.device}, batch_size {args.batch_size}')

    # 配置调度器
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=200,  # 总的退火阶段
        lr_min=1e-6,  # 最小学习率
        # warmup_t=1,  # 热身阶段步数
        # warmup_lr_init=1e-4,  # 热身初始学习率
        cycle_limit=1,  # 余弦退火循环次数
        t_in_epochs=True  # 是否以epoch为单位调整
    )
    # TODO：修改训练逻辑

    train_one_epoch, val_one_epoch = build_trainer()
    for epoch in range(start_epoch, args.end_epoch):
        scheduler.step(epoch)
        train_one_epoch(args, train_dl, model, optimizer, epoch, writer, loss_wrapper, mode='train')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_pre1': best_pre1,
            'optimizer': optimizer.state_dict(),
        }, args.save_path)

        # mIoU = val_one_epoch(args, model, epoch)

        if epoch % args.epoch_print_result == 0 and epoch > 0:
            mIoU = val_one_epoch(args, model, epoch)

            best_pre1 = max(best_pre1, mIoU)
            print(f' * mIoU {mIoU:.3f}')
            print(f' * best mIoU {best_pre1:.3f}')

            writer.add_scalar('val mIoU', mIoU, epoch)

            if best_pre1 == mIoU:
                shutil.copyfile(args.save_path, args.save_path.replace('checkpoint', 'best_weight'))

            # 每轮权重
            shutil.copyfile(args.save_path, args.save_path.replace('.pth', f'_mIoU_{mIoU:.1%}_epoch_{epoch}.pth'))


def adjust_learning_rate(optimizer, epoch):
    # 手动调节lr
    # if epoch > 1:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr  # 修改优化器里的参数lr
    if epoch > 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    if epoch > 25:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    if epoch > 55:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-6


if __name__ == '__main__':
    main()
