import argparse
import json
import os
import os.path
import random
import re
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
from dataloader.GLC_L2H import GLC_L2H
from utils.label_maps.lm_pl import label_lr_class
from utils.misc import Wrapper, adjust_checkpoint

# TODO：未来修改训练过程的点
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
    args.in_channels = 3
    args.mobile_name = 'SpiderNet'
    args.label_type = 'ESA_GLC10_label'
    args.num_classes = len(label_lr_class[args.label_type])

    # save
    job_name = f'{args.mobile_name}_cls{args.num_classes}_{args.label_type}'
    args.save_path = rf'run/weights1/checkpoint_{job_name}.pth'
    args.pretrain_path = rf'run/weights1/checkpoint_{job_name}.pth'
    args.log_path = rf'run/logs/{job_name}'
    args.fig_path = rf'run/figures/{job_name}'

    # optimizer
    args.lr = 1e-3

    # dataset
    args.batch_size = 8
    args.num_workers = 2
    args.num_chips_per_tile = 4
    args.image_size = 224
    args.stride = 112
    # args.image_re_path = r'F:\datasets\OpenEarthMap\OpenEarthMap_wo_xBD\*\labels\*.tif'

    # val
    args.val_dir = r'F:\datasets\Remote Sensing\OpenEarthMap_with_xBD'

    # other
    args.epoch_print_result = 2

    #
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.fig_path, exist_ok=True)

    print('args', args)
    pass


def build_dataloader():
    root_dir = r'F:\datasets\Remote Sensing\OpenEarthMap_with_xBD'
    json_path = os.path.join(root_dir, 'my_dataset_split.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    train_fn_list = data['train']
    test_fn_list = data['test']

    train_image_list = []
    for fn in train_fn_list:
        city = re.compile(r"^(.*)_\d+\.tif$").findall(fn)[0]
        # if city in ['tonga', 'leilane_estates']:
        #     continue
        train_image_list.append(os.path.join(root_dir, rf'{city}\images\{fn}'))

    test_list = []
    for fn in test_fn_list:
        city = re.compile(r"^(.*)_\d+\.tif$").findall(fn)[0]
        # if city in ['tonga', 'leilane_estates']:
        #     continue
        test_list.append(os.path.join(root_dir, rf'{city}\images\{fn}'))

    train_label_lr_list = [filename.replace('images', args.label_type.split('_label')[0]) for filename in train_image_list]

    train_ds = GLC_L2H(train_image_list, train_label_lr_list, label_type=args.label_type,
                       chip_size=args.image_size, num_chips_per_tile=args.num_chips_per_tile)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                          pin_memory=True)

    return train_dl, test_list


def get_model():
    model = models.build_models(args.mobile_name,
                                in_channels=args.in_channels,
                                num_classes=args.num_classes)

    return model


def build_trainer2():
    from trainer.poland.train_pl_out1 import train_one_epoch
    from trainer.poland.val_pl_out1 import val_one_epoch

    return train_one_epoch, val_one_epoch


def build_trainer():
    from trainer.poland.train_pl_out2 import train_one_epoch
    from trainer.poland.val_pl_out2 import val_one_epoch

    return train_one_epoch, val_one_epoch


def main():
    init_all()
    writer = SummaryWriter(args.log_path)

    # model
    model = get_model().to(args.device)

    macs, params = get_model_complexity_info(model, (args.in_channels, 224, 224), as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # data
    train_dl, test_list = build_dataloader()

    # loss
    loss_wrapper = Wrapper()

    # TODO: 检查 ignore_index
    ce_loss = nn.CrossEntropyLoss(ignore_index=0).to(args.device)
    loss_wrapper.register('ce_loss', ce_loss)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4,
                                  betas=(0.9, 0.999),
                                  amsgrad=False, )

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

            weight_path = r'F:\Projects2\SpiderNet\run\weights1\best_weight_SpiderNet2_Chesapeake_ny.pth'
            checkpoint = torch.load(weight_path, weights_only=False)
            updated_checkpoint = adjust_checkpoint(checkpoint['state_dict'], model.state_dict())
            model.load_state_dict(updated_checkpoint)
            print("=> load checkpoint found at '{}'".format(weight_path))

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
        model.freeze_parameter(freeze=epoch < 5)
        train_one_epoch(args, train_dl, model, optimizer, epoch, writer, loss_wrapper, mode='train')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_pre1': best_pre1,
            'optimizer': optimizer.state_dict(),
        }, args.save_path)
        # mIoU = val_one_epoch(args, model, epoch, test_list)
        if epoch % args.epoch_print_result == 0 and epoch > 0:
            #
            mIoU = val_one_epoch(args, model, epoch, test_list)

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
