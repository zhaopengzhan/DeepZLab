import os
import random
import shutil
import warnings

import numpy as np
import torch
from ptflops import get_model_complexity_info
from timm.scheduler import CosineLRScheduler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from utils.misc import adjust_checkpoint

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = 'nul'
os.environ['PYTHONWARNINGS'] = 'ignore'


def init_all():
    # from configs.Chesapeake.Chesapeake_weak_out2_config import ConfigManager
    # from configs.Mars.Mars_Doft_config import ConfigManager
    # from configs.Mars.Mars_Gale_Doftv4_im16_config import ConfigManager
    from configs.Mars.Mars_Gale_Doftv4_im16_config import ConfigManager

    configManager = ConfigManager()
    args = configManager.get_config()

    # args
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # makedirs
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.fig_path, exist_ok=True)

    print('args', args)
    return configManager
    pass


def main():
    configManager = init_all()
    args, train_dl, val_list, model, train_one_epoch, val_one_epoch = configManager.get_all()
    writer = SummaryWriter(args.log_path)

    model = model.to(args.device)
    # summary(model, (4, 224, 224))
    macs, params = get_model_complexity_info(model, (args.in_channels, args.image_size, args.image_size),
                                             as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

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

            # weight_path = r'F:\Projects4\DeepZLab\run\weights\best_weight_Doftv4_cls9_Gale_Crater.pth'
            # checkpoint = torch.load(weight_path, weights_only=False)
            # updated_checkpoint = adjust_checkpoint(checkpoint['state_dict'], model.state_dict())
            # model.load_state_dict(updated_checkpoint, strict=False)
            # print("=> load checkpoint found at '{}'".format(weight_path))

    print(f'train start, device {args.device}, batch_size {args.batch_size}')

    # 配置调度器
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=200,  # 总的退火阶段
        lr_min=args.lr / 1e3,  # 最小学习率
        warmup_t=1,  # 热身阶段步数
        warmup_lr_init=args.lr,  # 热身初始学习率
        cycle_limit=2,  # 余弦退火循环次数
        t_in_epochs=True  # 是否以epoch为单位调整
    )

    for epoch in range(start_epoch, args.end_epoch):
        scheduler.step(epoch)
        train_one_epoch(args, train_dl, model, optimizer, epoch, writer, mode='train')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_pre1': best_pre1,
            'optimizer': optimizer.state_dict(),
        }, args.save_path)

        # mIoU = val_one_epoch(args, model, epoch)

        if epoch % args.epoch_print_result == 0 and epoch > 0:
            mIoU = val_one_epoch(args, model, epoch, val_list)

            best_pre1 = max(best_pre1, mIoU)
            print(f' * mIoU {mIoU:.3f}')
            print(f' * best mIoU {best_pre1:.3f}')

            writer.add_scalar('val mIoU', mIoU, epoch)

            if best_pre1 == mIoU:
                shutil.copyfile(args.save_path, args.save_path.replace('checkpoint', 'best_weight'))

            # 每轮权重
            # shutil.copyfile(args.save_path, args.save_path.replace('.pth', f'_mIoU_{mIoU:.1%}_epoch_{epoch}.pth'))


def adjust_learning_rate(optimizer, epoch):
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
