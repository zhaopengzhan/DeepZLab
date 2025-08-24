import gc
import logging
import os
import random
import shutil

import numpy as np
import torch
from ptflops import get_model_complexity_info
from timm.scheduler import CosineLRScheduler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = 'nul'
os.environ['PYTHONWARNINGS'] = 'ignore'


def init_all():
    from configs.Chesapeake.l2h_config import ConfigManager

    configManager = ConfigManager()
    args = configManager.get_config()

    # args
    gc.collect()
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # makedirs
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.fig_path, exist_ok=True)
    os.makedirs(args.fig_path, exist_ok=True)

    # logger
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.log_path, 'train.log'), encoding="utf-8")
        ],
        force=True  # ✅ 覆盖旧配置，无需手动清理
    )

    logger = logging.getLogger()

    logger.info(f'args {args}')
    return configManager, logger


def main():
    configManager, logger = init_all()
    args, train_dl, val_list, model, train_one_epoch, val_one_epoch = configManager.get_all()
    writer = SummaryWriter(args.log_path)

    model = model.to(args.device)

    # summary(model, (4, 224, 224))
    macs, params = get_model_complexity_info(model, (args.in_channels, args.image_size, args.image_size),
                                             as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4,
                                  betas=(0.9, 0.999),
                                  amsgrad=False, )

    best_pre1 = 0
    start_epoch = 0
    if hasattr(args, "pretrain_path") and os.path.isfile(args.pretrain_path):
        print("=> loading checkpoint '{}'".format(args.pretrain_path))
        checkpoint = torch.load(args.pretrain_path, weights_only=False)

        start_epoch = checkpoint['epoch']
        best_pre1 = checkpoint['best_pre1']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("=> have loaded checkpoint '{}' (epoch {})"
                    .format(args.pretrain_path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found")

    logger.info(f'train start, device {args.device}, batch_size {args.batch_size}')

    # scheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=int(args.end_epoch * 0.5),
        lr_min=args.lr * 1e-4,
        warmup_t=int(args.end_epoch * 0.01),
        warmup_lr_init=args.lr * 1e-2,
        cycle_limit=2,
        t_in_epochs=True
    )

    for epoch in range(start_epoch, args.end_epoch):
        scheduler.step(epoch)
        train_one_epoch(args, train_dl, model, optimizer, epoch, writer)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_pre1': best_pre1,
            'optimizer': optimizer.state_dict(),
        }, args.save_path)

        if epoch % args.epoch_print_result == 0:
            mIoU = val_one_epoch(args, model, epoch, val_list)

            best_pre1 = max(best_pre1, mIoU)
            logger.info(f' * mIoU {mIoU:.3f}')
            logger.info(f' * best mIoU {best_pre1:.3f}')

            writer.add_scalar('val mIoU', mIoU, epoch)

            if best_pre1 == mIoU:
                shutil.copyfile(args.save_path, args.save_path.replace('checkpoint', 'best_weight'))

            # 每轮权重
            # shutil.copyfile(args.save_path, args.save_path.replace('.pth', f'_mIoU_{mIoU:.1%}_epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()
