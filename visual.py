import os
import random
import warnings

import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = 'nul'
os.environ['PYTHONWARNINGS'] = 'ignore'


def init_all():
    from configs.Mars.Mars_Gale_Doftv4_im16_visual_config import ConfigManager

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

    model = model.to(args.device)

    macs, params = get_model_complexity_info(model, (args.in_channels, args.image_size, args.image_size),
                                             as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if args.pretrain_path:
        if os.path.isfile(args.pretrain_path):
            print("=> loading checkpoint '{}'".format(args.pretrain_path))
            checkpoint = torch.load(args.pretrain_path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("=> have loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrain_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain_path))

    print(f'train start, device {args.device}, batch_size {args.batch_size}')

    val_one_epoch(args, model, -1, val_list, return_metric=False)


if __name__ == '__main__':
    main()
