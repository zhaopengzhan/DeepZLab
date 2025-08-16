import argparse
import glob

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import models
from configs.BaseConfig import BaseConfig
from dataloader.Chesapeake import Chesapeake_L2H
from utils.misc import *


class ConfigManager(BaseConfig):

    def set_config(self):
        args = edict()

        # train
        args.seed = 1234
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args.end_epoch = 1000
        args.epoch_print_result = 1

        # optimizer
        args.lr = 1e-3

        # model
        args.num_classes = 17
        args.in_channels = 4
        args.mobile_name = 'Doftv3'

        # save
        job_name = f'{args.mobile_name}_small2_Chesapeake_ny'
        args.save_path = rf'run/weights1/checkpoint_{job_name}.pth'
        args.pretrain_path = rf'run/weights1/checkpoint_{job_name}.pth'
        args.log_path = rf'run/logs/{job_name}'
        args.fig_path = rf'run/figures/{job_name}'

        # dataset
        args.batch_size = 12
        args.num_workers = 3
        args.num_chips_per_tile = 12
        args.image_size = 64
        args.stride = 32

        # path
        args.image_re_path = r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-train_tiles\*_naip-new.tif'
        args.val_dir = r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles'

        return args
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description='DeepZLab')
        # train
        parser.add_argument('--seed', type=int, required=False)
        parser.add_argument('--device', type=str, required=False)
        parser.add_argument('--end_epoch', type=int, required=False)
        parser.add_argument('--epoch_print_result', type=int, required=False)

        # optimizer
        parser.add_argument('--lr', type=float, required=False)

        # model
        parser.add_argument('--num_classes', type=int, required=False)
        parser.add_argument('--in_channels', type=int, required=False)
        parser.add_argument('--mobile_name', type=str, required=False)

        # save
        parser.add_argument('--save_path', type=str, required=False)
        parser.add_argument('--pretrain_path', type=str, required=False)
        parser.add_argument('--log_path', type=str, required=False)
        parser.add_argument('--fig_path', type=str, required=False)

        # dataset
        parser.add_argument('--batch_size', type=int, required=False)
        parser.add_argument('--num_workers', type=int, required=False)
        parser.add_argument('--num_chips_per_tile', type=int, required=False)
        parser.add_argument('--image_size', type=int, required=False)
        parser.add_argument('--stride', type=int, required=False)

        # path
        parser.add_argument('--image_re_path', type=str, required=False)
        parser.add_argument('--val_dir', type=str, required=False)

        args = parser.parse_args()
        return args

    @calRunTime
    def set_train_dl(self):
        args = self.config
        train_image_list = glob.glob(args.image_re_path)
        train_label_lr_list = [filename.replace('_naip-new.tif', '_nlcd.tif') for filename in train_image_list]
        train_label_hr_list = [filename.replace('_naip-new.tif', '_lc.tif') for filename in train_image_list]

        train_ds = Chesapeake(train_image_list, train_label_lr_list, train_label_hr_list,
                              chip_size=args.image_size, num_chips_per_tile=args.num_chips_per_tile)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              pin_memory=True)

        return train_dl
        pass

    def set_test_dl(self):
        pass

    @calRunTime
    def set_model(self):
        args = self.config
        model = models.build_models(args.mobile_name,
                                    in_channels=args.in_channels,
                                    num_classes=args.num_classes,
                                    img_size=args.image_size)

        return model
        pass

    def set_val_one_epoch(self):
        from trainer.chesapeake.val import val_one_epoch
        return val_one_epoch
        pass

    def set_train_one_epoch(self):
        from trainer.chesapeake.train import train_one_epoch
        return train_one_epoch
        pass
