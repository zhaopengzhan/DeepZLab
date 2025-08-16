import argparse
import glob

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import models
from configs.BaseConfig import BaseConfig
from dataloader.Chesapeake_hierarchy import Chesapeake
from dataloader.SHLU import SHLU
from utils.misc import *


class ConfigManager(BaseConfig):

    def set_config(self):
        args = edict()

        # train
        args.seed = 42
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args.end_epoch = 1000
        args.epoch_print_result = 5

        # optimizer
        args.lr = 1e-3

        # model
        args.num_classes = 21
        args.in_channels = 3
        args.mobile_name = 'Skip_FCN_invert_SHLU'
        args.label_type = 'SHLU'

        # save
        special = 'e13'
        job_name = f'{args.mobile_name}_cls{args.num_classes}_in{args.in_channels}_{args.label_type}_{special}'
        args.save_path = rf'run/weights/checkpoint_{job_name}.pth'
        args.pretrain_path = rf'run/weights/checkpoint_{job_name}.pth'
        args.log_path = rf'run/logs/{job_name}'
        args.fig_path = rf'run/figures/{job_name}'

        # dataset
        args.batch_size = 16
        args.num_workers = 4
        args.num_chips_per_tile = 32
        args.image_size = 512
        args.stride = args.image_size // 2

        # path
        args.image_re_path = r'F:\zpz\datasets\SHLU\HR_Optical_image\*.tif'
        # args.val_dir = r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles'
        # args.val_dir = r'F:\datasets\Remote Sensing\cvpr_chesapeake_landcover\*_extended-debuffered-test_tiles'

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
        image_list = glob.glob(args.image_re_path)
        build_list = [fn.replace('HR_Optical_image', 'Building_label_train') for fn in image_list]
        landuse_list = [fn.replace('HR_Optical_image', 'Landuse_label_train') for fn in image_list]

        train_ds = SHLU(image_list, build_list, landuse_list,
                              chip_size=args.image_size, num_chips_per_tile=args.num_chips_per_tile)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              pin_memory=True)

        return train_dl
        pass

    def set_val_list(self):
        args = self.config
        image_list = glob.glob(rf'{args.val_dir}\*_naip-new.tif')
        return image_list
        pass

    @calRunTime
    def set_model(self):
        args = self.config
        model = models.build_models(args.mobile_name,
                                    in_channels=args.in_channels,
                                    num_classes=args.num_classes,
                                    image_size=args.image_size,
                                    )

        return model
        pass


    def set_train_one_epoch(self):
        from trainer.SHLU.train_invert  import train_one_epoch
        return train_one_epoch
        pass


    def set_val_one_epoch(self):
        from trainer.SHLU.val_invert import val_one_epoch
        return val_one_epoch
        pass