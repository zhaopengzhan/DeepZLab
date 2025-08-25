import argparse

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import models
from configs.BaseConfig import BaseConfig
from dataloader import build_dataloader
from utils.misc import *


class ConfigManager(BaseConfig):

    def set_config(self):
        args = edict()

        # train
        args.seed = 42
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args.end_epoch = 1000
        args.epoch_print_result = 10
        args.batch_print_result = 10

        # optimizer
        args.lr = 1e-4

        # model
        args.num_classes = 17
        args.in_channels = 4
        args.model_name = 'SpectralTokenizer'

        # dataset
        args.train_data = 'Chesapeake_L2H'
        args.val_data = 'Chesapeake_val'
        args.batch_size = 12
        args.num_workers = 4
        args.num_chips_per_tile = 360
        args.image_size = 384
        args.stride = int(args.image_size / 2)

        # save
        special = 'e1'
        job_name = f'{args.model_name}_o{args.num_classes}_i{args.in_channels}_ds{args.train_data}_{special}'
        args.save_path = rf'run/weights/checkpoint_{job_name}.pth'
        # args.pretrain_path = rf'run/weights/checkpoint_{job_name}.pth'
        args.log_path = rf'run/logs/{job_name}'
        args.fig_path = rf'run/figures/{job_name}'

        # path
        args.image_re_path = r'F:\zpz\datasets\Remote Sensing\cvpr_chesapeake_landcover\*_extended-debuffered-train_tiles\*_naip-new.tif'
        args.val_dir = r'F:\zpz\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles'

        return args
        pass

    @calRunTime
    def set_train_dl(self):
        args = self.config
        # train_image_list = glob.glob(args.image_re_path)
        # train_label_lr_list = [filename.replace('_naip-new.tif', '_nlcd.tif') for filename in train_image_list]
        # train_label_hr_list = [filename.replace('_naip-new.tif', '_lc.tif') for filename in train_image_list]

        train_ds = build_dataloader(
            args.train_data
        )

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              pin_memory=True)

        return train_dl
        pass

    def set_val_list(self):
        args = self.config
        # val_list = glob.glob(rf'{args.val_dir}\*_naip-new.tif')
        val_list = [
            r'F:\zpz\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles\m_4107506_se_18_1_naip-new.tif',
            r'F:\zpz\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-test_tiles\m_4307563_nw_18_1_naip-new.tif',
        ]


        return val_list
        pass

    @calRunTime
    def set_model(self):
        args = self.config
        model = models.build_model(
            args.model_name,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            image_size=args.image_size
        )

        return model
        pass

    def set_train_one_epoch(self):
        from trainer.chesapeake.train_L2HNet import train_one_epoch
        return train_one_epoch
        pass

    def set_val_one_epoch(self):
        from trainer.chesapeake.val_L2HNet import val_one_epoch
        return val_one_epoch
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description='DeepZLab')

        # train
        parser.add_argument('--device', type=str, required=False)
        parser.add_argument('--end_epoch', type=int, required=False)

        args = parser.parse_args()
        return args
        pass
