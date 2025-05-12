import gc
import glob
import os
import random
import re

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader.GLC_L2H import BaseDataset
from utils import cm
from utils.label_maps import get_xx_label_map
from utils.metrics import Evaluator, evaluate
from utils.misc import AverageMeter,  calRunTimer


class TestDataset(Dataset, BaseDataset):
    def __init__(self, image_path, image_size=224, stride=112):
        super().__init__()
        with rasterio.open(image_path) as image_ds:
            height, width = image_ds.height, image_ds.width
            image_np = image_ds.read()

        self.image_np = image_np
        self.image_size = image_size
        self.stride = stride

        chip_coordinates = []
        # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.image_size, self.stride)) + [height - self.image_size]:
            for x in list(range(0, width - self.image_size, self.stride)) + [width - self.image_size]:
                chip_coordinates.append((y, x))

        self.chip_coordinates = chip_coordinates

    def __len__(self):
        return len(self.chip_coordinates)

    def __getitem__(self, item):

        y, x = self.chip_coordinates[item]

        image_patch = self.image_np[:, y:y + self.image_size, x:x + self.image_size]

        image_ts = self.image_trans1(image_patch)

        return image_ts, np.array((y, x))


@torch.no_grad()
def val_one_epoch(args, model, epoch, test_list):
    '''
    Param:
        args.val_dir 验证集的根目录
        args.fig_path 输出路径
        args.image_size=224
        args.num_class=17
        args.stride=112

    '''
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    random.shuffle(test_list)

    auxiliary_list = [r'F:\datasets\Remote Sensing\OpenEarthMap_with_xBD\dolnoslaskie\images\dolnoslaskie_1.tif',
                      r'F:\datasets\Remote Sensing\OpenEarthMap_with_xBD\kujawsko-pomorskie\images\kujawsko-pomorskie_5.tif']
    temp_list = test_list[:200] + auxiliary_list
    pbar = tqdm(temp_list)

    for idx, image_path in enumerate(pbar):

        dataset = TestDataset(image_path, image_size=args.image_size, stride=args.stride)
        dataloader = DataLoader(dataset, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)

        with rasterio.open(image_path) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        output = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        # 权重核 中心生成的置信度高
        kernel = np.ones((args.image_size, args.image_size), dtype=np.float32)
        kernel[args.stride // 2:-args.stride // 2, args.stride // 2:-args.stride // 2] = 5

        # with calRunTimer(block_name='模型输出阶段'):
        for i, (image, coords) in enumerate(dataloader):
            pbar.set_description(f"image_idx:{idx}; batch_idx:{i}; path:{os.path.basename(image_path)}")
            image = image.to(args.device)

            # with calRunTimer(block_name='模型输出pred_cnn = model(image)'):
            with torch.no_grad():
                pred_cnn = model(image)
            logist = pred_cnn.cpu().numpy()

            # with calRunTimer(block_name='模型赋值'):
            for bs in range(logist.shape[0]):
                y, x = coords[bs]
                output[:, y:y + args.image_size, x:x + args.image_size] += logist[bs] * kernel
                counts[y:y + args.image_size, x:x + args.image_size] += kernel
        output = output / counts
        output = output.squeeze()

        output = output.argmax(axis=0).astype(np.uint8)

        # -------------------
        # Save output
        # -------------------
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        basename = os.path.basename(image_path).split('.')[0]

        mask_16cls = get_xx_label_map(f'{args.label_type}_train', args.label_type)[output]
        # TODO: cm 类型需要手动换一下
        if args.label_type == 'ESA_GLC10_label':
            color_map = {key: value['color'] for key, value in cm.color_map_ESA_GLC10.items()}
        if args.label_type == 'GLC_FCS30_label':
            color_map = {key: value['color'] for key, value in cm.color_map_FCS_GLC30.items()}
        if args.label_type == 'Esri_GLC10_label':
            color_map = {key: value['color'] for key, value in cm.color_map_Esri_GLC10.items()}

        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls.tif', "w+", **output_profile) as f:
            f.write(mask_16cls, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        mask_4cls = get_xx_label_map(f'{args.label_type}_train', 'Target_4_cls')[output]
        color_map = {key: value['color'] for key, value in cm.color_map_4_cls.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}.tif', "w", **output_profile) as f:
            f.write(mask_4cls, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        # print(f'finish save at {args.fig_path}/{basename}_epoch_{epoch}.tif')

    # for end
    return calu_metric(args, epoch)


def calu_metric(args, epoch):
    miou_record = AverageMeter()
    hist_record = AverageMeter()

    image_list = glob.glob(rf'{args.fig_path}/*_epoch_{epoch}.tif')

    for image_path in image_list:
        basename = os.path.basename(image_path).split(f'_epoch_{epoch}.tif')[0]
        # args.val_dir F:\datasets\Remote Sensing\OpenEarthMap_with_xBD
        city = re.compile(r"^(.*)_\d+").findall(basename)[0]
        label_hr_path = os.path.join(args.val_dir, rf'{city}\labels\{basename}.tif')

        label_hr = rasterio.open(label_hr_path).read().squeeze()
        label_hr = get_xx_label_map('HR_ground_truth', 'Target_4_cls')[label_hr]

        # output
        pred_hr = rasterio.open(image_path).read().squeeze()

        metric_res = evaluate(pred_hr, label_hr, num_class=5)

        miou_record.update(metric_res['miou'])
        hist_record.update(metric_res['hist'])

    class_iou = Evaluator(num_class=5).class_intersection_over_union(hist_record.avg)

    np.set_printoptions(suppress=True, precision=3)
    print(f'* 小图算miou再平均：{miou_record.avg:.1%}')
    print(f'* 整个数据集求平均：{class_iou}')
    print(f'* 去掉背景类：{class_iou[1:].sum() / 4:.1%}')
    print(f'* 不去掉背景类：{class_iou.sum() / 5:.1%}')

    return class_iou[1:].sum() / 4
