import glob
import logging
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader import build_dataloader
from utils import cm
from utils.label_maps import get_xx_label_map
from utils.metrics import Evaluator
from utils.metrics import evaluate
from utils.misc import AverageMeter

logger = logging.getLogger(__name__)


@torch.no_grad()
def val_one_epoch(args, model, epoch, val_list):
    model.eval()
    logger.info(f'start val at {args.val_dir}')

    pbar = tqdm(val_list)
    for idx, image_path in enumerate(pbar):
        val_ds = build_dataloader(
            args.val_data,
            image_path=image_path,
            image_size=args.image_size,
            stride=args.stride
        )
        val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=4, pin_memory=True)
        pbar.reset(total=len(val_list) * len(val_dl))

        with rasterio.open(image_path) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()
            READ_FORBIDDEN = {"blockxsize", "blockysize", "compress", "interleave", "tiled"}
            input_profile = {k: v for k, v in input_profile.items() if k.lower() not in READ_FORBIDDEN}

        output = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        # 权重核 中心生成的置信度高
        kernel = np.ones((args.image_size, args.image_size), dtype=np.float32)
        kernel[args.stride // 2:-args.stride // 2, args.stride // 2:-args.stride // 2] = 5

        # inference
        for i, (image, coords) in enumerate(val_dl):
            pbar.update(1)
            pbar.set_postfix(idx=idx, path=os.path.basename(image_path))
            image = image.to(args.device)

            with torch.no_grad():
                pred_cnn = model(image)
                logist = (pred_cnn).cpu().numpy()

            for bs in range(logist.shape[0]):
                y, x = coords[bs]
                output[:, y:y + args.image_size, x:x + args.image_size] += logist[bs] * kernel
                counts[y:y + args.image_size, x:x + args.image_size] += kernel

        output = output / counts
        output = output.squeeze()
        output = output.argmax(axis=0).astype(np.uint8)

        # profile output
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        basename = os.path.basename(image_path).split('.')[0]

        mask_16cls = get_xx_label_map('nlcd_label_train', 'nlcd_label')[output]
        color_map = {key: value['color'] for key, value in cm.color_map_nlcd.items()}
        output_path = os.path.join(args.fig_path, f'{basename}_epoch_{epoch}_16cls.tif')
        with rasterio.open(output_path, "w+", **output_profile) as f:
            f.write(mask_16cls, 1)
        with rasterio.open(output_path, "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        mask_4cls = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[output]
        color_map = {key: value['color'] for key, value in cm.color_map_4_cls.items()}
        output_path = os.path.join(args.fig_path, f'{basename}_epoch_{epoch}.tif')
        with rasterio.open(output_path, "w", **output_profile) as f:
            f.write(mask_4cls, 1)
        with rasterio.open(output_path, "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

    pbar.close()
    return calu_metric(args, epoch)


def calu_metric(args, epoch):
    miou_record = AverageMeter()
    hist_record = AverageMeter()

    image_list = glob.glob(rf'{args.fig_path}/*_epoch_{epoch}.tif')

    for image_path in image_list:
        filename = os.path.basename(image_path).split(f'_epoch_{epoch}.tif')[0]

        label_hr_path = os.path.join(args.val_dir, f'{filename}.tif').replace("naip-new", "lc")
        label_hr = rasterio.open(label_hr_path).read().squeeze()
        label_hr = get_xx_label_map('lc_label', 'Target_4_cls')[label_hr]

        pred_hr = rasterio.open(image_path).read().squeeze()

        metric_res = evaluate(pred_hr, label_hr, num_class=5)

        miou_record.update(metric_res['miou'])
        hist_record.update(metric_res['hist'])

    class_iou = Evaluator(num_class=5).class_intersection_over_union(hist_record.avg)

    np.set_printoptions(suppress=True, precision=3)
    print(f'* 小图算miou再平均：{miou_record.avg:.1%}')
    logger.info(f'* 整个数据集求平均：{class_iou}')
    print(f'* 去掉背景类：{class_iou[1:].mean():.1%}')
    print(f'* 不去掉背景类：{class_iou.mean():.1%}')

    return class_iou[1:].mean()
