import gc
import glob
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.others.Mars import Mars
from utils import cm
from utils.metrics import Evaluator, evaluate
from utils.misc import AverageMeter


@torch.no_grad()
def val_one_epoch(args, model, epoch, val_list, return_metric=True):
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    for image_path in val_list:
        dataset = Mars([image_path], chip_size=args.image_size, stride=args.stride)
        dataloader = DataLoader(dataset, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)

        with rasterio.open(image_path) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        output = np.zeros((args.num_classes, input_height, input_width))
        counts = np.zeros((input_height, input_width))

        # 权重核 中心生成的置信度高
        kernel = np.ones((args.image_size, args.image_size))
        kernel[args.stride // 2:-args.stride // 2, args.stride // 2:-args.stride // 2] = 5

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (image, _, coords) in pbar:
            pbar.set_description(f"batch_idx:{i}; path:{os.path.basename(image_path)}")
            image = image.to(args.device)

            with torch.no_grad():
                pred_cnn = model(image)
            logist = pred_cnn.cpu().numpy()

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

        color_map = {key: value['color'] for key, value in cm.color_map_GaleCrater.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}.tif', "w+", **output_profile) as f:
            f.write(output, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

    # for end
    if return_metric:
        return calu_metric(args, epoch)


def calu_metric(args, epoch):
    miou_record = AverageMeter()
    hist_record = AverageMeter()

    image_list = glob.glob(rf'{args.fig_path}/*_epoch_{epoch}.tif')

    for image_path in image_list:
        basename = os.path.basename(image_path).split(f'_epoch_{epoch}.tif')[0]
        basename = basename.replace('img', 'gt')
        label_path = os.path.join(args.val_dir, f'{basename}.tif')

        label_hr = rasterio.open(label_path).read()

        # output
        predict = rasterio.open(image_path).read()
        predict = predict[:,:-1,:]
        metric_res = evaluate(predict, label_hr, num_class=args.num_classes)

        miou_record.update(metric_res['miou'])
        hist_record.update(metric_res['hist'])

    class_iou = Evaluator(num_class=args.num_classes).class_intersection_over_union(hist_record.avg)

    np.set_printoptions(suppress=True, precision=3)
    print(f'* 小图算miou再平均：{miou_record.avg:.1%}')
    print(f'* 整个数据集求平均：{class_iou}')
    print(f'* 去掉背景类：{class_iou[1:].sum() / 4:.1%}')
    print(f'* 不去掉背景类：{class_iou.sum() / 5:.1%}')

    return class_iou[1:].sum() / 4
