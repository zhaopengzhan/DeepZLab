import gc
import glob
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader.Chesapeake import BaseDataset
from utils import cm
from utils.label_maps import get_xx_label_map
from utils.metrics import Evaluator, evaluate
from utils.misc import AverageMeter


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
def val_one_epoch(args, model, epoch):
    '''
    Param:
        args.val_dir 验证集的根目录
        args.fig_path 输出路径
        args.image_size=224
        args.num_class=17
        args.stride=112

    '''
    model.eval()
    print(f'start visual image at {args.val_dir}')
    image_list = glob.glob(f'{args.val_dir}\*_naip-new.tif')
    # image_list = glob.glob(f'{args.val_dir}\m_4107506_se_18_1_naip-new.tif')
    # 每次训练前先清空一下内存
    gc.collect()
    torch.cuda.empty_cache()
    pbar = tqdm(image_list)
    for idx, image_path in enumerate(pbar):
        dataset = TestDataset(image_path, image_size=args.image_size, stride=args.stride)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        with rasterio.open(image_path) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        output = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        # 每个搞个单独输出
        output_cnn = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        output_fusion = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        output_trans = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        # 权重核 中心生成的置信度高
        kernel = np.ones((args.image_size, args.image_size), dtype=np.float32)
        kernel[args.stride // 2:-args.stride // 2, args.stride // 2:-args.stride // 2] = 5

        for i, (image, coords) in enumerate(dataloader):
            pbar.set_description(f"image_idx:{idx}; batch_idx:{i}; path:{os.path.basename(image_path)}")
            image = image.cuda()

            with torch.no_grad():
                pred_cnn, pred_fusion, pred_trans, *_ = model(image)
            logist = (pred_cnn + pred_fusion).cpu().numpy()

            pred_cnn, pred_fusion, pred_trans = pred_cnn.cpu().numpy(), pred_fusion.cpu().numpy(), pred_trans.cpu().numpy()

            for bs in range(logist.shape[0]):
                y, x = coords[bs]
                output[:, y:y + args.image_size, x:x + args.image_size] += logist[bs] * kernel
                #
                output_cnn[:, y:y + args.image_size, x:x + args.image_size] += pred_cnn[bs] * kernel
                output_fusion[:, y:y + args.image_size, x:x + args.image_size] += pred_fusion[bs] * kernel
                output_trans[:, y:y + args.image_size, x:x + args.image_size] += pred_trans[bs] * kernel
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

        mask_16cls = get_xx_label_map('nlcd_label_train', 'nlcd_label')[output]
        color_map = {key: value['color'] for key, value in cm.color_map_nlcd.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls.tif', "w+", **output_profile) as f:
            f.write(mask_16cls, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        mask_4cls = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[output]
        # TODO: cm 类型换一下
        color_map = {key: value['color'] for key, value in cm.color_map_4_cls.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}.tif', "w", **output_profile) as f:
            f.write(mask_4cls, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        # 单独的16cls
        output_cnn = output_cnn.squeeze().argmax(axis=0).astype(np.uint8)
        mask_cnn = get_xx_label_map('nlcd_label_train', 'nlcd_label')[output_cnn]
        color_map = {key: value['color'] for key, value in cm.color_map_nlcd.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls_cnn.tif', "w", **output_profile) as f:
            f.write(mask_cnn, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls_cnn.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        output_fusion = output_fusion.squeeze().argmax(axis=0).astype(np.uint8)
        mask_fusion = get_xx_label_map('nlcd_label_train', 'nlcd_label')[output_fusion]
        color_map = {key: value['color'] for key, value in cm.color_map_nlcd.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls_fusion.tif', "w", **output_profile) as f:
            f.write(mask_fusion, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls_fusion.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        output_trans = output_trans.squeeze().argmax(axis=0).astype(np.uint8)
        mask_trans = get_xx_label_map('nlcd_label_train', 'nlcd_label')[output_trans]
        color_map = {key: value['color'] for key, value in cm.color_map_nlcd.items()}
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls_trans.tif', "w", **output_profile) as f:
            f.write(mask_trans, 1)
        with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_16cls_trans.tif', "r+", **output_profile) as f:
            f.write_colormap(1, color_map)

        # 单独的
        # output_cnn = output_cnn.squeeze().argmax(axis=0).astype(np.uint8)
        # mask_cnn = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[output_cnn]
        # color_map = {key: value['color'] for key, value in cm.color_map_4_cls.items()}
        # with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_cnn.tif', "w", **output_profile) as f:
        #     f.write(mask_cnn, 1)
        # with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_cnn.tif', "r+", **output_profile) as f:
        #     f.write_colormap(1, color_map)
        #
        # output_fusion = output_fusion.squeeze().argmax(axis=0).astype(np.uint8)
        # mask_fusion = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[output_fusion]
        # color_map = {key: value['color'] for key, value in cm.color_map_4_cls.items()}
        # with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_fusion.tif', "w", **output_profile) as f:
        #     f.write(mask_fusion, 1)
        # with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_fusion.tif', "r+", **output_profile) as f:
        #     f.write_colormap(1, color_map)
        #
        # output_trans = output_trans.squeeze().argmax(axis=0).astype(np.uint8)
        # mask_trans = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[output_trans]
        # color_map = {key: value['color'] for key, value in cm.color_map_4_cls.items()}
        # with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_trans.tif', "w", **output_profile) as f:
        #     f.write(mask_trans, 1)
        # with rasterio.open(f'{args.fig_path}/{basename}_epoch_{epoch}_trans.tif', "r+", **output_profile) as f:
        #     f.write_colormap(1, color_map)


        # print(f'finish save at {args.fig_path}/{basename}_epoch_{epoch}.tif')
    # calu_metric_extra(args, epoch, 'cnn')
    # calu_metric_extra(args, epoch, 'fusion')
    # calu_metric_extra(args, epoch, 'trans')
    # for end
    return calu_metric(args, epoch)


def calu_metric(args, epoch):
    miou_record = AverageMeter()
    hist_record = AverageMeter()

    # image_list = glob.glob(f'{args.fig_path}/m_3707621_sw_18_1_naip-new_epoch_30_4cls.tif')
    image_list = glob.glob(rf'{args.fig_path}/*_epoch_{epoch}.tif')

    for image_path in image_list:
        basename = os.path.basename(image_path).split(f'_epoch_{epoch}.tif')[0]
        label_hr_path = rf'{args.val_dir}\{basename}.tif'.replace("naip-new", "lc")

        label_hr = rasterio.open(label_hr_path).read().squeeze()
        label_hr = get_xx_label_map('lc_label', 'Target_4_cls')[label_hr]

        pred_hr = rasterio.open(image_path).read().squeeze()

        metric_res = evaluate(pred_hr, label_hr, num_class=5)

        miou_record.update(metric_res['miou'])
        hist_record.update(metric_res['hist'])

    class_iou = Evaluator(num_class=5).class_intersection_over_union(hist_record.avg)

    np.set_printoptions(suppress=True, precision=3)
    # print(f'* 小图算miou再平均：{miou_record.avg:.1%}')
    print(f'* 整个数据集求平均：{class_iou}')
    # print(f'* 去掉背景类：{class_iou[1:].sum() / 4:.1%}')
    # print(f'* 不去掉背景类：{class_iou.sum() / 5:.1%}')

    return class_iou[1:].sum() / 4



def calu_metric_extra(args, epoch, suffix=''):
    miou_record = AverageMeter()
    hist_record = AverageMeter()

    # image_list = glob.glob(f'{args.fig_path}/m_3707621_sw_18_1_naip-new_epoch_30_4cls.tif')
    image_list = glob.glob(rf'{args.fig_path}/*_epoch_{epoch}.tif')

    for image_path in image_list:
        basename = os.path.basename(image_path).split(f'_epoch_{epoch}.tif')[0]
        label_hr_path = rf'{args.val_dir}\{basename}.tif'.replace("naip-new", "lc")

        label_hr = rasterio.open(label_hr_path).read().squeeze()
        label_hr = get_xx_label_map('lc_label', 'Target_4_cls')[label_hr]

        # 根据后缀读取路径，然后转换
        pred_lr_path = image_path.replace(".tif", f"_{suffix}.tif")
        pred_lr = rasterio.open(pred_lr_path).read().squeeze()
        # pred_hr = get_xx_label_map('nlcd_label', 'Target_4_cls')[pred_lr]

        metric_res = evaluate(pred_lr, label_hr, num_class=5)

        miou_record.update(metric_res['miou'])
        hist_record.update(metric_res['hist'])

    class_iou = Evaluator(num_class=5).class_intersection_over_union(hist_record.avg)

    np.set_printoptions(suppress=True, precision=3)
    # print(f'* 小图算miou再平均：{miou_record.avg:.1%}')
    print(suffix)
    print(f'* 整个数据集求平均：{class_iou}')
    print(f'* 去掉背景类：{class_iou[1:].sum() / 4:.1%}')
    # print(f'* 不去掉背景类：{class_iou.sum() / 5:.1%}')

    return class_iou[1:].sum() / 4

