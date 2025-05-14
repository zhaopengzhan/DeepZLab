import gc

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.metrics import evaluate
from utils.misc import AverageMeter, Wrapper


def train_one_epoch(args, train_dl, model, optimizer, epoch, writer, mode='train'):
    print(f'{mode} epoch %d, lr %.2e' % (epoch, optimizer.param_groups[0]['lr']))
    model.freeze_parameter(epoch<25)
    # loss
    # weights = torch.tensor([0.01, 1., 1., 1., 1., 1., 1., 1., 1.]).to(args.device)
    cross_entropy_loss = nn.CrossEntropyLoss().to(args.device)

    loss_record = Wrapper()
    if mode == 'train':
        loss_record.register('loss_ce', AverageMeter())

    metric_record = Wrapper()
    metric_record.register('miou', AverageMeter())
    metric_record.register('oa', AverageMeter())
    metric_record.register('iou', AverageMeter())

    if mode == 'train':
        model.train()
    else:
        model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    pbar = tqdm(train_dl)
    for batch_idx, (image_patch, label_patch) in enumerate(pbar):
        pbar.set_description(f"batch_idx:{batch_idx} Epoch:{epoch} ")

        image_patch = image_patch.to(args.device)
        label_patch = label_patch.to(args.device)

        # run
        if mode == 'train':
            pred_cnn = model(image_patch)
        else:
            with torch.no_grad():
                pred_cnn = model(image_patch)

        if mode == 'train':
            loss_lr = cross_entropy_loss(pred_cnn, label_patch)

            # backward
            optimizer.zero_grad()
            loss_lr.backward()
            optimizer.step()

            # loss record
            loss_record['loss_ce'].update(loss_lr.item(), args.batch_size)

        # 计算评价指标
        metric_record = calculate_metrics(args, pred_cnn, label_patch, metric_record)

        # set_postfix
        losses_dict = {}
        if mode == 'train':
            losses_dict = {key: f'{loss_record[key].val:.3f}({loss_record[key].avg:.3f})' for key in loss_record.keys()}

        # 创建字典用于显示
        metrics_dict = {
            key: f'{safe_percent(metric_record[key].val)}({safe_percent(metric_record[key].avg)})'
            for key in metric_record.keys()
        }

        # 使用 set_postfix 显示字典中的内容
        pbar.set_postfix({**metrics_dict, **losses_dict})

        if mode == 'train' and batch_idx % args.epoch_print_result == 0:
            for loss_name in loss_record.keys():
                writer.add_scalar(f"{mode} {loss_name}", loss_record[loss_name].val,
                                  epoch * len(train_dl) + batch_idx)

    # train end
    for miou_name in metric_record.keys():
        value = metric_record[miou_name].avg
        if not (isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and np.size(value) > 1):
            writer.add_scalar(f"{mode} {miou_name}_avg", metric_record[miou_name].avg, epoch)

    return metric_record
    pass


def calculate_metrics(
        args,
        logist,
        label_patch,
        metric_record

):
    label_patch = label_patch.cpu().detach().numpy()
    mask = logist.argmax(axis=1).cpu().detach().numpy()

    metric_merge = evaluate(mask, label_patch, num_class=args.num_classes)
    metric_record['miou'].update(metric_merge['miou'], args.batch_size)
    metric_record['oa'].update(metric_merge['pixel_accuracy'], args.batch_size)
    metric_record['iou'].update(metric_merge['class_iou'], args.batch_size)

    return metric_record

def safe_percent(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return f"{x.item():.1%}"
        else:
            return ", ".join(f"{v:.1%}" for v in x)
    return f"{x:.1%}"
