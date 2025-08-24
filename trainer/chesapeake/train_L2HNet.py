import gc
import logging

import torch
from torch import nn
from tqdm import tqdm

from utils.label_maps import get_xx_label_map
from utils.metrics import evaluate
from utils.misc import AverageMeter, Wrapper

logger = logging.getLogger(__name__)


def train_one_epoch(args, train_dl, model, optimizer, epoch, writer):
    logger.info(f"train at epoch {epoch}, lr {optimizer.param_groups[0]['lr']:.2e}")

    # loss
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0).to(args.device)

    # record
    loss_record = Wrapper()
    loss_record.register('loss_lr', AverageMeter())

    metric_record = Wrapper()
    metric_record.register('miou_lr', AverageMeter())
    metric_record.register('miou_hr', AverageMeter())

    #
    gc.collect()
    torch.cuda.empty_cache()
    model.train()

    pbar = tqdm(train_dl)
    for batch_idx, (image_patch, label_lr_patch, label_hr_patch) in enumerate(pbar):
        pbar.set_description(f"batch_idx:{batch_idx} Epoch:{epoch} ")

        image_patch = image_patch.to(args.device)
        label_lr_patch = label_lr_patch.to(args.device)
        label_hr_patch = label_hr_patch.to(args.device)

        # run
        outputs = model(image_patch)
        loss_lr = cross_entropy_loss(outputs, label_lr_patch.to(torch.long))
        loss = loss_lr

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss record
        loss_record['loss_lr'].update(loss_lr.item(), args.batch_size)

        # 计算评价指标
        metric_record = calculate_metrics(outputs, label_lr_patch, label_hr_patch,
                                          metric_record, args)

        # set_postfix
        losses_dict = {key: f'{loss_record[key].val:.3f}({loss_record[key].avg:.3f})' for key in loss_record.keys()}

        # 创建字典用于显示
        metrics_dict = {key: f'{metric_record[key].val:.1%}({metric_record[key].avg:.1%})' for key in
                        metric_record.keys()}

        # 使用 set_postfix 显示字典中的内容
        pbar.set_postfix({**metrics_dict, **losses_dict})

        if batch_idx % args.batch_print_result == 0:
            for loss_name in loss_record.keys():
                writer.add_scalar(f"train {loss_name}", loss_record[loss_name].val,
                                  epoch * len(train_dl) + batch_idx)

    # for end
    for miou_name in metric_record.keys():
        writer.add_scalar(f"train {miou_name}_avg", metric_record[miou_name].avg, epoch)

    return metric_record
    pass


def calculate_metrics(
        outputs,
        label_lr_patch,
        label_hr_patch,
        metric_record,
        args
):
    label_lr_patch = label_lr_patch.cpu().numpy()
    label_hr_patch = label_hr_patch.cpu().numpy()

    # lr
    mask = (outputs).argmax(axis=1).cpu().detach().numpy()
    metric = evaluate(mask, label_lr_patch, num_class=args.num_classes)
    metric_record['miou_lr'].update(metric['miou'], args.batch_size)

    # hr
    mask = (outputs).argmax(axis=1).cpu().detach().numpy()
    mask = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[mask]
    Target_4_cls = get_xx_label_map('lc_label', 'Target_4_cls')[label_hr_patch]
    metric = evaluate(mask, Target_4_cls, num_class=5)
    metric_record['miou_hr'].update(metric['miou'], args.batch_size)

    return metric_record
