import gc

import torch
from tqdm import tqdm

from utils.label_maps import get_xx_label_map
from utils.metrics import evaluate
from utils.misc import AverageMeter, Wrapper


def train_one_epoch(args, train_dl, model, optimizer, epoch, writer, loss_wrapper, mode='train'):
    print(f'{mode} epoch %d, lr %.2e' % (epoch, optimizer.param_groups[0]['lr']))

    loss_record = Wrapper()
    if mode == 'train':
        loss_record.register('loss_lr', AverageMeter())

    metric_record = Wrapper()
    metric_record.register('miou_lr', AverageMeter())

    if mode == 'train':
        model.train()
    else:
        model.eval()

    # 每次训练前先清空一下内存
    gc.collect()
    torch.cuda.empty_cache()

    pbar = tqdm(train_dl)
    for batch_idx, (image_patch, label_lr_patch) in enumerate(pbar):
        pbar.set_description(f"batch_idx:{batch_idx} Epoch:{epoch} ")

        image_patch = image_patch.to(args.device)
        label_lr_patch = label_lr_patch.to(args.device)

        # run
        if mode == 'train':
            pred_cnn = model(image_patch)
        else:
            with torch.no_grad():
                pred_cnn = model(image_patch)

        if mode == 'train':
            loss_lr = loss_wrapper['ce_loss'](pred_cnn, label_lr_patch.to(torch.long))

            # 向后传播
            optimizer.zero_grad()
            loss_lr.backward()
            optimizer.step()

            # loss record
            loss_record['loss_lr'].update(loss_lr.item(), args.batch_size)

            label_lr_patch = label_lr_patch.cpu()

        # 计算评价指标
        metric_record = calculate_metrics(pred_cnn, label_lr_patch, metric_record, args)

        # set_postfix
        losses_dict = {}
        if mode == 'train':
            losses_dict = {key: f'{loss_record[key].val:.3f}({loss_record[key].avg:.3f})' for key in loss_record.keys()}

        # 创建字典用于显示
        metrics_dict = {key: f'{metric_record[key].val:.1%}({metric_record[key].avg:.1%})' for key in
                        metric_record.keys()}

        # 使用 set_postfix 显示字典中的内容
        pbar.set_postfix({**metrics_dict, **losses_dict})

        if mode == 'train' and batch_idx % args.epoch_print_result == 0:
            for loss_name in loss_record.keys():
                writer.add_scalar(f"{mode} {loss_name}", loss_record[loss_name].val,
                                  epoch * len(train_dl) + batch_idx)

    # train end
    for miou_name in metric_record.keys():
        writer.add_scalar(f"{mode} {miou_name}_avg", metric_record[miou_name].avg, epoch)

    return metric_record
    pass


def calculate_metrics(
        pred_lseg,
        label_lr_patch,
        metric_record,
        args
):
    # 计算LR Label的指标
    mask_merge = pred_lseg.argmax(axis=1).cpu().detach().numpy()
    # mask_merge = get_xx_label_map(f'{args.label_type}_train', args.label_type)[mask_merge]
    metric_merge = evaluate(mask_merge, label_lr_patch.numpy(), num_class=args.num_classes)
    metric_record['miou_lr'].update(metric_merge['miou'], args.batch_size)

    return metric_record
