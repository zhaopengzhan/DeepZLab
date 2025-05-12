import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.label_maps import get_xx_label_map
from utils.metrics import evaluate
from utils.misc import AverageMeter, Wrapper


def train_one_epoch(args, train_dl, model, optimizer, epoch, writer, loss_wrapper, mode='train'):
    print(f'{mode} epoch %d, lr %.2e' % (epoch, optimizer.param_groups[0]['lr']))

    loss_record = Wrapper()
    if mode == 'train':
        loss_record.register('loss_lr', AverageMeter())
        # loss_record.register('loss_trans', AverageMeter())
        # loss_record.register('loss_plat', AverageMeter())
        # loss_record.register('loss_sum', AverageMeter())
        # loss_record.register('loss_hr', AverageMeter())
        # loss_record.register('loss_dice', AverageMeter())
        # loss_record.register('loss_focal', AverageMeter())

    metric_record = Wrapper()
    metric_record.register('miou_merge', AverageMeter())
    # metric_record.register('miou_cnn', AverageMeter())
    # metric_record.register('miou_trans', AverageMeter())
    # metric_record.register('miou_plat', AverageMeter())
    # metric_record.register('miou_fusion', AverageMeter())
    # metric_record.register('miou_bs', AverageMeter())
    # metric_record.register('miou_merge3', AverageMeter())

    if mode == 'train':
        model.train()
    else:
        model.eval()

    # 每次训练前先清空一下内存
    gc.collect()
    torch.cuda.empty_cache()

    pbar = tqdm(train_dl)
    for batch_idx, (image_patch, label_lr_patch, label_hr_patch) in enumerate(pbar):
        pbar.set_description(f"batch_idx:{batch_idx} Epoch:{epoch} ")

        image_patch = image_patch.to(args.device)
        label_lr_patch = label_lr_patch.to(args.device)
        label_hr_patch = label_hr_patch.to(args.device)

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
            # TODO: 报错梯度重复计算，原因是输出为nan
            loss_lr.backward()
            optimizer.step()

            # loss record
            loss_record['loss_lr'].update(loss_lr.item(), args.batch_size)

            label_lr_patch = label_lr_patch.cpu()
            label_hr_patch = label_hr_patch.cpu()

        # 计算评价指标
        metric_record = calculate_metrics(pred_cnn, label_lr_patch, label_hr_patch,
                                          metric_record, args)

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
        label_hr_patch,
        metric_record,
        args
):
    # 计算 trans 模型的指标
    # mask_trans = pred_trans.argmax(axis=1).cpu().detach().numpy()
    # mask_trans = get_xx_label_map('nlcd_label_train','Target_4_cls')[mask_trans]
    # metric_trans = evaluate(mask_trans, label_hr_patch.numpy(), num_class=5)
    # metric_record['miou_trans'].update(metric_trans['miou'], args.batch_size)

    # 计算 cnn 模型的指标
    # mask_cnn = pred_cnn.argmax(axis=1).cpu().detach().numpy()
    # mask_cnn = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[mask_cnn]
    # metric_cnn = evaluate(mask_cnn, label_hr_patch.numpy(), num_class=5)
    # metric_record['miou_cnn'].update(metric_cnn['miou'], args.batch_size)

    # 计算平台模型的指标
    # mask_plat = mask_plat.cpu().detach().numpy()
    # mask_plat = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[mask_plat]
    # metric_plat = evaluate(mask_plat, label_hr_patch.numpy(), num_class=5)
    # metric_record['miou_plat'].update(metric_plat['miou'], args.batch_size)

    # 计算融合模型的指标
    mask_merge = (pred_lseg).argmax(axis=1).cpu().detach().numpy()
    mask_merge = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[mask_merge]
    metric_merge = evaluate(mask_merge, label_hr_patch.numpy(), num_class=5)
    metric_record['miou_merge'].update(metric_merge['miou'], args.batch_size)

    # 计算基准指标
    # label_np = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[label_lr_patch.numpy()]
    # metric_bs = evaluate(label_np, label_hr_patch.numpy(), num_class=5)
    # metric_record['miou_bs'].update(metric_bs['miou'], args.batch_size)

    # 计算融合模型的指标
    # mask_merge = (pred_fusion).argmax(axis=1).cpu().detach().numpy()
    # mask_merge = get_xx_label_map('nlcd_label_train', 'Target_4_cls')[mask_merge]
    # metric_merge = evaluate(mask_merge, label_hr_patch.numpy(), num_class=5)
    # metric_record['miou_fusion'].update(metric_merge['miou'], args.batch_size)

    # 计算融合模型的指标
    # mask_merge = (pred_trans + pred_cnn + pred_fusion).argmax(axis=1).cpu().detach().numpy()
    # mask_merge = get_xx_label_map('nlcd_label_train','Target_4_cls')[mask_merge]
    # metric_merge = evaluate(mask_merge, label_hr_patch.numpy(), num_class=5)
    # metric_record['miou_merge3'].update(metric_merge['miou'], args.batch_size)

    return metric_record
