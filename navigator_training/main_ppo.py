import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'models'))

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optim_factory import create_optimizer

from dataset_utils import build_dataset
import models.dynamic_deitS_global_scale_single
from engine_ppo_global_scale_single import train_one_epoch, evaluate

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler




def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('AdaptivePath Navigator training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')


    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--spatial_hflip', default=0.5, type=float)
    parser.add_argument('--spatial_vflip', default=0.5, type=float)
    parser.add_argument('--spatial_affine_prob', default=0.8, type=float)
    parser.add_argument('--spatial_degrees', default=15.0, type=float)
    parser.add_argument('--spatial_translate', default=0.12, type=float)
    parser.add_argument('--spatial_scale_min', default=0.90, type=float)
    parser.add_argument('--spatial_scale_max', default=1.10, type=float)
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', required=True, type=str,
                        help='ImageFolder root containing train and val')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    ##### AdaptivePath Navigator configuration #####
    parser.add_argument('--feature_in_chans', default=384, type=int)

    parser.add_argument('--policy_net_hidden_chans', default=128, type=int)
    parser.add_argument('--policy_net_kernel_size', default=3, type=int)

    
    parser.add_argument('--recover_n', default=5, type=int)
    parser.add_argument('--remaining_blocks', default=4, type=int)

    parser.add_argument('--glance_input_size', default=112, type=int)
    parser.add_argument('--glance_net_depth', default=12, type=int)
    parser.add_argument('--glance_net_mlp_ratio', default=4, type=int)
    parser.add_argument('--glance_net_drop_path', default=0.1, type=float)

    parser.add_argument('--focus_patch_size', default=112, type=int)
    parser.add_argument('--focus_scale_min', default=0.25, type=float)
    parser.add_argument('--focus_scale_max', default=1.0, type=float)
    parser.add_argument('--focus_net_reg_size', default=160, type=int)
    parser.add_argument('--focus_net_depth', default=12, type=int)
    parser.add_argument('--focus_net_mlp_ratio', default=4, type=int)
    parser.add_argument('--focus_net_drop_path', default=0.1, type=float)

    parser.add_argument('--multi_cls_drop_path', default=0.1, type=float)

    #####  Loss weight & KD  #####
    
    parser.add_argument('--loss_reg_focus_net_weight', default=1.0, type=float)

    parser.add_argument('--kd_temp', default=1.0, type=float)
    parser.add_argument('--kd_alpha', default=1.0, type=float)

    #####  lr  #####
    parser.add_argument('--backbone_lr_scale', default=1.0, type=float)
    parser.add_argument('--stn_lr_ratio', default=1.0, type=float)

    #####  RL  #####
    parser.add_argument('--seq_l', default=4, type=int)  
    parser.add_argument('--gamma', default=0.7, type=float)  
    parser.add_argument('--ppo_update_steps', default=2, type=int)  
    parser.add_argument('--update_policy_freq', default=1, type=int)  
    parser.add_argument('--eps_clip', default=0.2, type=float)  

    parser.add_argument('--ppo_std_start', type=float, default=0.25)
    parser.add_argument('--ppo_std_end', type=float, default=0.10)
    parser.add_argument('--ppo_std_decay', type=str, choices=['linear', 'cosine'], default='cosine')

    parser.add_argument('--ppo_clip_grid', type=float, default=None)
    parser.add_argument('--ppo_lam', default=1.0, type=float)  
    parser.add_argument('--adv_normalization', default=1, type=int)  
    parser.add_argument('--ppo_entropy_coef', default=0.01, type=float)
    parser.add_argument('--ppo_feature_policy_coef', default=0.01, type=float)
    parser.add_argument('--ppo_feature_value_coef', default=0.01, type=float)
    parser.add_argument('--focus_consistency_weight', default=0.0, type=float,
                        help='Weight for evidence-distribution focus consistency updates.')
    parser.add_argument('--focus_consistency_every_n_steps', default=10, type=int,
                        help='Run focus consistency update every N optimizer steps.')
    parser.add_argument('--focus_consistency_max_batch', default=8, type=int,
                        help='Max samples used by the extra focus consistency forward pass.')
    parser.add_argument('--focus_consistency_heatmap_size', default=32, type=int,
                        help='Spatial size of the differentiable evidence heatmap.')
    parser.add_argument('--focus_consistency_sigma', default=0.10, type=float,
                        help='Base Gaussian sigma, in normalized coordinates, for focus heatmaps.')
    parser.add_argument('--focus_consistency_geo_weight', default=1.0, type=float,
                        help='Weight for geometric equivariance consistency.')
    parser.add_argument('--focus_consistency_step_geo_weight', default=0.5, type=float,
                        help='Weight for step-wise geometric consistency in addition to set-level consistency.')
    parser.add_argument('--focus_consistency_app_weight', default=0.5, type=float,
                        help='Weight for appearance/stain robustness consistency.')
    parser.add_argument('--focus_consistency_scale_weight', default=0.2, type=float,
                        help='Weight for focus scale consistency.')
    parser.add_argument('--focus_consistency_hflip', default=0.5, type=float)
    parser.add_argument('--focus_consistency_vflip', default=0.5, type=float)
    parser.add_argument('--focus_consistency_rotate90', default=1.0, type=float,
                        help='Probability of applying a random 90-degree rotation in the geometry branch.')
    parser.add_argument('--focus_consistency_app_brightness', default=0.15, type=float)
    parser.add_argument('--focus_consistency_app_contrast', default=0.15, type=float)
    parser.add_argument('--focus_consistency_app_saturation', default=0.10, type=float)

    parser.add_argument('--num_ppo_update_iters', default=1, type=int)  
    return parser




def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    # ----------------------------
    # val sampler 保持不变
    # ----------------------------
    if args.dist_eval and dataset_val is not None:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # ----------------------------
    # NEW: DataLoader(train)
    # 注意：sampler!=None 时不要再设置 shuffle=True
    # ----------------------------
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=False,
        persistent_workers=(args.num_workers > 0),
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    args.model = 'dynamic_deitS'
    extra_model_kwargs = {
        "focus_scale_min": args.focus_scale_min,
        "focus_scale_max": args.focus_scale_max,
    }

    model = models.dynamic_deitS_global_scale_single.dynamic_deitS( 

        # dynamic model parameters
        seq_len                 =  args.seq_l,
        feature_in_chans        =  args.feature_in_chans,

        # policy_patch parameters
        policy_net_hidden_chans =  args.policy_net_hidden_chans, 
        policy_net_kernel_size  =  args.policy_net_kernel_size,

        # policy_recover parameters
        recover_n               =  args.recover_n,
        remaining_blocks        =  args.remaining_blocks,

        ## glance_net
        # glance_input_size, glance_net_depth, glance_net_mlp_ratio, glance_net_drop_path,
        glance_input_size     =  args.glance_input_size,
        glance_net_depth      =  args.glance_net_depth,
        glance_net_mlp_ratio  =  args.glance_net_mlp_ratio,
        glance_net_drop_path  =  args.glance_net_drop_path,

        ## focus_net
        # focus_patch_size, focus_net_reg_size, focus_net_depth, focus_net_mlp_ratio, focus_net_drop_path, 
        focus_patch_size      =  args.focus_patch_size,
        focus_net_reg_size    =  args.focus_net_reg_size,
        focus_net_depth       =  args.focus_net_depth,
        focus_net_mlp_ratio   =  args.focus_net_mlp_ratio,
        focus_net_drop_path   =  args.focus_net_drop_path,

        ## multi_cls
        # multi_cls_drop_path
        multi_cls_drop_path = args.multi_cls_drop_path,

        # deit parameters
        num_classes=args.nb_classes,
        **extra_model_kwargs,
    )


    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        if "model" not in checkpoint:
            raise KeyError("Phase I checkpoint must contain a model state_dict")
        print("Load ckpt from %s" % args.finetune)
        utils.load_state_dict(model, checkpoint["model"], prefix="")
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if utils.is_main_process():
        print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        backbone_lr_scale=args.backbone_lr_scale)

    loss_scaler = NativeScaler()

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, args=args)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    max_accuracy_glance = 0.0


    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, args=args
        )

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch
                )

        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, args=args)
            main_metric_name = "Main-path"

            # ===== 主路径 =====
            print(f"{main_metric_name} Accuracy on {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best"
                    )
            print(f"{main_metric_name} Max accuracy: {max_accuracy:.2f}%")

            # ===== Glance =====
            print(f"Glance Accuracy on {len(dataset_val)} test images: {test_stats['acc1_glance']:.1f}%")
            if max_accuracy_glance < test_stats["acc1_glance"]:
                max_accuracy_glance = test_stats["acc1_glance"]
            print(f"Glance Max accuracy: {max_accuracy_glance:.2f}%")

            if log_writer is not None:
                # 主路径
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

                # glance
                log_writer.update(test_acc1_glance=test_stats['acc1_glance'], head="perf", step=epoch)
                log_writer.update(test_loss_glance=test_stats['loss_glance'], head="perf", step=epoch)


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AdaptivePath Navigator training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    

    main(args)
