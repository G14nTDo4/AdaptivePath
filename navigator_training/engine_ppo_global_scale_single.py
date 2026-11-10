import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import torch.nn.functional as F

import utils



_IMAGE_NET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_IMAGE_NET_DEFAULT_STD = (0.229, 0.224, 0.225)
_IMAGE_NET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
_IMAGE_NET_INCEPTION_STD = (0.5, 0.5, 0.5)


def _image_mean_std(args, device, dtype):
    if getattr(args, 'imagenet_default_mean_and_std', True):
        mean, std = _IMAGE_NET_DEFAULT_MEAN, _IMAGE_NET_DEFAULT_STD
    else:
        mean, std = _IMAGE_NET_INCEPTION_MEAN, _IMAGE_NET_INCEPTION_STD
    mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _normalize_distribution(x, eps=1e-8):
    x = x.reshape(x.shape[0], -1)
    return x / x.sum(dim=1, keepdim=True).clamp_min(eps)


def _js_divergence(p, q, eps=1e-8):
    p = _normalize_distribution(p, eps=eps).clamp_min(eps)
    q = _normalize_distribution(q, eps=eps).clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - m.log())).sum(dim=1)
        + (q * (q.log() - m.log())).sum(dim=1)
    ).mean()


def _stack_actions_and_scales(action_locs, args):
    actions = torch.stack(action_locs, dim=1).float()
    scale_actions = actions[:, :, 2].clamp(0.0, 1.0)
    patch_scales = (
        float(getattr(args, 'focus_scale_min', 0.25))
        + scale_actions * (
            float(getattr(args, 'focus_scale_max', 1.0))
            - float(getattr(args, 'focus_scale_min', 0.25))
        )
    )
    return actions, patch_scales


def _actions_to_crop_centers(action_locs, args):
    actions, patch_scales = _stack_actions_and_scales(action_locs, args)
    crop_side = float(getattr(args, 'focus_patch_size', 112)) / float(getattr(args, 'input_size', 512))
    crop_side = crop_side * patch_scales

    top_left_y = actions[:, :, 0].clamp(0.0, 1.0) * (1.0 - crop_side)
    top_left_x = actions[:, :, 1].clamp(0.0, 1.0) * (1.0 - crop_side)
    centers_y = (top_left_y + 0.5 * crop_side).clamp(0.0, 1.0)
    centers_x = (top_left_x + 0.5 * crop_side).clamp(0.0, 1.0)
    centers = torch.stack([centers_y, centers_x], dim=-1)
    return centers, patch_scales


def _centers_to_evidence_heatmaps(centers, patch_scales, args):
    bsz, num_steps, _ = centers.shape
    heatmap_size = int(getattr(args, 'focus_consistency_heatmap_size', 32))
    base_sigma = float(getattr(args, 'focus_consistency_sigma', 0.10))
    device = centers.device
    dtype = centers.dtype

    coords = torch.linspace(0.0, 1.0, heatmap_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    grid_y = grid_y.view(1, 1, heatmap_size, heatmap_size)
    grid_x = grid_x.view(1, 1, heatmap_size, heatmap_size)

    centers_y = centers[:, :, 0].view(bsz, num_steps, 1, 1)
    centers_x = centers[:, :, 1].view(bsz, num_steps, 1, 1)
    sigma = (base_sigma * patch_scales).clamp_min(1.0 / heatmap_size).view(bsz, num_steps, 1, 1)
    dist2 = (grid_y - centers_y).pow(2) + (grid_x - centers_x).pow(2)
    step_heatmaps = torch.exp(-0.5 * dist2 / sigma.pow(2))
    step_heatmaps = step_heatmaps / step_heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    total_heatmap = step_heatmaps.sum(dim=1)
    total_heatmap = total_heatmap / total_heatmap.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return step_heatmaps, total_heatmap, patch_scales


def _actions_to_evidence_heatmaps(action_locs, args):
    centers, patch_scales = _actions_to_crop_centers(action_locs, args)
    return _centers_to_evidence_heatmaps(centers, patch_scales, args)


def _apply_random_geo_aug(samples, args):
    meta = {'hflip': False, 'vflip': False, 'rot_k': 0}
    augmented = samples
    if torch.rand((), device=samples.device).item() < float(getattr(args, 'focus_consistency_hflip', 0.5)):
        augmented = torch.flip(augmented, dims=(-1,))
        meta['hflip'] = True
    if torch.rand((), device=samples.device).item() < float(getattr(args, 'focus_consistency_vflip', 0.5)):
        augmented = torch.flip(augmented, dims=(-2,))
        meta['vflip'] = True
    if torch.rand((), device=samples.device).item() < float(getattr(args, 'focus_consistency_rotate90', 1.0)):
        rot_k = int(torch.randint(0, 4, (), device=samples.device).item())
        if rot_k:
            augmented = torch.rot90(augmented, k=rot_k, dims=(-2, -1))
        meta['rot_k'] = rot_k
    return augmented.contiguous(), meta


def _invert_geo_centers(centers, meta):
    y = centers[:, :, 0]
    x = centers[:, :, 1]

    rot_k = int(meta.get('rot_k', 0)) % 4
    if rot_k == 1:
        y, x = x, 1.0 - y
    elif rot_k == 2:
        y, x = 1.0 - y, 1.0 - x
    elif rot_k == 3:
        y, x = 1.0 - x, y

    if meta.get('vflip', False):
        y = 1.0 - y
    if meta.get('hflip', False):
        x = 1.0 - x

    return torch.stack([y.clamp(0.0, 1.0), x.clamp(0.0, 1.0)], dim=-1)


def _apply_mild_appearance_aug(samples, args):
    mean, std = _image_mean_std(args, samples.device, samples.dtype)
    rgb = (samples * std + mean).clamp(0.0, 1.0)
    bsz = rgb.shape[0]

    brightness = float(getattr(args, 'focus_consistency_app_brightness', 0.15))
    if brightness > 0:
        factor = torch.empty((bsz, 1, 1, 1), device=rgb.device, dtype=rgb.dtype).uniform_(
            1.0 - brightness, 1.0 + brightness
        )
        rgb = rgb * factor

    contrast = float(getattr(args, 'focus_consistency_app_contrast', 0.15))
    if contrast > 0:
        factor = torch.empty((bsz, 1, 1, 1), device=rgb.device, dtype=rgb.dtype).uniform_(
            1.0 - contrast, 1.0 + contrast
        )
        rgb_mean = rgb.mean(dim=(-2, -1), keepdim=True)
        rgb = (rgb - rgb_mean) * factor + rgb_mean

    saturation = float(getattr(args, 'focus_consistency_app_saturation', 0.10))
    if saturation > 0:
        factor = torch.empty((bsz, 1, 1, 1), device=rgb.device, dtype=rgb.dtype).uniform_(
            1.0 - saturation, 1.0 + saturation
        )
        gray = (0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3])
        rgb = (rgb - gray) * factor + gray

    rgb = rgb.clamp(0.0, 1.0)
    return ((rgb - mean) / std).contiguous()


def _focus_consistency_enabled(args):
    return float(getattr(args, 'focus_consistency_weight', 0.0)) > 0.0


def _should_run_focus_consistency(args, data_iter_step, update_freq):
    if not _focus_consistency_enabled(args):
        return False
    every_n = int(getattr(args, 'focus_consistency_every_n_steps', 10))
    if every_n <= 0:
        return False
    if (data_iter_step + 1) % update_freq != 0:
        return False
    return (data_iter_step + 1) % every_n == 0


def compute_focus_consistency_loss(model, samples, args):
    max_batch = int(getattr(args, 'focus_consistency_max_batch', 8))
    if max_batch > 0 and samples.shape[0] > max_batch:
        samples = samples[:max_batch]

    was_training = model.training
    model.eval()
    try:
        original_outputs = model(
            samples,
            seq_l=args.seq_l,
            force_deterministic_actions=True,
            detach_action_locs=False,
        )
        original_steps, original_total, original_scales = _actions_to_evidence_heatmaps(
            original_outputs['action_locs'], args
        )

        geo_samples, geo_meta = _apply_random_geo_aug(samples, args)
        geo_outputs = model(
            geo_samples,
            seq_l=args.seq_l,
            force_deterministic_actions=True,
            detach_action_locs=False,
        )
        geo_centers, geo_scales = _actions_to_crop_centers(geo_outputs['action_locs'], args)
        geo_centers = _invert_geo_centers(geo_centers, geo_meta)
        geo_steps, geo_total, geo_scales = _centers_to_evidence_heatmaps(geo_centers, geo_scales, args)

        geo_set_loss = _js_divergence(original_total, geo_total)
        geo_step_loss = _js_divergence(
            original_steps.reshape(-1, original_steps.shape[-2], original_steps.shape[-1]),
            geo_steps.reshape(-1, geo_steps.shape[-2], geo_steps.shape[-1]),
        )
        scale_loss = F.l1_loss(original_scales, geo_scales)

        device = samples.device
        total_loss = torch.zeros((), device=device, dtype=samples.dtype)
        total_loss = total_loss + float(getattr(args, 'focus_consistency_geo_weight', 1.0)) * geo_set_loss
        total_loss = total_loss + float(getattr(args, 'focus_consistency_step_geo_weight', 0.5)) * geo_step_loss
        total_loss = total_loss + float(getattr(args, 'focus_consistency_scale_weight', 0.2)) * scale_loss

        app_loss = torch.zeros((), device=device, dtype=samples.dtype)
        if float(getattr(args, 'focus_consistency_app_weight', 0.5)) > 0:
            app_samples = _apply_mild_appearance_aug(samples, args)
            app_outputs = model(
                app_samples,
                seq_l=args.seq_l,
                force_deterministic_actions=True,
                detach_action_locs=False,
            )
            _, app_total, _ = _actions_to_evidence_heatmaps(app_outputs['action_locs'], args)
            app_loss = _js_divergence(original_total, app_total)
            total_loss = total_loss + float(getattr(args, 'focus_consistency_app_weight', 0.5)) * app_loss

        stats = {
            'focus_consistency_raw': total_loss.detach(),
            'focus_consistency_geo': geo_set_loss.detach(),
            'focus_consistency_geo_step': geo_step_loss.detach(),
            'focus_consistency_app': app_loss.detach(),
            'focus_consistency_scale': scale_loss.detach(),
        }
        return total_loss, stats
    finally:
        model.train(was_training)


def build_ppo_rollout(ppo_update_collect_dict, update_policy_freq):
    expected_outputs_ppo = {
        'x_glance': [],
        'x_focus': [],
        'actions': [],
        'actions_logprobs': [],
        '_state_values': [],
        'policy_features_preact': [],
        'value_features_preact': [],
        'states': [],
    }

    expected_outputs_ppo['x_glance'].append(
        torch.cat(
            [ppo_update_collect_dict[m]['x_glance'][0] for m in range(update_policy_freq)],
            dim=0
        )
    )

    for key in [
        'x_focus', 'actions', 'actions_logprobs', '_state_values',
        'policy_features_preact', 'value_features_preact', 'states'
    ]:
        for step_idx in range(len(ppo_update_collect_dict[0][key])):
            expected_outputs_ppo[key].append(
                torch.cat(
                    [ppo_update_collect_dict[m][key][step_idx] for m in range(update_policy_freq)],
                    dim=0
                )
            )

    return expected_outputs_ppo


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ppo_std_this_iter', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for focus_step_index in range(args.seq_l):
        metric_logger.add_meter(f'pos_mean_x_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_mean_y_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_std_x_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_std_y_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'scale_action_mean_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'scale_action_std_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'patch_scale_mean_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'patch_scale_std_s{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    mse_loss = torch.nn.MSELoss()

    optimizer.zero_grad()
    ppo_update_collect_dict = []
    targets_collect_dict = []

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if args.ppo_std_decay == 'cosine':
            if epoch < args.warmup_epochs:
                ppo_std_this_iter = args.ppo_std_start
            else:
                ppo_std_this_iter = args.ppo_std_end + (args.ppo_std_start - args.ppo_std_end) * lr_schedule_values[it] / args.lr
        elif args.ppo_std_decay == 'linear':
            ppo_std_this_iter = args.ppo_std_end + (args.ppo_std_start - args.ppo_std_end) * (1 - it / len(lr_schedule_values))
        else:
            raise NotImplementedError

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if _focus_consistency_enabled(args):
            policy_consistency_samples = samples.detach().clone()
        else:
            policy_consistency_samples = samples

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            expected_outputs = model(samples, seq_l=args.seq_l, ppo_std_this_iter=ppo_std_this_iter)

            outputs_reg_focus_net = expected_outputs['outputs_reg_focus_net']
            loss_reg_focus_net = criterion(outputs_reg_focus_net, targets)
            loss_reg_focus_net = args.loss_reg_focus_net_weight * loss_reg_focus_net

            out_teacher = expected_outputs['x_focus'][-1].detach()

            loss_KD = F.kl_div(
                F.log_softmax(expected_outputs['x_glance'][0] / args.kd_temp, dim=1),
                F.softmax(out_teacher / args.kd_temp, dim=1),
                reduction='batchmean'
            ) * (args.kd_temp ** 2)

            if len(expected_outputs['x_focus']) > 1:
                loss_KD = loss_KD + sum(
                    [
                        F.kl_div(
                            F.log_softmax(_x_focus / args.kd_temp, dim=1),
                            F.softmax(out_teacher / args.kd_temp, dim=1),
                            reduction='batchmean'
                        ) * (args.kd_temp ** 2)
                        for _x_focus in expected_outputs['x_focus'][:-1]
                    ]
                )

            loss_glance = criterion(expected_outputs['x_glance'][0], targets)
            loss_focus = sum([criterion(_x_focus, targets) for _x_focus in expected_outputs['x_focus']])

            loss = loss_reg_focus_net + loss_focus + loss_glance + args.kd_alpha * loss_KD
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(
            loss, optimizer,
            model=model, skip_backbones=False, skip_policy_net=True,
            clip_grad=max_norm,
            parameters=model.parameters(), create_graph=is_second_order,
            update_grad=(data_iter_step + 1) % update_freq == 0
        )

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()

        ppo_update_collect_dict.append(expected_outputs)
        targets_collect_dict.append(targets)

        if (data_iter_step + 1) % args.update_policy_freq == 0:
            targets_ppo = torch.cat(targets_collect_dict, dim=0)

            for name, param in model.named_parameters():
                if ('policy_net_patch_stateV' in name) or ('policy_net_patch' in name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            expected_outputs_ppo = build_ppo_rollout(
                ppo_update_collect_dict=ppo_update_collect_dict,
                update_policy_freq=args.update_policy_freq,
            )

            rollout_outputs = expected_outputs_ppo['x_glance'] + expected_outputs_ppo['x_focus']
            list_to_compute_reward = [
                (
                    F.cross_entropy(rollout_outputs[focus_step_index].detach(), targets_ppo, reduction='none')
                    - F.cross_entropy(rollout_outputs[focus_step_index + 1].detach(), targets_ppo, reduction='none')
                ).unsqueeze(-1)
                for focus_step_index in range(args.seq_l)
            ]
            mb_rewards = torch.cat(list_to_compute_reward, dim=1)

            old_states = expected_outputs_ppo['states']
            old_actions = expected_outputs_ppo['actions']
            old_actions_logprobs = torch.cat(expected_outputs_ppo['actions_logprobs'], dim=1).detach()
            old_state_values = torch.cat(expected_outputs_ppo['_state_values'], dim=1).detach()
            old_policy_features_preact = torch.stack(
                expected_outputs_ppo['policy_features_preact'], dim=1
            ).detach()
            old_value_features_preact = torch.stack(
                expected_outputs_ppo['value_features_preact'], dim=1
            ).detach()

            mb_values = old_state_values
            mb_advs = torch.zeros_like(mb_rewards)
            lastgaelam = torch.zeros((mb_advs.shape[0],), device=targets_ppo.device)

            for t in reversed(range(args.seq_l)):
                if t == args.seq_l - 1:
                    nextnonterminal = 0.0
                    nextvalues = 0.0
                else:
                    nextnonterminal = 1.0
                    nextvalues = mb_values[:, t + 1]
                delta = mb_rewards[:, t] + args.gamma * nextvalues * nextnonterminal - mb_values[:, t]
                mb_advs[:, t] = lastgaelam = delta + args.gamma * args.ppo_lam * nextnonterminal * lastgaelam

            mb_returns = mb_advs + mb_values

            if args.adv_normalization:
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

            advantages = mb_advs
            num_ppo_update_iters = args.num_ppo_update_iters
            ppo_total_batch_size = targets_ppo.shape[0]
            mini_batch_size = int(ppo_total_batch_size / num_ppo_update_iters)

            for _ in range(args.ppo_update_steps):
                inds = torch.randperm(ppo_total_batch_size, device=targets_ppo.device)
                for i in range(num_ppo_update_iters):
                    batch_index = inds[
                        torch.arange(i * mini_batch_size, (i + 1) * mini_batch_size, device=targets_ppo.device)
                    ]

                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        logprobs, state_values, dist_entropy, policy_features_preact, value_features_preact = model(
                            old_states=old_states,
                            old_actions=old_actions,
                            seq_l=args.seq_l,
                            flag='evaluate_policy_net',
                            batch_index=batch_index,
                            ppo_std_this_iter=ppo_std_this_iter,
                        )

                        ratios = torch.exp(logprobs - old_actions_logprobs[batch_index].detach())
                        surr1 = ratios * advantages[batch_index]
                        surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages[batch_index]
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = mse_loss(state_values, mb_returns[batch_index].detach())

                        feature_policy_loss = F.mse_loss(
                            policy_features_preact,
                            old_policy_features_preact[batch_index].detach(),
                        )
                        feature_value_loss = F.mse_loss(
                            value_features_preact,
                            old_value_features_preact[batch_index].detach(),
                        )

                        ppo_loss = (
                            policy_loss
                            + value_loss
                            - args.ppo_entropy_coef * dist_entropy.mean()
                            + args.ppo_feature_policy_coef * feature_policy_loss
                            + args.ppo_feature_value_coef * feature_value_loss
                        )

                    optimizer.zero_grad()
                    _ = loss_scaler(
                        ppo_loss, optimizer,
                        model=model, skip_backbones=True, skip_policy_net=False,
                        clip_grad=args.ppo_clip_grid,
                        parameters=model.parameters()
                    )

            ppo_update_collect_dict = []
            targets_collect_dict = []

            for name, param in model.named_parameters():
                if ('policy_net_patch_stateV' in name) or ('policy_net_patch' in name):
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)

            if _should_run_focus_consistency(args, data_iter_step, update_freq):
                for name, param in model.named_parameters():
                    if ('policy_net_patch_stateV' in name) or ('policy_net_patch' in name):
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    focus_consistency_raw, focus_consistency_stats = compute_focus_consistency_loss(
                        model, policy_consistency_samples, args
                    )
                    focus_consistency_loss = args.focus_consistency_weight * focus_consistency_raw

                if not math.isfinite(focus_consistency_loss.item()):
                    print("Focus consistency loss is {}, stopping training".format(
                        focus_consistency_loss.item()
                    ))
                    assert math.isfinite(focus_consistency_loss.item())

                _ = loss_scaler(
                    focus_consistency_loss, optimizer,
                    model=model, skip_backbones=True, skip_policy_net=False,
                    clip_grad=args.ppo_clip_grid,
                    parameters=model.parameters()
                )
                optimizer.zero_grad()

                for name, param in model.named_parameters():
                    if ('policy_net_patch_stateV' in name) or ('policy_net_patch' in name):
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)

                metric_logger.update(focus_consistency_loss=focus_consistency_loss.item())
                metric_logger.update(**{
                    key: value.item() for key, value in focus_consistency_stats.items()
                })

        metric_logger.update(loss=loss_value)

        for focus_step_index in range(args.seq_l):
            pos_std = expected_outputs['pos_std'][focus_step_index]
            pos_mean = expected_outputs['pos_mean'][focus_step_index]
            metric_logger.update(**{
                f'pos_std_x_s{focus_step_index}': pos_std[1].item(),
                f'pos_std_y_s{focus_step_index}': pos_std[0].item(),
                f'pos_mean_x_s{focus_step_index}': pos_mean[1].item(),
                f'pos_mean_y_s{focus_step_index}': pos_mean[0].item(),
                f'scale_action_mean_s{focus_step_index}': expected_outputs['scale_action_mean'][focus_step_index].item(),
                f'scale_action_std_s{focus_step_index}': expected_outputs['scale_action_std'][focus_step_index].item(),
                f'patch_scale_mean_s{focus_step_index}': expected_outputs['patch_scale_mean'][focus_step_index].item(),
                f'patch_scale_std_s{focus_step_index}': expected_outputs['patch_scale_std'][focus_step_index].item(),
            })

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(ppo_std_this_iter=ppo_std_this_iter)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            expected_outputs = model(images, seq_l=args.seq_l)

            output_glance = expected_outputs['x_glance'][0]
            loss_glance = criterion(output_glance, target)

            output_main = expected_outputs['x_focus'][-1]
            loss_main = criterion(output_main, target)

        acc1_main, = accuracy(output_main, target, topk=(1,))
        acc1_glance, = accuracy(output_glance, target, topk=(1,))

        batch_size = images.shape[0]

        metric_logger.update(loss=loss_main.item())
        metric_logger.update(loss_glance=loss_glance.item())
        metric_logger.meters['acc1'].update(acc1_main.item(), n=batch_size)
        metric_logger.meters['acc1_glance'].update(acc1_glance.item(), n=batch_size)

        for focus_step_index in range(args.seq_l):
            output_step = expected_outputs['x_focus'][focus_step_index]
            acc1_step, = accuracy(output_step, target, topk=(1,))
            metric_logger.meters[f'acc1_step_{focus_step_index}'].update(
                acc1_step.item(), n=batch_size
            )

    metric_logger.synchronize_between_processes()

    print('* Main Path Acc@1 {:.3f} loss {:.3f}'.format(
        metric_logger.acc1.global_avg,
        metric_logger.loss.global_avg
    ))

    for focus_step_index in range(args.seq_l):
        print('  Step {} Acc@1 {:.3f}'.format(
            focus_step_index,
            metric_logger.meters[f'acc1_step_{focus_step_index}'].global_avg
        ))

    print('* Glance Acc@1 {:.3f} loss {:.3f}'.format(
        metric_logger.acc1_glance.global_avg,
        metric_logger.loss_glance.global_avg
    ))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
