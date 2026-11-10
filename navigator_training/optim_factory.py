import torch


def get_parameter_groups(model, weight_decay, skip_list, backbone_lr_scale, policy_lr_scale):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        no_decay = parameter.ndim == 1 or name.endswith(".bias") or name in skip_list
        is_policy = "policy_net_patch" in name
        parameter_type = "policy" if is_policy else "backbone"
        decay_type = "no_decay" if no_decay else "decay"
        group_name = f"{parameter_type}_{decay_type}"
        if group_name not in groups:
            groups[group_name] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": policy_lr_scale if is_policy else backbone_lr_scale,
                "fix_step": 0,
                "group_name": group_name,
            }
        groups[group_name]["params"].append(parameter)
    return list(groups.values())


def create_optimizer(args, model, skip_list=None, backbone_lr_scale=0.01):
    if skip_list is None:
        skip_list = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    parameters = get_parameter_groups(
        model,
        weight_decay=args.weight_decay,
        skip_list=skip_list,
        backbone_lr_scale=backbone_lr_scale,
        policy_lr_scale=args.stn_lr_ratio,
    )
    return torch.optim.AdamW(parameters, lr=args.lr, eps=1e-8, weight_decay=0.0)
