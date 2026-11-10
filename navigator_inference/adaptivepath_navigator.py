import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_DIR = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPOSITORY_DIR, 'navigator_training')
MODELS_DIR = os.path.join(TRAINING_DIR, 'models')
for extra_path in [SCRIPT_DIR, REPOSITORY_DIR, TRAINING_DIR, MODELS_DIR]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    import cv2
except ImportError as exc:
    raise ImportError("adaptivepath_navigator.py requires opencv-python") from exc

try:
    import openslide
except ImportError as exc:
    raise ImportError("adaptivepath_navigator.py requires openslide-python") from exc

from models.dynamic_deitS_global_scale_single import dynamic_deitS


def load_checkpoint(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Navigator checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Navigator checkpoint must contain a model state_dict")
    return checkpoint["model"]


def load_model(args, device):
    print(f"Loading Navigator from {args.ckpt_path} ...")
    model = dynamic_deitS(
        seq_len=args.seq_l,
        feature_in_chans=args.feature_in_chans,
        recover_n=args.recover_n,
        remaining_blocks=args.remaining_blocks,
        glance_input_size=args.glance_input_size,
        glance_net_depth=12,
        glance_net_mlp_ratio=4,
        glance_net_drop_path=0.0,
        focus_patch_size=args.focus_patch_size,
        focus_net_reg_size=args.focus_net_reg_size,
        focus_net_depth=12,
        focus_net_mlp_ratio=4,
        focus_net_drop_path=0.0,
        multi_cls_drop_path=0.0,
        policy_net_hidden_chans=args.policy_net_hidden_chans,
        policy_net_kernel_size=args.policy_net_kernel_size,
        num_classes=args.num_classes,
        focus_scale_min=args.focus_scale_min,
        focus_scale_max=args.focus_scale_max,
    )
    model.load_state_dict(load_checkpoint(args.ckpt_path), strict=True)
    model.to(device).eval()
    return model


def get_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


class WSISlide:
    def __init__(self, path, thumbnail_max_size=4096):
        self.path = path
        self.slide = openslide.OpenSlide(path)
        self.level0_width, self.level0_height = self.slide.dimensions
        self.thumbnail = self.slide.get_thumbnail((thumbnail_max_size, thumbnail_max_size)).convert('RGB')
        self.thumb_width, self.thumb_height = self.thumbnail.size
        self.scale_x = self.level0_width / float(self.thumb_width)
        self.scale_y = self.level0_height / float(self.thumb_height)

    def read_region_level0(self, bbox_level0):
        x, y, w, h = bbox_level0
        return self.slide.read_region((int(x), int(y)), 0, (int(w), int(h))).convert('RGB')

    def level0_point_to_thumbnail(self, x_level0, y_level0):
        return (
            float(x_level0) / self.scale_x,
            float(y_level0) / self.scale_y,
        )

    def level0_bbox_to_thumbnail(self, bbox_level0):
        x, y, w, h = bbox_level0
        tx, ty = self.level0_point_to_thumbnail(x, y)
        tw = float(w) / self.scale_x
        th = float(h) / self.scale_y
        return (tx, ty, tw, th)

    def close(self):
        self.slide.close()


def build_tissue_mask_from_thumbnail(
    thumb_rgb,
    sthresh=8,
    sthresh_up=255,
    mthresh=7,
    close=4,
    use_otsu=False,
    filter_params=None,
):
    if filter_params is None:
        filter_params = {
            'min_tissue_area_ratio': 5e-4,
            'min_hole_area_ratio': 1e-5,
            'max_n_holes': 8,
        }

    img_hsv = cv2.cvtColor(np.asarray(thumb_rgb), cv2.COLOR_RGB2HSV)
    img_sat = img_hsv[:, :, 1]
    blur_ksize = mthresh if mthresh % 2 == 1 else mthresh + 1
    img_med = cv2.medianBlur(img_sat, blur_ksize)

    if use_otsu:
        _, img_bin = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    contours_info = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = contours_info[-2], contours_info[-1]

    H, W = img_bin.shape[:2]
    thumb_area = float(H * W)
    min_tissue_area = thumb_area * filter_params['min_tissue_area_ratio']
    min_hole_area = thumb_area * filter_params['min_hole_area_ratio']
    max_n_holes = filter_params['max_n_holes']

    kept_contours = []
    kept_holes = []
    bboxes = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        parent_indices = [idx for idx, h in enumerate(hierarchy) if h[3] == -1]
        for parent_idx in parent_indices:
            parent = contours[parent_idx]
            parent_area = cv2.contourArea(parent)
            child_indices = [idx for idx, h in enumerate(hierarchy) if h[3] == parent_idx]
            hole_contours = [contours[idx] for idx in child_indices]
            hole_areas = [cv2.contourArea(hole) for hole in hole_contours]
            net_area = parent_area - sum(hole_areas)
            if net_area < min_tissue_area:
                continue

            filtered_holes = [
                hole for hole, hole_area in sorted(
                    zip(hole_contours, hole_areas), key=lambda item: item[1], reverse=True
                )
                if hole_area >= min_hole_area
            ][:max_n_holes]

            kept_contours.append(parent)
            kept_holes.append(filtered_holes)
            bboxes.append(cv2.boundingRect(parent))

    mask = np.zeros((H, W), dtype=np.uint8)
    if kept_contours:
        cv2.drawContours(mask, kept_contours, -1, color=1, thickness=-1)
        for hole_group in kept_holes:
            if hole_group:
                cv2.drawContours(mask, hole_group, -1, color=0, thickness=-1)

    return {
        'mask': mask,
        'sat': img_sat,
        'binary': img_bin,
        'bounding_boxes': bboxes,
        'stats': {
            'num_contours': len(contours),
            'num_kept_contours': len(kept_contours),
            'tissue_ratio': float(mask.mean()),
        },
    }



def tissue_ratio_for_bbox(tissue_mask_thumb, wsi, bbox_level0):
    tx, ty, tw, th = wsi.level0_bbox_to_thumbnail(bbox_level0)
    mask_h, mask_w = tissue_mask_thumb.shape[:2]
    left = max(0, min(int(math.floor(tx)), mask_w - 1))
    top = max(0, min(int(math.floor(ty)), mask_h - 1))
    right = max(left + 1, min(int(math.ceil(tx + tw)), mask_w))
    bottom = max(top + 1, min(int(math.ceil(ty + th)), mask_h))
    return float(tissue_mask_thumb[top:bottom, left:right].mean())


def white_ratio_for_patch(patch_img):
    patch_np = np.asarray(patch_img.convert('RGB'))
    white = (
        (patch_np[:, :, 0] > 235)
        & (patch_np[:, :, 1] > 235)
        & (patch_np[:, :, 2] > 235)
    )
    return float(white.mean())


def abnormal_margin(logits_1d):
    return float(float(logits_1d[0]) - float(logits_1d[1]))


def decode_action_yx_scale(action, args):
    y_norm = float(np.clip(action[0], 0.0, 1.0))
    x_norm = float(np.clip(action[1], 0.0, 1.0))
    scale_action = float(np.clip(action[2], 0.0, 1.0)) if len(action) > 2 else 1.0
    patch_scale = float(args.focus_scale_min + scale_action * (args.focus_scale_max - args.focus_scale_min))
    return y_norm, x_norm, scale_action, patch_scale


def get_autocast_dtype(device, amp_dtype):
    if device.type != 'cuda' or amp_dtype == 'off':
        return None
    if amp_dtype == 'bf16':
        return torch.bfloat16
    return torch.float16


def normalize_single_outputs(outputs):
    x_focus = outputs['x_focus']
    actions = outputs['actions']
    if x_focus and isinstance(x_focus[0], list):
        x_focus = x_focus[0]
    if actions and isinstance(actions[0], list):
        actions = actions[0]
    return x_focus, actions


def split_single_outputs(outputs):
    x_focus_steps, action_steps = normalize_single_outputs(outputs)
    batch_size = x_focus_steps[0].shape[0] if x_focus_steps else 0
    samples = []
    for sample_idx in range(batch_size):
        sample = {
            'x_focus': [],
            'actions': [],
        }
        for step_idx in range(len(x_focus_steps)):
            sample['x_focus'].append(x_focus_steps[step_idx][sample_idx].detach().cpu())
            sample['actions'].append(action_steps[step_idx][sample_idx].detach().cpu())
        samples.append(sample)
    return samples


def infer_tensor_batch(model, batch_tensors, device, args):
    batch = torch.stack(batch_tensors, dim=0)
    if device.type == 'cuda':
        batch = batch.pin_memory()
    batch = batch.to(device, non_blocking=True)
    autocast_dtype = get_autocast_dtype(device, args.amp_dtype)
    with torch.inference_mode():
        if autocast_dtype is None:
            outputs = model(batch, seq_l=args.seq_l, flag='forward_backbone')
        else:
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(batch, seq_l=args.seq_l, flag='forward_backbone')
    return split_single_outputs(outputs)


def prepare_batch_tensors(records, image_getter, transform, io_workers, executor=None):
    def load_one(rec):
        return transform(image_getter(rec))

    max_workers = max(1, min(io_workers, len(records)))
    if max_workers == 1:
        return [load_one(rec) for rec in records]
    if executor is not None:
        return list(executor.map(load_one, records))
    with ThreadPoolExecutor(max_workers=max_workers) as temp_executor:
        return list(temp_executor.map(load_one, records))


def run_inference_records(model, records, image_getter, transform, device, args):
    if not records:
        return []

    all_outputs = []
    starts = list(range(0, len(records), args.batch_size))
    max_workers = max(1, min(args.io_workers, args.batch_size))
    with ThreadPoolExecutor(max_workers=max_workers) as io_executor, ThreadPoolExecutor(max_workers=1) as prefetch_executor:
        next_future = prefetch_executor.submit(
            prepare_batch_tensors,
            records[starts[0]:starts[0] + args.batch_size],
            image_getter,
            transform,
            args.io_workers,
            io_executor,
        )
        for chunk_idx, start in enumerate(tqdm(starts, desc='Inference', leave=False)):
            batch_tensors = next_future.result()
            if chunk_idx + 1 < len(starts):
                next_start = starts[chunk_idx + 1]
                next_future = prefetch_executor.submit(
                    prepare_batch_tensors,
                    records[next_start:next_start + args.batch_size],
                    image_getter,
                    transform,
                    args.io_workers,
                    io_executor,
                )
            else:
                next_future = None
            all_outputs.extend(infer_tensor_batch(model, batch_tensors, device, args))
    return all_outputs


def focus_bbox_from_action(roi_bbox_level0, action, args):
    y_norm, x_norm, scale_action, patch_scale = decode_action_yx_scale(action, args)
    rx, ry, rw, rh = roi_bbox_level0
    focus_w = int(round(rw * args.focus_patch_size * patch_scale / float(args.input_size)))
    focus_h = int(round(rh * args.focus_patch_size * patch_scale / float(args.input_size)))
    focus_w = max(1, min(focus_w, rw))
    focus_h = max(1, min(focus_h, rh))
    left = rx + x_norm * max(rw - focus_w, 0)
    top = ry + y_norm * max(rh - focus_h, 0)
    return {
        'y_norm': y_norm,
        'x_norm': x_norm,
        'scale_action': scale_action,
        'patch_scale': patch_scale,
        'bbox_level0': (int(round(left)), int(round(top)), int(focus_w), int(focus_h)),
    }


