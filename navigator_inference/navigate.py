#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import torch

from adaptivepath_navigator import WSISlide
from selection import select_region_focuses, select_top_regions


MAX_REGION_BUDGET = 5
MAX_FOCUS_BUDGET = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaptivePath hierarchical WSI navigation")
    parser.add_argument("--wsi_csv", required=True, help="CSV with a required svs_path column")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--io_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Fixed paper configuration.
    args.amp_dtype = "fp16"
    args.input_size = 512
    args.glance_input_size = 112
    args.focus_patch_size = 112
    args.focus_scale_min = 0.25
    args.focus_scale_max = 1.0
    args.focus_net_reg_size = 160
    args.feature_in_chans = 384
    args.recover_n = 5
    args.remaining_blocks = 4
    args.policy_net_hidden_chans = 128
    args.policy_net_kernel_size = 3
    args.num_classes = 2
    args.seq_l = MAX_FOCUS_BUDGET
    args.thumbnail_max_size = 4096
    args.min_region_tissue_ratio = 0.15
    args.min_focus_tissue_ratio = 0.15
    args.max_focus_white_ratio = 0.80
    args.window_size_2p5x = 8192
    args.window_size_5x = 4096
    args.window_size_10x = 2048
    return args


def sliding_positions(start: int, span: int, window: int) -> list[int]:
    if span <= window:
        return [start]
    last = start + span - window
    positions = list(range(start, last + 1, window))
    if positions[-1] != last:
        positions.append(last)
    return positions


def candidate_regions(
    parents: Iterable[tuple[int, int, int, int]],
    window_size: int,
    wsi: Any,
    tissue_mask: np.ndarray,
    navigator: ModuleType,
    min_tissue_ratio: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for parent_id, parent in enumerate(parents):
        px, py, pw, ph = parent
        width = min(window_size, pw, wsi.level0_width)
        height = min(window_size, ph, wsi.level0_height)
        for y in sliding_positions(py, ph, height):
            for x in sliding_positions(px, pw, width):
                bbox = (
                    max(0, min(int(x), wsi.level0_width - width)),
                    max(0, min(int(y), wsi.level0_height - height)),
                    int(width),
                    int(height),
                )
                if bbox in seen:
                    continue
                seen.add(bbox)
                tissue_ratio = navigator.tissue_ratio_for_bbox(tissue_mask, wsi, bbox)
                if tissue_ratio < min_tissue_ratio:
                    continue
                records.append({
                    "region_id": len(records),
                    "parent_region_id": None if (px == 0 and py == 0 and pw == wsi.level0_width and ph == wsi.level0_height) else parent_id,
                    "bbox_level0": bbox,
                    "tissue_ratio": tissue_ratio,
                })
    return records


def save_focuses(
    selected: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    level_dir: Path,
    wsi: Any,
    tissue_mask: np.ndarray,
    navigator: ModuleType,
    args: argparse.Namespace,
) -> None:
    focus_dir = level_dir / "focus"
    focus_dir.mkdir(parents=True, exist_ok=True)
    for region, sample in zip(selected, outputs):
        candidates = []
        for step_idx, (logits, action) in enumerate(zip(sample["x_focus"], sample["actions"])):
            decoded = navigator.focus_bbox_from_action(region["bbox_level0"], action.numpy(), args)
            bbox = tuple(int(v) for v in decoded["bbox_level0"])
            image = wsi.read_region_level0(bbox)
            tissue_ratio = navigator.tissue_ratio_for_bbox(tissue_mask, wsi, bbox)
            white_ratio = navigator.white_ratio_for_patch(image)
            reason = ""
            if tissue_ratio < args.min_focus_tissue_ratio:
                reason = "low_tissue"
            elif white_ratio > args.max_focus_white_ratio:
                reason = "high_white"
            item = {
                "step_idx": step_idx,
                "step_region_score": navigator.abnormal_margin(logits),
                "bbox_level0": list(bbox),
                "location_level0": [bbox[0], bbox[1]],
                "focus_tissue_ratio": tissue_ratio,
                "focus_white_ratio": white_ratio,
                "filter_reason": reason,
            }
            if not reason:
                name = f"region_{region['rank']:02d}_focus_{step_idx:02d}.jpg"
                image.save(focus_dir / name)
                item["image_path"] = str((focus_dir / name).relative_to(level_dir.parent))
            image.close()
            candidates.append(item)
        region["focuses"] = select_region_focuses(candidates, MAX_FOCUS_BUDGET)
        region["filtered_focuses"] = [row for row in candidates if row["filter_reason"]]


def run_level(
    key: str,
    magnification: float,
    window_size: int,
    parents: list[tuple[int, int, int, int]],
    slide_dir: Path,
    model: Any,
    transform: Any,
    wsi: Any,
    tissue_mask: np.ndarray,
    navigator: ModuleType,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = candidate_regions(
        parents, window_size, wsi, tissue_mask, navigator, args.min_region_tissue_ratio
    )
    outputs = navigator.run_inference_records(
        model,
        records,
        image_getter=lambda row: wsi.read_region_level0(row["bbox_level0"]),
        transform=transform,
        device=torch.device(args.device),
        args=args,
    )
    scored = []
    for record, output in zip(records, outputs):
        if not output["x_focus"]:
            continue
        row = dict(record)
        row["abnormal_score"] = navigator.abnormal_margin(output["x_focus"][-1])
        row["step_region_scores"] = [navigator.abnormal_margin(x) for x in output["x_focus"]]
        row["filter_reason"] = ""
        row["_output"] = output
        scored.append(row)
    selected = select_top_regions(scored, MAX_REGION_BUDGET)
    selected_outputs = []
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        selected_outputs.append(row.pop("_output"))
    level_dir = slide_dir / key
    level_dir.mkdir(parents=True, exist_ok=True)
    save_focuses(selected, selected_outputs, level_dir, wsi, tissue_mask, navigator, args)
    payload = {
        "magnification": magnification,
        "window_size_level0": window_size,
        "num_candidate_regions": len(records),
        "regions": selected,
    }
    (level_dir / "navigation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return selected


def list_wsi(csv_path: str) -> list[Path]:
    import csv

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "svs_path" not in reader.fieldnames:
            raise ValueError("WSI CSV must contain an svs_path column")
        paths = [Path(row["svs_path"].strip()) for row in reader if row["svs_path"].strip()]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"WSI not found: {missing[0]}")
    return paths


def main() -> None:
    args = parse_args()
    import adaptivepath_navigator as navigator
    device = torch.device(args.device)
    model = navigator.load_model(args, device)
    if args.input_size != 512:
        raise ValueError(f"AdaptivePath Navigator input_size must be 512, got {args.input_size}")
    transform = navigator.get_transform(args.input_size)
    def transform_and_close(image):
        tensor = transform(image)
        image.close()
        return tensor
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for wsi_path in list_wsi(args.wsi_csv):
        slide_id = hashlib.sha256(wsi_path.name.encode("utf-8")).hexdigest()[:16]
        slide_dir = output_root / slide_id
        final_path = slide_dir / "navigation.json"
        if final_path.exists() and not args.overwrite:
            print(f"[Skip] {slide_id}")
            continue
        slide_dir.mkdir(parents=True, exist_ok=True)
        wsi = WSISlide(str(wsi_path), thumbnail_max_size=args.thumbnail_max_size)
        try:
            tissue = navigator.build_tissue_mask_from_thumbnail(wsi.thumbnail)["mask"]
            parents = [(0, 0, wsi.level0_width, wsi.level0_height)]
            levels = {}
            level_specs = (
                ("2p5x", 2.5, args.window_size_2p5x),
                ("5x", 5.0, args.window_size_5x),
                ("10x", 10.0, args.window_size_10x),
            )
            for key, mag, window in level_specs:
                regions = run_level(
                    key, mag, window, parents, slide_dir, model, transform_and_close, wsi,
                    tissue, navigator, args,
                )
                levels[key] = regions
                parents = [tuple(row["bbox_level0"]) for row in regions]
                if not parents:
                    break
            payload = {
                "slide_id": slide_id,
                "navigation_budget": {
                    "K_max_regions_per_magnification": MAX_REGION_BUDGET,
                    "T_max_focuses_per_region": MAX_FOCUS_BUDGET,
                },
                "score_semantics": (
                    "abnormal_score is the final Region-level margin logits[0]-logits[1]; "
                    "step_region_score is an intermediate Region prediction after a focus observation."
                ),
                "levels": levels,
            }
            final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[Done] {slide_id}")
        finally:
            wsi.close()


if __name__ == "__main__":
    main()
