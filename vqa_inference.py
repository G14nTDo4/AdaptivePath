#!/usr/bin/env python3
"""End-to-end AdaptivePath VQA inference from Navigator focus images."""
from __future__ import annotations

import argparse
import contextvars
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


LEVELS = (("2p5x", 2.5), ("5x", 5.0), ("10x", 10.0))
MAX_REGION_BUDGET = 5
MAX_FOCUS_BUDGET = 4
OUTPUT_TOKEN_BUDGET = 2048
MAX_POSITIVE_FINDINGS = 6

LEVEL_ROLES = {
    2.5: (
        "Evaluate low-power global architecture, growth pattern, tissue distribution, "
        "and lesion boundaries. Do not require cytologic details that this scale cannot resolve."
    ),
    5.0: (
        "Evaluate intermediate-power tissue architecture, epithelial organization, stromal "
        "relationships, and invasive or interface patterns."
    ),
    10.0: (
        "Evaluate high-power cytology, including nuclear features, cytoplasm, keratinization, "
        "nucleoli, and cellular cohesion. Do not replace architecture with cytology alone."
    ),
}

NOT_OBSERVED_RE = re.compile(
    r"\b(no|not|without|absent|absence|lack|lacking|unobserved|not observed)\b",
    re.IGNORECASE,
)

ACTIVE_TASK: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("active_vqa_task", default=None)
)

MORPHOLOGY_INTERPRETER_SYSTEM_PROMPT = (
    "Describe only directly visible histopathologic evidence relevant to the "
    "question. Do not answer the slide-level question. Do not infer an IHC "
    "marker, receptor status, molecular subtype, sequencing result, clinical "
    "history, or control staining unless it is directly visible and identifiable "
    "in this image. If the requested evidence is not visually assessable, state "
    "that explicitly. Report tissue architecture, cell morphology, staining "
    "appearance, necrosis, stroma, and other visible findings concisely."
)

DELIBERATOR_ANSWER_SYSTEM_PROMPT = (
    "Answer one pathology multiple-choice question independently from the "
    "current magnification. Use only the supplied directly observed region "
    "morphology. Compare the observed morphology against every option "
    "symmetrically before selecting the answer. Missing or unobserved findings "
    "are neutral, and abnormality scores are navigation metadata."
)

DELIBERATOR_REVIEW_SYSTEM_PROMPT = (
    "Review the magnification-specific answer against directly observed region "
    "morphology. Audit every option symmetrically and assess whether the evidence "
    "is sufficient for the proposed answer. Missing, unmentioned, absent, or "
    "unobserved features are neutral. A conflict must be supported by directly "
    "observed positive morphology."
)

ARBITER_SYSTEM_PROMPT = (
    "Make a new final pathology decision from option-wise evidence audits and "
    "evidence-sufficiency reviews across magnifications. Do not count votes, use "
    "absent findings, treat higher magnification as automatically superior, or "
    "use class prevalence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end AdaptivePath VQA inference from Navigator focus images"
    )
    parser.add_argument(
        "--case_manifest", required=True,
        help="JSON list with navigation_path, question, and choices",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir")
    parser.add_argument("--vlm_ckpt")
    parser.add_argument("--llm_ckpt")
    parser.add_argument("--cuda_visible_devices", default="0,1")
    parser.add_argument("--vlm_device", default="cuda:0")
    parser.add_argument("--llm_device", default="cuda:1")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--vlm_max_new_tokens", type=int, default=512)
    parser.add_argument("--max_memory_gib", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and (not args.vlm_ckpt or not args.llm_ckpt):
        parser.error("--vlm_ckpt and --llm_ckpt are required unless --dry_run is set")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("shard_id must satisfy 0 <= shard_id < num_shards")
    return args


def question_id(slide_id: str, question: str) -> str:
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return f"{slide_id}_{digest}"



def load_case_manifest(path: str) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("case_manifest must contain a JSON list")
    cases = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest record {index} is not an object")
        navigation_value = str(item.get("navigation_path") or "").strip()
        question = str(item.get("question") or "").strip()
        choices = [str(choice) for choice in item.get("choices") or []]
        if not navigation_value or not question or not choices:
            raise ValueError(
                f"Manifest record {index} requires navigation_path, question, and choices"
            )
        navigation_path = Path(navigation_value)
        slide_id = str(item.get("slide_id") or navigation_path.parent.name)
        cases.append(
            {
                **item,
                "slide_id": slide_id,
                "question_id": str(
                    item.get("question_id") or question_id(slide_id, question)
                ),
                "question": question,
                "choices": choices,
                "ground_truth": str(item.get("ground_truth") or ""),
                "navigation_path": str(navigation_path),
            }
        )
    return cases


def resolve_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = load_case_manifest(args.case_manifest)
    selected = [
        case for index, case in enumerate(cases)
        if index % args.num_shards == args.shard_id
    ][args.start_index:]
    return selected[:args.limit] if args.limit is not None else selected


def resolve_focus_path(
    raw_path: str,
    navigation_path: Path,
) -> Path:
    source = Path(raw_path)
    if source.is_absolute():
        raise ValueError(f"Focus image_path must be relative: {raw_path}")
    path = navigation_path.parent / source
    if not path.is_file():
        raise FileNotFoundError(f"Focus image not found: {path}")
    return path


def load_and_validate_navigation(
    navigation_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not navigation_path.is_file():
        raise FileNotFoundError(f"Navigation JSON not found: {navigation_path}")
    navigation = json.loads(navigation_path.read_text(encoding="utf-8"))
    # Only the first round is mandatory. Later rounds are checked when requested.
    count = validate_navigation_level(navigation, navigation_path, LEVELS[0][0])
    if count == 0:
        raise ValueError("No focus images at the first navigation level")
    return navigation, {LEVELS[0][0]: count}


def validate_navigation_level(
    navigation: dict[str, Any], navigation_path: Path, level_key: str,
) -> int:
    regions = (navigation.get("levels") or {}).get(level_key)
    if not isinstance(regions, list) or not regions:
        return 0
    if len(regions) > MAX_REGION_BUDGET:
        raise ValueError(
            f"Navigation level {level_key} exceeds region budget "
            f"K={MAX_REGION_BUDGET}: found {len(regions)}"
        )
    count = 0
    for region in regions:
        focuses = region.get("focuses") or []
        if not isinstance(focuses, list):
            raise ValueError(f"Region focuses must be a list at navigation level: {level_key}")
        if len(focuses) > MAX_FOCUS_BUDGET:
            raise ValueError(
                f"Region at {level_key} exceeds focus budget "
                f"T={MAX_FOCUS_BUDGET}: found {len(focuses)}"
            )
        for focus in focuses:
            resolve_focus_path(str(focus.get("image_path") or ""), navigation_path)
            count += 1
    return count


def cache_key(case: dict[str, Any]) -> str:
    payload = json.dumps(
        {"question": case["question"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe_focus(
    image_path: Path,
    question: str,
    magnification: float,
    processor: Any,
    model: Any,
    max_tokens: int,
) -> str:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    user_prompt = (
        f"Slide-level question: {question}\n"
        f"Magnification: {magnification}x"
    )
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": MORPHOLOGY_INTERPRETER_SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens)
    trimmed = [
        output[len(source):] for source, output in zip(inputs.input_ids, generated)
    ]
    description = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    image.close()
    return description


def collect_descriptions(
    navigation: dict[str, Any],
    navigation_path: Path,
    level_key: str,
    case: dict[str, Any],
    cache_root: Path,
    processor: Any,
    vlm: Any,
    max_tokens: int,
) -> list[dict[str, Any]]:
    magnification = dict(LEVELS)[level_key]
    records = []
    level_cache = cache_root / case["question_id"] / cache_key(case) / level_key
    level_cache.mkdir(parents=True, exist_ok=True)
    for region in navigation["levels"][level_key]:
        for focus in region.get("focuses") or []:
            image_path = resolve_focus_path(
                str(focus.get("image_path") or ""), navigation_path
            )
            cache_path = level_cache / (
                f"region_{int(region['rank']):02d}_"
                f"focus_{int(focus['step_idx']):02d}.txt"
            )
            if cache_path.is_file():
                description = cache_path.read_text(encoding="utf-8").strip()
            else:
                description = describe_focus(
                    image_path, case["question"], magnification,
                    processor, vlm, max_tokens,
                )
                cache_path.write_text(description, encoding="utf-8")
            records.append(
                {
                    "magnification": magnification,
                    "region_rank": int(region["rank"]),
                    "region_id": int(region["region_id"]),
                    "abnormal_score": float(region["abnormal_score"]),
                    "region_bbox_level0": region.get("bbox_level0"),
                    "focus_step": int(focus["step_idx"]),
                    "focus_bbox_level0": focus.get("bbox_level0"),
                    "location_level0": focus.get("location_level0"),
                    "description": description,
                    "image_path": str(focus.get("image_path") or ""),
                }
            )
    return records


def device_map(value: str) -> str | dict[str, str]:
    return "auto" if value == "auto" else {"": value}


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:index + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


def has_required_llm_fields(parsed: dict[str, Any] | None) -> bool:
    if parsed is None:
        return False
    task = ACTIVE_TASK.get() or {}
    kind = task.get("kind")
    if kind == "evidence_review":
        return isinstance(parsed.get("option_evidence"), list) and bool(
            parsed.get("evidence_sufficiency")
        )
    if kind == "region_aggregation":
        return isinstance(parsed.get("positive_findings"), list)
    if kind in {"level_answer", "final_adjudication"}:
        return bool(parsed.get("answer_label") or parsed.get("answer_text") or parsed.get("answer"))
    return True


def call_llm_json(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens)
    output = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    parsed = extract_json_object(output)
    if not has_required_llm_fields(parsed):
        task = ACTIVE_TASK.get() or {}
        if task.get("kind") == "evidence_review":
            repair_instruction = (
                "Return exactly one valid JSON object with option_evidence as a list "
                f"of {len(task.get('choices') or [])} entries in option order. Each entry "
                "has supporting_findings and explicit_conflicts arrays. Also include "
                "evidence_sufficiency as sufficient or insufficient and a "
                "sufficiency_reason string. Preserve the original evidence assessment."
            )
        else:
            repair_instruction = "Repair the response into one valid JSON object containing all requested fields."
        repair = [
            {"role": "system", "content": repair_instruction + " Return JSON only."},
            {
                "role": "user",
                "content": "Original task:\n" + messages[1]["content"] + "\n\nResponse to repair:\n" + output,
            },
        ]
        repair_prompt = tokenizer.apply_chat_template(
            repair, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        repair_inputs = tokenizer([repair_prompt], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            repaired_ids = model.generate(**repair_inputs, max_new_tokens=max_tokens)
        repaired = tokenizer.decode(
            repaired_ids[0][repair_inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        parsed = extract_json_object(repaired)
        if not has_required_llm_fields(parsed):
            raise ValueError("LLM returned JSON without required task fields")
        output += "\n[JSON_REPAIR]\n" + repaired
    return output, parsed


def save_result(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def option_table(choices: list[str]) -> str:
    return "\n".join(
        f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
    )


def canonical_answer(parsed: dict[str, Any], choices: list[str]) -> dict[str, str]:
    raw_label = str(parsed.get("answer_label") or "").strip().upper()
    raw_text = str(parsed.get("answer_text") or parsed.get("answer") or "").strip()
    label_index = ord(raw_label) - ord("A") if len(raw_label) == 1 else -1
    if 0 <= label_index < len(choices):
        index = label_index
    else:
        normalized = normalize_text(raw_text)
        matches = [
            index
            for index, choice in enumerate(choices)
            if normalized
            in {
                normalize_text(choice),
                normalize_text(f"{chr(ord('A') + index)}. {choice}"),
            }
        ]
        if len(matches) != 1:
            raise ValueError(f"Answer does not map to exactly one choice: {parsed}")
        index = matches[0]
    label = chr(ord("A") + index)
    text = str(choices[index])
    return {
        "answer_label": label,
        "answer_text": text,
        "answer": f"{label}. {text}",
    }


def group_focus_descriptions(
    descriptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in descriptions:
        key = (int(row["region_rank"]), int(row["region_id"]))
        grouped[key].append(row)

    regions = []
    for (region_rank, region_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["focus_step"]))
        first = rows[0]
        regions.append(
            {
                "region_rank": region_rank,
                "region_id": region_id,
                "abnormal_score": first["abnormal_score"],
                "region_bbox_level0": first["region_bbox_level0"],
                "focus_descriptions": [
                    {
                        "focus_step": row["focus_step"],
                        "focus_bbox_level0": row["focus_bbox_level0"],
                        "location_level0": row["location_level0"],
                        "description": row["description"],
                    }
                    for row in rows
                ],
            }
        )
    return regions


def finding_text(item: Any) -> str:
    return str(item.get("finding") if isinstance(item, dict) else item or "").strip()


def is_not_observed(item: Any) -> bool:
    return bool(NOT_OBSERVED_RE.search(finding_text(item)))


def contains_option_name(text: str, choices: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(choice) in normalized for choice in choices)


def prepare_source(source: dict[str, Any]) -> dict[str, Any]:
    source = dict(source)
    source["choices"] = [str(choice) for choice in source.get("choices") or []]
    return source


def finalize_result(
    result: dict[str, Any],
    source: dict[str, Any],
    final: dict[str, Any],
) -> None:
    answer_label = final["answer_label"]
    final["original_answer_label"] = answer_label
    result["pred_answer_original_label"] = answer_label


def level_role(magnification: float) -> str:
    return LEVEL_ROLES[float(magnification)]


@contextmanager
def task_scope(
    kind: str, question: str, choices: list[str]
) -> Iterator[None]:
    token = ACTIVE_TASK.set(
        {"kind": kind, "question": question, "choices": list(choices)}
    )
    try:
        yield
    finally:
        ACTIVE_TASK.reset(token)


def context_checked_call(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    checked_messages = [dict(message) for message in messages]
    token_budget = min(max_tokens, OUTPUT_TOKEN_BUDGET)
    prompt = tokenizer.apply_chat_template(
        checked_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    context_limit = int(model.config.max_position_embeddings)
    projected_tokens = input_tokens + token_budget
    task = ACTIVE_TASK.get()
    print(
        json.dumps(
            {
                "event": "context_audit",
                "kind": task["kind"] if task else "unknown",
                "input_tokens": input_tokens,
                "max_new_tokens": token_budget,
                "projected_tokens": projected_tokens,
                "context_limit": context_limit,
            }
        ),
        flush=True,
    )
    if projected_tokens > context_limit:
        raise ValueError(
            "Context budget exceeded without truncation: "
            f"{projected_tokens} > {context_limit}"
        )
    return call_llm_json(model, tokenizer, checked_messages, token_budget)


def compact_positive_findings(
    items: Any,
    choices: list[str],
) -> list[str]:
    findings: list[str] = []
    observed: set[str] = set()
    for item in items if isinstance(items, list) else []:
        text = finding_text(item)
        normalized = normalize_text(text)
        if (
            not text
            or is_not_observed(item)
            or contains_option_name(text, choices)
            or normalized in observed
        ):
            continue
        observed.add(normalized)
        findings.append(text)
        if len(findings) == MAX_POSITIVE_FINDINGS:
            break
    return findings


def normalize_minimal_regions(
    summaries: list[dict[str, Any]],
    choices: list[str],
) -> list[dict[str, Any]]:
    normalized = []
    for row in summaries:
        findings = compact_positive_findings(
            row.get("positive_findings"), choices
        )
        summary = "; ".join(findings)
        normalized.append(
            {
                "region_rank": int(row["region_rank"]),
                "region_id": int(row["region_id"]),
                "abnormal_score": float(row["abnormal_score"]),
                "morphology_summary": summary,
                "positive_findings": findings,
            }
        )
    return normalized


def normalize_evidence_items(items: Any, choices: list[str]) -> list[str]:
    findings: list[str] = []
    observed: set[str] = set()
    for item in items if isinstance(items, list) else []:
        finding = finding_text(item)
        normalized = normalize_text(finding)
        if (
            not finding
            or is_not_observed(item)
            or contains_option_name(finding, choices)
            or normalized in observed
        ):
            continue
        observed.add(normalized)
        findings.append(finding)
    return findings


def normalize_option_evidence(
    parsed: dict[str, Any], choices: list[str]
) -> dict[str, Any]:
    rows = parsed.get("option_evidence")
    if not isinstance(rows, list):
        raise ValueError(f"option_evidence is not a list: {parsed}")

    labels = [chr(ord("A") + index) for index in range(len(choices))]
    has_labels = any(
        isinstance(row, dict) and row.get("answer_label") for row in rows
    )
    indexed: dict[str, dict[str, Any]] = {}
    if has_labels:
        for row in rows:
            if not isinstance(row, dict):
                continue
            answer = canonical_answer(row, choices)
            indexed[answer["answer_label"]] = row
        if set(indexed) != set(labels):
            raise ValueError(
                "Option evidence label mismatch: "
                f"expected={set(labels)} observed={set(indexed)}"
            )
    else:
        if len(rows) != len(choices):
            raise ValueError(
                "Option evidence count mismatch: "
                f"expected={len(choices)} observed={len(rows)}"
            )
        normalized_rows = []
        for row in rows:
            if isinstance(row, dict):
                normalized_rows.append(row)
            elif isinstance(row, list):
                normalized_rows.append(
                    {
                        "supporting_findings": row,
                        "explicit_conflicts": [],
                    }
                )
            elif isinstance(row, str) and row.strip():
                normalized_rows.append(
                    {
                        "supporting_findings": [row],
                        "explicit_conflicts": [],
                    }
                )
            else:
                normalized_rows.append(
                    {
                        "supporting_findings": [],
                        "explicit_conflicts": [],
                    }
                )
        indexed = dict(zip(labels, normalized_rows))

    return {
        "option_evidence": [
            {
                "answer_label": label,
                "answer_text": choices[index],
                "answer": f"{label}. {choices[index]}",
                "supporting_findings": normalize_evidence_items(
                    indexed[label].get("supporting_findings"), choices
                ),
                "explicit_conflicts": normalize_evidence_items(
                    indexed[label].get("explicit_conflicts"), choices
                ),
            }
            for index, label in enumerate(labels)
        ]
    }


def aggregate_regions(
    model: Any,
    tokenizer: Any,
    question: str,
    choices: list[str],
    magnification: float,
    grouped_regions: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    system = (
        "Aggregate directly observed pathology morphology from one Region. Relate "
        "descriptions of different focus crops from that same Region, merge synonymous "
        "observations, and omit every absent, unmentioned, uncertain, or not-observed "
        "feature. Do not output diagnoses, answer options, conflicts, limitations, "
        "confidence, metadata, or recommendations. Return one compact JSON object only."
    )
    raw_outputs = []
    complete = []
    for source in grouped_regions:
        descriptions = [
            str(focus.get("description") or "").strip()
            for focus in source.get("focus_descriptions") or []
            if str(focus.get("description") or "").strip()
        ]
        user = (
            f"Question: {question}\nOptions:\n{option_table(choices)}\n"
            f"Magnification: {magnification}x\n"
            f"Scale-specific role: {level_role(magnification)}\n\n"
            f"Focus descriptions from one Region:\n"
            f"{json.dumps(descriptions, ensure_ascii=False, indent=2)}\n\n"
            "Return only positive_findings, a list of at most 6 short strings. "
            "Do not return Region rank, ID, score, location, input order, or any "
            "other metadata. Omit absent or not-observed features."
        )
        try:
            with task_scope("region_aggregation", question, choices):
                raw, parsed = context_checked_call(
                    model,
                    tokenizer,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    OUTPUT_TOKEN_BUDGET,
                )
            findings = parsed.get("positive_findings")
            if not isinstance(findings, list):
                raise ValueError(
                    f"positive_findings is not a list: {parsed}"
                )
            raw_outputs.append(raw)
            complete.append(
                {
                    "region_rank": int(source["region_rank"]),
                    "region_id": int(source["region_id"]),
                    "abnormal_score": float(source["abnormal_score"]),
                    "morphology_summary": "",
                    "positive_findings": findings,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            print(
                json.dumps(
                    {
                        "event": "region_aggregation_json_error",
                        "kind": "region_aggregation",
                        "magnification": magnification,
                        "source_region_rank": source.get("region_rank"),
                        "error": type(error).__name__,
                    }
                ),
                flush=True,
            )
            raise

    return "\n".join(raw_outputs), {
        "aggregation_mode": "independent_regions",
        "region_summaries": normalize_minimal_regions(complete, choices),
    }


def answer_level(
    model: Any,
    tokenizer: Any,
    question: str,
    choices: list[str],
    task: str,
    magnification: float,
    region_summaries: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    regions = normalize_minimal_regions(region_summaries, choices)
    system = DELIBERATOR_ANSWER_SYSTEM_PROMPT
    user = (
        f"Question: {question}\n"
        f"Options:\n{option_table(choices)}\n"
        f"Current magnification: {magnification}x\n"
        f"Region morphology:\n"
        f"{json.dumps(regions, ensure_ascii=False, indent=2)}\n\n"
        "Return only answer_label and reason. answer_label must be one displayed "
        "option letter. Do not repeat answer_text or assess evidence sufficiency."
    )
    with task_scope("level_answer", question, choices):
        raw, parsed = context_checked_call(
            model,
            tokenizer,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            OUTPUT_TOKEN_BUDGET,
        )
    answer = {
        **canonical_answer(parsed, choices),
        "reason": str(parsed.get("reason") or "").strip(),
    }
    return raw, answer


def review_sufficiency(
    model: Any,
    tokenizer: Any,
    question: str,
    choices: list[str],
    magnification: float,
    region_summaries: list[dict[str, Any]],
    answer: dict[str, Any],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    regions = normalize_minimal_regions(region_summaries, choices)
    system = DELIBERATOR_REVIEW_SYSTEM_PROMPT
    independent_answer = answer["answer"]
    independent_rationale = answer["reason"]
    user = (
        f"Question: {question}\nOptions:\n{option_table(choices)}\n"
        f"Current magnification: {magnification}x\n"
        f"Region morphology:\n"
        f"{json.dumps(regions, ensure_ascii=False, indent=2)}\n\n"
        f"Independent answer: {independent_answer}\n"
        f"Independent rationale: {independent_rationale}\n\n"
        "Return option_evidence as a list in the exact option order shown above, "
        "with one entry per option. Each entry must contain only "
        "supporting_findings and explicit_conflicts, both lists of concise, "
        "distinct, directly observed morphology strings. Also return "
        "evidence_sufficiency as exactly sufficient or insufficient and a "
        "sufficiency_reason. Do not return a new answer, confidence, probability, "
        "missing findings, option labels, or option texts."
    )
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    messages = base_messages
    last_error: Exception | None = None
    for attempt in range(1, 4):
        raw = ""
        parsed: dict[str, Any] = {}
        try:
            with task_scope("evidence_review", question, choices):
                raw, parsed = context_checked_call(
                    model,
                    tokenizer,
                    messages,
                    OUTPUT_TOKEN_BUDGET,
                )
            audit = normalize_option_evidence(parsed, choices)
            status = str(
                parsed.get("evidence_sufficiency") or ""
            ).strip().lower()
            if status not in {"sufficient", "insufficient"}:
                raise ValueError(f"Invalid evidence_sufficiency: {parsed}")
            return raw, {
                "option_evidence_audit": audit,
                "evidence_sufficiency": status,
                "sufficiency_reason": str(
                    parsed.get("sufficiency_reason") or ""
                ).strip(),
            }
        except (TypeError, ValueError) as error:
            last_error = error
            print(
                json.dumps(
                    {
                        "event": "vqa_review_retry",
                        "attempt": attempt,
                        "expected_options": len(choices),
                        "error": type(error).__name__,
                        "error_kind": str(error).split(":", 1)[0][:80],
                        "parsed_keys": list(parsed)[:10],
                        "option_evidence_type": type(parsed.get("option_evidence")).__name__,
                    }
                ),
                flush=True,
            )
            correction = {
                "role": "user",
                "content": (
                    "The response is invalid. Return one corrected "
                    f"JSON object with exactly {len(choices)} option_evidence "
                    "entries in the displayed order, followed by one valid "
                    "evidence_sufficiency and sufficiency_reason. Do not add, "
                    "remove, merge, or repeat options."
                ),
            }
            messages = base_messages + (
                [{"role": "assistant", "content": raw}] if raw else []
            ) + [correction]
    raise ValueError("Evidence review failed after 3 attempts") from last_error


def adjudicate(
    model: Any,
    tokenizer: Any,
    question: str,
    choices: list[str],
    task: str,
    level_results: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    evidence_input = []
    for row in level_results:
        review = row["sufficiency_review"]
        audit = normalize_option_evidence(
            review["option_evidence_audit"], choices
        )
        review["option_evidence_audit"] = audit
        evidence_input.append(
            {
                "magnification": row["magnification"],
                "option_evidence_audit": audit,
                "evidence_sufficiency": review["evidence_sufficiency"],
                "sufficiency_reason": review["sufficiency_reason"],
            }
        )

    system = ARBITER_SYSTEM_PROMPT
    user = (
        f"Question: {question}\n"
        f"Options:\n{option_table(choices)}\n\n"
        f"Evidence reviews by magnification:\n"
        f"{json.dumps(evidence_input, ensure_ascii=False, indent=2)}\n\n"
        "Return only answer_label and reason. answer_label must be one displayed "
        "option letter. The reason must cite decisive positive findings and their "
        "magnifications. Do not repeat answer_text or justify by vote counts."
    )
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    messages = base_messages
    last_error: ValueError | None = None
    for attempt in range(1, 4):
        raw = ""
        try:
            with task_scope("final_adjudication", question, choices):
                raw, parsed = context_checked_call(
                    model, tokenizer, messages, OUTPUT_TOKEN_BUDGET
                )
            final = {
                **canonical_answer(parsed, choices),
                "reason": str(parsed.get("reason") or "").strip(),
            }
            return raw, final
        except ValueError as error:
            last_error = error
            print(
                json.dumps({
                    "event": "vqa_arbiter_retry",
                    "attempt": attempt,
                    "error": type(error).__name__,
                }),
                flush=True,
            )
            correction = {
                "role": "user",
                "content": (
                    "The final response is invalid. Return one JSON object with "
                    "answer_label set to exactly one displayed option letter "
                    "and reason set to a concise string. Return no other text."
                ),
            }
            messages = base_messages + (
                [{"role": "assistant", "content": raw}] if raw else []
            ) + [correction]
    raise ValueError("Final adjudication failed after 3 attempts") from last_error


def main() -> None:
    args = parse_args()
    cases = resolve_cases(args)
    output_dir = Path(args.output_dir) / f"shard_{args.shard_id}"
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output_dir) / "cache"

    valid_cases = []
    invalid_cases = []
    for case in cases:
        navigation_text = str(case.get("navigation_path") or "")
        if not navigation_text:
            invalid_cases.append(
                {"slide_id": case.get("slide_id"), "error": "missing_navigation"}
            )
            continue
        navigation_path = Path(navigation_text)
        try:
            navigation, counts = load_and_validate_navigation(navigation_path)
            qid = str(
                case.get("question_id")
                or question_id(str(case["slide_id"]), str(case["question"]))
            )
            valid_cases.append(
                {
                    **case,
                    "question_id": qid,
                    "navigation_path": str(navigation_path),
                    "navigation": navigation,
                    "focus_counts": counts,
                }
            )
        except (OSError, TypeError, ValueError) as error:
            invalid_cases.append(
                {"slide_id": case.get("slide_id"), "error": type(error).__name__}
            )

    print(
        json.dumps(
            {
                "event": "input_validation",
                "selected_cases": len(cases),
                "valid_cases": len(valid_cases),
                "invalid_cases": len(invalid_cases),
                "invalid_examples": invalid_cases[:5],
                "focus_images": sum(
                    sum(case["focus_counts"].values()) for case in valid_cases
                ),
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not valid_cases:
        raise ValueError("No valid cases remain after navigation validation")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.vlm_ckpt)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_ckpt,
        torch_dtype="auto",
        device_map=device_map(args.vlm_device),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckpt)
    llm_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": device_map(args.llm_device),
    }
    if args.max_memory_gib is not None and args.llm_device == "auto":
        llm_kwargs["max_memory"] = {
            index: f"{args.max_memory_gib}GiB"
            for index in range(torch.cuda.device_count())
        }
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_ckpt, **llm_kwargs
    ).eval()

    for index, raw_case in enumerate(valid_cases, start=1):
        case = prepare_source(raw_case)
        result_path = output_dir / f"{case['question_id']}.json"
        if result_path.is_file() and not args.overwrite:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("final_adjudication"):
                print(
                    f"[SkipComplete] {index}/{len(valid_cases)} {case['question_id']}",
                    flush=True,
                )
                continue
        result = {
            "question_id": case["question_id"],
            "slide_id": case["slide_id"],
            "question": case["question"],
            "choices": case["choices"],
            "ground_truth": case.get("ground_truth"),
            "task": case.get("task"),
            "stages": [],
            "level_results": [],
        }
        started = time.perf_counter()
        try:
            for level_key, magnification in LEVELS:
                if not validate_navigation_level(
                    case["navigation"], Path(case["navigation_path"]), level_key
                ):
                    result["stopping_reason"] = "no_next_observation"
                    break
                descriptions = collect_descriptions(
                    case["navigation"],
                    Path(case["navigation_path"]),
                    level_key,
                    case,
                    cache_dir,
                    processor,
                    vlm,
                    args.vlm_max_new_tokens,
                )
                stage = {
                    "magnification": magnification,
                    "descriptions": descriptions,
                }
                result["stages"].append(stage)
                grouped = group_focus_descriptions(descriptions)
                aggregation_raw, aggregation = aggregate_regions(
                    llm,
                    tokenizer,
                    case["question"],
                    case["choices"],
                    magnification,
                    grouped,
                    OUTPUT_TOKEN_BUDGET,
                )
                answer_raw, answer = answer_level(
                    llm,
                    tokenizer,
                    case["question"],
                    case["choices"],
                    "",
                    magnification,
                    aggregation["region_summaries"],
                    OUTPUT_TOKEN_BUDGET,
                )
                review_raw, review = review_sufficiency(
                    llm,
                    tokenizer,
                    case["question"],
                    case["choices"],
                    magnification,
                    aggregation["region_summaries"],
                    answer,
                    OUTPUT_TOKEN_BUDGET,
                )
                result["level_results"].append(
                    {
                        "level_key": level_key,
                        "magnification": magnification,
                        "source_description_count": len(descriptions),
                        "region_aggregation": aggregation,
                        "region_aggregation_raw_output": aggregation_raw,
                        "independent_answer": answer,
                        "independent_answer_raw_output": answer_raw,
                        "sufficiency_review": review,
                        "sufficiency_review_raw_output": review_raw,
                    }
                )
                save_result(result_path, result)
                print(
                    f"[Level] {index}/{len(valid_cases)} {case['question_id']} "
                    f"{magnification}x answer={answer['answer']} "
                    f"sufficiency={review['evidence_sufficiency']}",
                    flush=True,
                )
                if review["evidence_sufficiency"] == "sufficient":
                    result["stopping_reason"] = "sufficient_evidence"
                    break

            # An insufficient final review ends at the configured scale budget.
            result.setdefault("stopping_reason", "magnification_budget_exhausted")
            final_raw, final = adjudicate(
                llm,
                tokenizer,
                case["question"],
                case["choices"],
                "",
                result["level_results"],
                OUTPUT_TOKEN_BUDGET,
            )
            result["final_adjudication"] = final
            result["final_adjudication_raw_output"] = final_raw
            result["pred_answer"] = final["answer"]
            result["pred_answer_label"] = final["answer_label"]
            result["pred_answer_text"] = final["answer_text"]
            finalize_result(result, case, final)
            result["inference_seconds"] = time.perf_counter() - started
            save_result(result_path, result)
            print(
                f"[Final] {index}/{len(valid_cases)} {case['question_id']} "
                f"answer={final['answer']}",
                flush=True,
            )
        except Exception as error:
            result["status"] = "failed"
            result["error"] = type(error).__name__
            save_result(result_path, result)
            print(
                f"[Failed] {index}/{len(valid_cases)} {case['question_id']} "
                f"{result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
