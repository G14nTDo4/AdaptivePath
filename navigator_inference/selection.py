from __future__ import annotations

from typing import Any, Iterable


def select_top_regions(
    regions: Iterable[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Select up to top_k highest-scoring regions after tissue filtering."""
    eligible = [row for row in regions if row.get("filter_reason", "") == ""]
    return sorted(eligible, key=lambda row: float(row["abnormal_score"]), reverse=True)[:top_k]


def select_region_focuses(
    focuses: Iterable[dict[str, Any]],
    max_per_region: int = 4,
) -> list[dict[str, Any]]:
    """Keep up to max_per_region sequential observations without re-ranking."""
    eligible = [row for row in focuses if row.get("filter_reason", "") == ""]
    return sorted(eligible, key=lambda row: int(row["step_idx"]))[:max_per_region]
