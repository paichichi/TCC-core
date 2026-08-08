#!/usr/bin/env python3
"""Select one downstream-validated winner per backbone for the next stage."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PARAM_FIELDS = [
    "backbone",
    "rho",
    "epsilon",
    "lr",
    "max_forward_step",
    "lambda_q",
    "lambda_sa",
    "lambda_mv",
]
SELECTION_FIELDS = [
    *PARAM_FIELDS,
    "selected_run_name",
    "raw_best_run_name",
    "stage_label",
    "overall_mean",
    "overall_std",
    "level1_mean",
    "level1_std",
    "level2_mean",
    "level2_std",
    "runner_up_gap",
    "pooled_standard_error",
    "decisive_at_one_se",
    "within_one_se_count",
    "selection_rule",
]
AUDIT_FIELDS = [
    "stage_label",
    "backbone",
    "rank",
    "selected",
    "run_name",
    "overall_mean",
    "overall_std",
    "level1_mean",
    "level1_std",
    "level2_mean",
    "level2_std",
    "gap_to_best",
    "within_best_pooled_se",
]


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def finite_float(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}: non-finite {field}")
    return value


def validate_protocol(row: dict[str, str], source: str) -> None:
    expected = {
        "repeats": "3",
        "tasks_per_repeat": "12",
        "episodes_per_task": "10",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"{source}: expected {field}={value}, found {row.get(field)!r}"
            )


def canonical_distance(row: dict[str, object]) -> float:
    """Multiplicative distance from the preregistered conservative defaults.

    Hyperparameter search is noisy because each RVT2-lite candidate contains
    one policy-training seed.  When several candidates are statistically
    indistinguishable from the raw best, prefer the least changed setting
    rather than exploiting noise in the validation suite.
    """

    backbone = str(row["backbone"])
    canonical_lr = {
        "vit_imagenet": 7.5e-5,
        "r3m_bn_bi": 1.0e-4,
    }[backbone]
    defaults = {
        "rho": 0.5,
        "epsilon": 0.05,
        "lr": canonical_lr,
        "max_forward_step": 1.0,
        "lambda_q": 0.1,
        "lambda_sa": 1.0,
        "lambda_mv": 0.5,
    }
    distance = 0.0
    for field, default in defaults.items():
        value = float(row[field])
        if value <= 0 or default <= 0:
            raise ValueError(
                f"Canonical-distance fields must be positive: {field}={value}"
            )
        distance += abs(math.log2(value / default))
    return distance


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lite-results", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--stage-label", required=True)
    args = parser.parse_args()

    manifest = read_rows(args.manifest, delimiter="\t")
    lite = read_rows(args.lite_results)
    manifest_by_name = {row["run_name"]: row for row in manifest}
    if len(manifest_by_name) != len(manifest):
        raise ValueError(f"{args.manifest}: duplicate run_name")
    lite_by_name = {row["run_name"]: row for row in lite}

    missing = sorted(set(manifest_by_name) - set(lite_by_name))
    if missing:
        raise ValueError(
            f"Stage {args.stage_label} is incomplete: "
            f"{len(missing)} RVT2-lite result(s) missing: {missing}"
        )

    grouped: dict[str, list[dict[str, object]]] = {}
    for run_name, params in manifest_by_name.items():
        source = f"{args.lite_results}:{run_name}"
        result = lite_by_name[run_name]
        validate_protocol(result, source)
        row: dict[str, object] = {
            **params,
            "run_name": run_name,
        }
        for field in (
            "overall_mean",
            "overall_std",
            "level1_mean",
            "level1_std",
            "level2_mean",
            "level2_std",
        ):
            row[field] = finite_float(result, field, source)
        backbone = params["backbone"]
        if backbone not in {"vit_imagenet", "r3m_bn_bi"}:
            raise ValueError(f"{args.manifest}: unsupported backbone {backbone!r}")
        grouped.setdefault(backbone, []).append(row)

    if set(grouped) != {"vit_imagenet", "r3m_bn_bi"}:
        raise ValueError("Manifest must contain both backbone families.")

    selections: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for backbone in ("vit_imagenet", "r3m_bn_bi"):
        candidates = sorted(
            grouped[backbone],
            key=lambda row: (
                -float(row["overall_mean"]),
                -float(row["level2_mean"]),
                float(row["overall_std"]),
                str(row["run_name"]),
            ),
        )
        raw_best = candidates[0]
        best_se = float(raw_best["overall_std"]) / math.sqrt(3)
        if len(candidates) > 1:
            runner_up = candidates[1]
            runner_up_gap = float(raw_best["overall_mean"]) - float(
                runner_up["overall_mean"]
            )
            pooled_se = math.sqrt(
                best_se**2
                + (float(runner_up["overall_std"]) / math.sqrt(3)) ** 2
            )
        else:
            runner_up_gap = math.inf
            pooled_se = 0.0

        within_one_se: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_se = float(candidate["overall_std"]) / math.sqrt(3)
            gap = float(raw_best["overall_mean"]) - float(
                candidate["overall_mean"]
            )
            comparison_se = math.sqrt(best_se**2 + candidate_se**2)
            if gap <= comparison_se:
                within_one_se.append(candidate)

        # One-standard-error rule: only exploit an apparent performance gain
        # when it exceeds evaluation noise.  Otherwise retain the candidate
        # closest to the preregistered defaults.  Remaining keys make the
        # decision deterministic and auditable.
        best = min(
            within_one_se,
            key=lambda row: (
                canonical_distance(row),
                -float(row["overall_mean"]),
                -float(row["level2_mean"]),
                float(row["overall_std"]),
                str(row["run_name"]),
            ),
        )
        selected = {
            field: best[field] for field in PARAM_FIELDS
        }
        selected.update(
            {
                "selected_run_name": best["run_name"],
                "raw_best_run_name": raw_best["run_name"],
                "stage_label": args.stage_label,
                "overall_mean": best["overall_mean"],
                "overall_std": best["overall_std"],
                "level1_mean": best["level1_mean"],
                "level1_std": best["level1_std"],
                "level2_mean": best["level2_mean"],
                "level2_std": best["level2_std"],
                "runner_up_gap": runner_up_gap,
                "pooled_standard_error": pooled_se,
                "decisive_at_one_se": runner_up_gap > pooled_se,
                "within_one_se_count": len(within_one_se),
                "selection_rule": (
                    "raw_best"
                    if best["run_name"] == raw_best["run_name"]
                    else "one_se_then_canonical_default"
                ),
            }
        )
        selections.append(selected)

        for rank, candidate in enumerate(candidates, start=1):
            candidate_se = float(candidate["overall_std"]) / math.sqrt(3)
            gap = float(raw_best["overall_mean"]) - float(
                candidate["overall_mean"]
            )
            comparison_se = math.sqrt(best_se**2 + candidate_se**2)
            audit.append(
                {
                    "stage_label": args.stage_label,
                    "backbone": backbone,
                    "rank": rank,
                    "selected": candidate["run_name"] == best["run_name"],
                    "run_name": candidate["run_name"],
                    "overall_mean": candidate["overall_mean"],
                    "overall_std": candidate["overall_std"],
                    "level1_mean": candidate["level1_mean"],
                    "level1_std": candidate["level1_std"],
                    "level2_mean": candidate["level2_mean"],
                    "level2_std": candidate["level2_std"],
                    "gap_to_best": gap,
                    "within_best_pooled_se": gap <= comparison_se,
                }
            )

    write_tsv(args.selection_output, selections, SELECTION_FIELDS)
    write_tsv(args.audit_output, audit, AUDIT_FIELDS)


if __name__ == "__main__":
    main()
