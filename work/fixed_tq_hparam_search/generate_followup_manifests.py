#!/usr/bin/env python3
"""Generate interpretable follow-up grids from explicitly selected anchors."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path


FIELDS = [
    "run_name",
    "backbone",
    "rho",
    "epsilon",
    "lr",
    "max_forward_step",
    "lambda_q",
    "lambda_sa",
    "lambda_mv",
]


def decimal(value: str | float) -> Decimal:
    return Decimal(str(value))


def plain(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return "0" if rendered in {"-0", ""} else rendered


def tag(value: Decimal) -> str:
    return plain(value).replace("-", "m").replace(".", "p")


def load_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "backbone",
        "rho",
        "epsilon",
        "lr",
        "max_forward_step",
        "lambda_q",
        "lambda_sa",
        "lambda_mv",
    }
    if not rows:
        raise ValueError("Selection is empty.")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Selection is missing columns: {sorted(missing)}")
    for row in rows:
        if row["backbone"] not in {"vit_imagenet", "r3m_bn_bi"}:
            raise ValueError(f"Unsupported backbone: {row['backbone']}")
    return rows


def make_name(stage: str, row: dict[str, Decimal | str]) -> str:
    backbone = row["backbone"]
    assert isinstance(backbone, str)
    return (
        f"hps{stage}_{backbone}"
        f"_rho{tag(row['rho'])}"
        f"_eps{tag(row['epsilon'])}"
        f"_lr{tag(row['lr'])}"
        f"_qstep{tag(row['max_forward_step'])}"
        f"_lq{tag(row['lambda_q'])}"
        f"_lsa{tag(row['lambda_sa'])}"
        f"_lmv{tag(row['lambda_mv'])}"
        "_i40000"
    )


def normalized(row: dict[str, str]) -> dict[str, Decimal | str]:
    return {
        "backbone": row["backbone"],
        "rho": decimal(row["rho"]),
        "epsilon": decimal(row["epsilon"]),
        "lr": decimal(row["lr"]),
        "max_forward_step": decimal(row["max_forward_step"]),
        "lambda_q": decimal(row["lambda_q"]),
        "lambda_sa": decimal(row["lambda_sa"]),
        "lambda_mv": decimal(row["lambda_mv"]),
    }


def generate_lr(rows: list[dict[str, str]]) -> list[dict[str, Decimal | str]]:
    generated = []
    for source in rows:
        anchor = normalized(source)
        for multiplier in map(decimal, ("0.5", "1", "2")):
            candidate = dict(anchor)
            candidate["lr"] = anchor["lr"] * multiplier
            generated.append(candidate)
    return generated


def require_one_per_backbone(
    rows: list[dict[str, str]], stage: str
) -> list[dict[str, Decimal | str]]:
    seen: set[str] = set()
    normalized_rows = []
    for source in rows:
        anchor = normalized(source)
        backbone = anchor["backbone"]
        assert isinstance(backbone, str)
        if backbone in seen:
            raise ValueError(f"{stage} requires one selected anchor per backbone.")
        seen.add(backbone)
        normalized_rows.append(anchor)
    return normalized_rows


def generate_q(rows: list[dict[str, str]]) -> list[dict[str, Decimal | str]]:
    generated = []
    for anchor in require_one_per_backbone(rows, "Q search"):
        for max_step in map(decimal, ("1", "3", "7")):
            for lambda_q in map(decimal, ("0.05", "0.1", "0.2")):
                candidate = dict(anchor)
                candidate["max_forward_step"] = max_step
                candidate["lambda_q"] = lambda_q
                generated.append(candidate)
    return generated


def generate_weights(rows: list[dict[str, str]]) -> list[dict[str, Decimal | str]]:
    generated = []
    for anchor in require_one_per_backbone(rows, "Weight search"):
        for lambda_mv in map(decimal, ("0.25", "0.5", "1")):
            candidate = dict(anchor)
            candidate["lambda_sa"] = decimal("1")
            candidate["lambda_mv"] = lambda_mv
            generated.append(candidate)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("lr", "q", "weights"))
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help=(
            "Optional manifest containing both the already evaluated selected "
            "anchor and the newly generated candidates."
        ),
    )
    args = parser.parse_args()

    selection = load_selection(args.selection)
    if args.stage == "lr":
        candidates = generate_lr(selection)
        stage_tag = "A2"
    elif args.stage == "q":
        candidates = generate_q(selection)
        stage_tag = "B"
    else:
        candidates = generate_weights(selection)
        stage_tag = "C"

    selected_anchors = [normalized(row) for row in selection]
    selected_keys = {
        tuple(str(anchor[field]) for field in FIELDS if field != "run_name")
        for anchor in selected_anchors
    }
    unique: dict[tuple[str, ...], dict[str, Decimal | str]] = {}
    for candidate in candidates:
        key = tuple(str(candidate[field]) for field in FIELDS if field != "run_name")
        if key in selected_keys:
            continue
        unique[key] = candidate

    def write_manifest(
        path: Path, output_rows: list[dict[str, Decimal | str]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            for candidate in output_rows:
                output = {
                    key: plain(value) if isinstance(value, Decimal) else value
                    for key, value in candidate.items()
                }
                if not output.get("run_name"):
                    output["run_name"] = make_name(stage_tag, candidate)
                writer.writerow(output)
        temporary.replace(path)

    generated_rows = list(unique.values())
    write_manifest(args.output, generated_rows)

    if args.comparison_output is not None:
        anchors_for_comparison: list[dict[str, Decimal | str]] = []
        for source, anchor in zip(selection, selected_anchors):
            selected_run_name = source.get("selected_run_name", "")
            if not selected_run_name:
                raise ValueError(
                    "--comparison-output requires selected_run_name in selection."
                )
            anchor_with_name = dict(anchor)
            anchor_with_name["run_name"] = selected_run_name
            anchors_for_comparison.append(anchor_with_name)
        write_manifest(
            args.comparison_output,
            [*anchors_for_comparison, *generated_rows],
        )


if __name__ == "__main__":
    main()
