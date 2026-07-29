#!/usr/bin/env python3
"""Launch RH20T multi-view training."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def normalize_devices(value: str) -> list[str]:
  devices = [item.strip() for item in str(value).split(",") if item.strip()]
  if not devices:
    raise ValueError("--device must contain at least one CUDA device id.")
  return devices


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_cfg_path", required=True, type=Path)
  parser.add_argument(
      "--device",
      default="0",
      help='CUDA device ids. Example: "0" or "0,1,2,3".',
  )
  parser.add_argument(
      "overrides",
      nargs=argparse.REMAINDER,
      help="Optional arguments forwarded to scripts/train_multiview_softdtw.py.",
  )
  args = parser.parse_args()

  devices = normalize_devices(args.device)
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

  train_script = Path(__file__).resolve().parent / "scripts/train_multiview_softdtw.py"
  forwarded = list(args.overrides)
  if forwarded and forwarded[0] == "--":
    forwarded = forwarded[1:]

  if len(devices) == 1:
    cmd = [
        sys.executable,
        "-u",
        str(train_script),
        "--config",
        str(args.exp_cfg_path),
        "--device",
        "cuda:0",
        *forwarded,
    ]
  else:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        str(train_script),
        "--config",
        str(args.exp_cfg_path),
        *forwarded,
    ]

  print(
      "CUDA_VISIBLE_DEVICES="
      f"{env['CUDA_VISIBLE_DEVICES']} "
      + " ".join(cmd),
      flush=True,
  )
  raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
  main()
