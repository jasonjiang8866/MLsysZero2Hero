"""Run local bounded smoke, validate a bundle, or query cloud capacity read-only."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .preflight import observe
from .results import Manifest, seal, verify


def run_smoke(output: Path, source: str, image: str | None, timeout: int) -> None:
    # Validate metadata BEFORE initializing CUDA, and never overwrite prior results.
    Manifest(source_sha=source, image_digest=image, files={})
    if not 1 <= timeout <= 60:
        raise ValueError("timeout_invalid")
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "LANG",
        )
        if key in os.environ
    }
    result = subprocess.run(
        [sys.executable, "-m", "mlsys_lab.cuda_smoke", str(output.resolve())],
        env=environment,
        timeout=timeout,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise ValueError("cuda_child_failed")
    seal(output, source, image)
    verify(output, source, image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--source-sha", required=True)
        sub.add_argument("--image-digest")
        if command == "smoke":
            sub.add_argument("--timeout-seconds", type=int, default=30)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            report = observe(args.output)
            passed = (
                report["query_status"] == "ok"
                and report.get("inventory_status") == "complete"
                and report.get("datacenter_status") == "ok"
            )
            print(
                json.dumps(
                    {"mode": "read_only", "query_complete": passed, "paid_launch_enabled": False}
                )
            )
            return 0 if passed else 2
        if not re.fullmatch(r"[a-f0-9]{40}", args.source_sha):
            raise ValueError("source_invalid")
        if args.command == "smoke":
            run_smoke(args.output, args.source_sha, args.image_digest, args.timeout_seconds)
        else:
            verify(args.output, args.source_sha, args.image_digest)
        print('{"kernel_verified":true,"cloud_acceptance":"not_evaluated"}')
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired):
        # Do not print exception details, provider bodies, child logs or environments.
        print('{"status":"failed","kernel_verified":false,"code":"lab_failed_closed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
