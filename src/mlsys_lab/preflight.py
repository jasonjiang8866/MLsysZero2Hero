"""GET-only RunPod discovery using a pinned, previously exercised broker adapter.

The reference broker image describes a known CUDA 12.8 memory-smoke environment
for catalog queries. It is NOT the course PTX worker and is never launched here.
"""

from __future__ import annotations

import json
from pathlib import Path

from runpod_ci_broker.models import Task
from runpod_ci_broker.planner import reasons
from runpod_ci_broker.provider import ProviderError, ReadOnlyRunPod

BROKER_SHA = "5822d7f79051f8b4945a6c394743fcd1772d8405"
GPU_IDS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
]


def discovery_task() -> Task:
    return Task(
        repository="jasonjiang8866/MLsysZero2Hero",
        scope="mlsys-initial-20261116",
        experiment_id="E00-discovery",
        source_sha="567edcc32529ad017d13edbc154971aea4d9d624",
        image=(
            "ghcr.io/jasonjiang8866/runpod-ci-broker-worker@sha256:"
            "9226c6ae489e22f89f014cec7f2dfd80ebe1b9d661fcfe318807035c75f3ae03"
        ),
        entrypoint=["broker-cuda-memory-smoke"],
        strategy="flexible_smoke",
        gpu_allowlist=GPU_IDS,
        gpu_preference=GPU_IDS,
        compatibility={"min_cuda_version": "12.8"},
        limits={"max_gpu_hourly_usd": 3.0, "max_queries": 2, "max_wait_seconds": 10},
    )


def observe(output: Path, api: ReadOnlyRunPod | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "mode": "read_only",
        "paid_launch_enabled": False,
        "gpu_executed": False,
        "broker_source_sha": BROKER_SHA,
        "secret_present": False,
        "query_status": "not_queried",
        "candidates": [],
        "launch_blockers": [
            "course_worker_image_unpublished",
            "live_controller_and_independent_recovery_not_integrated",
            "durable_course_ledger_not_bootstrapped",
            "live_storage_transfer_quote_unverified",
            "specific_gpu_selection_pending",
        ],
        "notice": "Catalog only. Candidate eligibility is not authorization or a reservation.",
    }
    try:
        # Only GET operations exist on this adapter. Never import the paid controller.
        client = api if api is not None else ReadOnlyRunPod()
        report["secret_present"] = True if api is None else "injected_test_adapter"
        task = discovery_task()
        candidates, inventory = client.snapshot(task)
        # Project only sanitized, bounded fields; no raw Pod/env/credential payload.
        report["query_status"] = inventory.get("query_status", "unknown")
        report["inventory_status"] = inventory.get("inventory", "unverified")
        report["observed_at"] = inventory.get("observed_at")
        report["scope_active_pod_count"] = inventory.get("scope_active_pod_count")
        report["datacenter_status"] = inventory.get("datacenter_status", "unverified")
        report["candidates"] = [
            {
                "gpu": candidate.gpu_id,
                "data_center": candidate.data_center,
                "usd_per_gpu_hour": candidate.hourly_usd,
                "vram_gb": candidate.memory_gb,
                "cuda_versions": candidate.cuda_versions,
                "capacity": candidate.capacity,
                "reasons": reasons(task, candidate),
            }
            for candidate in candidates
        ]
        if report["inventory_status"] != "complete":
            report["launch_blockers"].append("inventory_unverified")
        if report["scope_active_pod_count"] != 0:
            report["launch_blockers"].append("scope_not_verified_empty")
    except ProviderError as exc:
        allowed = {
            "credentials_missing",
            "authentication_failed",
            "permission_denied",
            "account_restricted",
            "rate_limited",
            "query_failed",
            "provider_unavailable",
        }
        report["query_status"] = exc.code if exc.code in allowed else "provider_query_failed"
    except (ValueError, OSError, KeyError, TypeError):
        report["query_status"] = "preflight_failed_closed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Fixed text summary: untrusted provider strings are not injected into Markdown.
    (output / "summary.md").write_text(
        "# Course RunPod preflight\n\nGET-only observation; no paid resources created.\n\n"
        "Inspect report.json for catalog/inventory results. Paid execution remains disabled.\n"
    )
    return report
