import json
import subprocess
import sys

import pytest
from runpod_ci_broker.models import Candidate

from mlsys_lab.cli import run_smoke
from mlsys_lab.preflight import discovery_task, observe


class ReadOnlyFake:
    def __init__(self, inventory="complete", scope_count=0, versions=None):
        self.inventory, self.scope_count = inventory, scope_count
        self.versions = versions or ["13.0"]

    def snapshot(self, task):
        return [
            Candidate(
                gpu_id=task.gpu_allowlist[0],
                data_center="test-dc",
                memory_gb=24,
                hourly_usd=0.5,
                capacity="available",
                cuda_versions=self.versions,
                evidence="read_only",
            )
        ], {
            "query_status": "ok",
            "inventory": self.inventory,
            "scope_active_pod_count": self.scope_count,
            "datacenter_status": "ok",
            "private_env": {"RUNPOD_API_KEY": "never-print-this"},
        }


def test_read_only_discovery_never_enables_launch(tmp_path):
    report = observe(tmp_path / "out", ReadOnlyFake())
    assert report["candidates"][0]["reasons"] == []
    assert report["paid_launch_enabled"] is False and report["gpu_executed"] is False
    assert "never-print-this" not in json.dumps(report)
    assert report["launch_blockers"]


def test_broker_campaign_is_not_reused():
    task = discovery_task()
    assert task.repository.endswith("MLsysZero2Hero")
    assert task.scope == "mlsys-initial-20261116"
    assert task.limits.total_budget_usd is None
    assert task.limits.prior_spend_usd == 0


@pytest.mark.parametrize(
    "inventory,count,reason",
    [
        ("unverified", 0, "inventory_unverified"),
        ("complete", 1, "scope_not_verified_empty"),
    ],
)
def test_unknown_or_occupied_inventory_blocks(tmp_path, inventory, count, reason):
    report = observe(tmp_path / "out", ReadOnlyFake(inventory, count))
    assert reason in report["launch_blockers"]


@pytest.mark.parametrize(
    "version,reason",
    [
        ("12.7", ["cuda_incompatible"]),
        ("12.8", []),
        ("12.9", []),
        ("13.0", []),
    ],
)
def test_catalog_cuda_floor(tmp_path, version, reason):
    report = observe(tmp_path / "out", ReadOnlyFake(versions=[version]))
    assert report["candidates"][0]["reasons"] == reason


def test_missing_key_is_sanitized_and_does_not_query(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    report = observe(tmp_path / "out")
    assert report["query_status"] == "credentials_missing"
    assert report["secret_present"] is False


def test_child_never_receives_account_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "private-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "also-private")

    def child(args, **kwargs):
        assert "RUNPOD_API_KEY" not in kwargs["env"]
        assert "GITHUB_TOKEN" not in kwargs["env"]
        assert kwargs["timeout"] == 30
        return subprocess.CompletedProcess(args, 2)

    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(ValueError, match="child"):
        run_smoke(tmp_path / "out", "a" * 40, None, 30)
    assert not (tmp_path / "out/manifest.json").exists()


def test_child_timeout_never_seals_result(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("child", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        run_smoke(tmp_path / "out", "a" * 40, None, 1)
    assert not (tmp_path / "out/manifest.json").exists()


def test_failed_run_cannot_reuse_previous_directory(tmp_path):
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path, "a" * 40, None, 30)


def test_cli_errors_do_not_echo_sensitive_inputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlsys_lab.cli",
            "smoke",
            "--output",
            str(tmp_path / "out"),
            "--source-sha",
            "private-secret",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "private-secret" not in result.stdout + result.stderr
    assert not (tmp_path / "out").exists()
