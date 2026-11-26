import json

import pytest
from runpod_ci_broker.ledger import SQLiteStore
from runpod_ci_broker.provider import ProviderError
from test_cli_preflight import ReadOnlyFake

from mlsys_lab import controller as c

SHA = "a" * 40
IMAGE = "sha256:" + "b" * 64


class API:
    def __init__(self):
        self.pod = None
        self.ids = []
        self.lose_ack = False

    def inventory(self):
        return [self.pod] if self.pod else []

    def create(self, body):
        self.ids.append("pod123")
        self.pod = {
            **body,
            "id": "pod123",
            "dataCenterId": "actual-dc",
            "status": "RUNNING",
            "cost": 0.5,
        }
        if self.lose_ack:
            raise ProviderError("timeout", ambiguous=True)
        return "pod123"

    def get(self, pod_id, timeout):
        return self.pod

    def delete(self, pod_id, timeout):
        assert self.pod["id"] == pod_id
        self.pod = None

    def active_ids(self, deadline):
        return {self.pod["id"]} if self.pod else set()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path / "budget.sqlite3")
    c.Budget(store).initialize()
    output = tmp_path / "bundle"

    def collect(runtime, reader, path, timeout):
        path.mkdir()
        (path / "manifest.json").write_text("fixture manifest")
        if runtime.job.experiment == "E01":
            (path / "summary.json").write_text('{"results": {}}')

    monkeypatch.setattr(c, "collect", collect)
    return store, API(), output


@pytest.mark.parametrize("upload", ["verified", "missing", "foreign"])
def test_fresh_runner_settles_from_durable_evidence(setup, upload):
    store, api, output = setup
    j = c.Journal(store, "123", "1")
    c.launch(j, api, ReadOnlyFake(), SHA, IMAGE, output, 2000, 1000)
    assert j.record["center"] == "actual-dc"
    assert api.pod["env"]["MLS_RESULT_TOKEN"] not in json.dumps(j.record)
    assert set(api.pod["env"]).isdisjoint({"RUNPOD_API_KEY", "RCB_LEDGER_TOKEN", "GITHUB_TOKEN"})
    artifact = {
        "digest": IMAGE,
        "id": 1234,
        "name": j.attempt,
        "expired": False,
        "workflow_run": {"id": 123, "head_sha": SHA},
    }
    if upload == "verified":
        c.receipt(j, artifact, "1234", IMAGE, output)
    if upload == "foreign":
        artifact["workflow_run"]["id"] = 321
        with pytest.raises(ValueError, match="receipt_invalid"):
            c.receipt(j, artifact, "1234", IMAGE, output)
    result = c.recover(c.Journal(store, "123", "1"), api)
    assert result == ("accepted" if upload == "verified" else "rejected")
    assert not api.pod and c.recover(c.Journal(store, "123", "1"), api) == result
    assert j.budget.load()[1].reservations[j.attempt].cleanup_evidence


def test_lost_create_response_reconciles_without_duplicate_post(setup):
    store, api, output = setup
    j = c.Journal(store, "123", "1")
    api.lose_ack = True
    with pytest.raises(ProviderError):
        c.launch(j, api, ReadOnlyFake(), SHA, IMAGE, output, 1000, 500)
    with pytest.raises(ValueError, match="already_exists"):
        c.launch(c.Journal(store, "123", "1"), api, ReadOnlyFake(), SHA, IMAGE, output, 1000, 500)
    assert c.recover(c.Journal(store, "123", "1"), api) == "rejected"
    assert len(api.ids) == 1 and not api.pod


def test_ambiguous_create_with_empty_inventory_keeps_budget_locked(setup):
    store, api, output = setup
    api.lose_ack = True
    j = c.Journal(store, "123", "1")
    with pytest.raises(ProviderError):
        c.launch(j, api, ReadOnlyFake(), SHA, IMAGE, output, 1000, 500)
    api.pod = None
    with pytest.raises(ValueError, match="creation_unresolved"):
        c.recover(c.Journal(store, "123", "1"), api)
    assert j.budget.load()[1].reservations[j.attempt].state != "closed"


def test_single_create_offers_only_eligible_equally_priced_centers(setup):
    store, api, output = setup

    class Catalog(ReadOnlyFake):
        def snapshot(self, task):
            rows, inventory = super().snapshot(task)
            original = rows[0]
            for changes in (
                {"data_center": "another-dc"},
                {"data_center": "premium-dc", "hourly_usd": 0.9},
                {"data_center": "other-gpu", "gpu_id": task.gpu_allowlist[1]},
                {"data_center": "old-cuda", "cuda_versions": ["12.7"]},
            ):
                rows.append(original.model_copy(update=changes))
            return rows, inventory

    journal = c.Journal(store, "123", "1")
    c.launch(journal, api, Catalog(), SHA, IMAGE, output, 2000, 1000)
    assert api.ids == ["pod123"]
    assert (
        api.pod["dataCenterIds"] == journal.record["eligible_centers"] == ["another-dc", "test-dc"]
    )


def test_e01_uses_course_budget_and_exact_worker(setup):
    store, api, output = setup
    j = c.Journal(store, "123", "1", "E01")
    c.launch(j, api, ReadOnlyFake(), SHA, c.e01_hosted.IMAGE_DIGEST, output, 5000, 10000)
    reserve = j.budget.load()[1].reservations["E01-123-1"]
    assert reserve.experiment == "E01" and reserve.gpu_seconds == 1200
    assert api.pod["image"] == "pytorch/pytorch@" + c.e01_hosted.IMAGE_DIGEST
    assert api.pod["disk"] == 30 and api.pod["mounts"] == {}
    assert api.pod["gpu"]["minCudaVersion"] == "12.8"
    assert not c.Journal(store, "123", "1").record
    assert c.recover(c.Journal(store, "123", "1", "E01"), api) == "rejected"
    assert api.pod is None


@pytest.mark.parametrize("scenario", ["lag", "other_writer", "never_visible"])
def test_journal_confirms_one_write_with_bounded_reads(setup, monkeypatch, scenario):
    store, *_ = setup
    journal = c.Journal(store, "123", "1", "E02")
    journal.save(phase="allocated")
    previous = (journal.version, journal.record)
    read, swap = store.read, store.swap
    calls = []
    writes = []

    def observe(key):
        calls.append(key)
        if scenario == "other_writer":
            return "different", {"phase": "different"}
        if scenario == "lag" and len(calls) == 3:
            return read(key)
        return previous

    def write(key, version, document):
        writes.append(key)
        swap(key, version, document)

    monkeypatch.setattr(store, "read", observe)
    monkeypatch.setattr(store, "swap", write)
    monkeypatch.setattr(c.time, "sleep", lambda delay: None)
    if scenario == "lag":
        journal.save(hourly_cost=0.5)
        assert journal.record["hourly_cost"] == 0.5
        assert len(calls) == 3
    else:
        with pytest.raises(ValueError, match="journal_write_unconfirmed"):
            journal.save(hourly_cost=0.5)
        assert len(calls) == (1 if scenario == "other_writer" else 4)
        assert (journal.version, journal.record) == previous
    assert writes == [journal.key]
