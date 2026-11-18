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
