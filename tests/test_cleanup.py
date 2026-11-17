import copy

import pytest
from runpod_ci_broker.provider import ProviderError
from test_worker_transport import runtime  # noqa: F401

from mlsys_lab.cleanup import Allocation, cleanup_owned


@pytest.fixture
def allocation(runtime):  # noqa: F811
    return Allocation(
        runtime=runtime,
        actions_run_id="123",
        actions_run_attempt=1,
        data_center="US-TEST-1",
        cloud="SECURE",
        create_intent=True,
    )


class API:
    def __init__(self, allocation):
        self.pod = {
            "id": allocation.runtime.pod_id,
            "name": allocation.name,
            "image": allocation.runtime.job.image,
            "mounts": {},
            "gpu": {"id": allocation.runtime.job.gpu, "count": 1},
            "cloud": allocation.cloud,
            "dataCenterId": allocation.data_center,
            "env": allocation.markers(),
        }
        self.deletes = 0
        self.inventory_calls = 0
        self.lost_delete_ack = False
        self.inventory_stale = False
        self.incomplete = False
        self.pod_id = allocation.runtime.pod_id

    def get(self, pod_id, timeout):
        assert pod_id == self.pod_id and timeout > 0
        return copy.deepcopy(self.pod)

    def delete(self, pod_id, timeout):
        self.deletes += 1
        self.pod = None
        if self.lost_delete_ack:
            raise ProviderError("lost_ack", ambiguous=True)

    def active_ids(self, deadline):
        self.inventory_calls += 1
        if self.incomplete:
            raise ProviderError("inventory_incomplete")
        return {self.pod_id} if self.inventory_stale else set()


def run(allocation, api, **kwargs):
    return cleanup_owned(
        allocation,
        api,
        load_trusted=lambda: allocation,
        checkpoint=lambda _: None,
        sleep=lambda _: None,
        **kwargs,
    )


def test_delete_receipt_alone_never_proves_cleanup(allocation):
    api = API(allocation)
    receipt = run(allocation, api, allow_termination=True)
    assert receipt.deleted and receipt.absence_checks == 2
    assert api.deletes == 1 and api.inventory_calls == 2


def test_termination_disabled_by_default(allocation):
    api = API(allocation)
    receipt = run(allocation, api)
    assert not receipt.deleted and api.deletes == 0
    assert receipt.verification == "termination_disabled"


@pytest.mark.parametrize(
    "field", ["id", "name", "image", "mounts", "gpu", "cloud", "dataCenterId", "env"]
)
def test_mismatched_ownership_never_deletes(allocation, field):
    api = API(allocation)
    api.pod[field] = {"unexpected": "value"} if field in {"mounts", "gpu", "env"} else "wrong"
    receipt = run(allocation, api, allow_termination=True)
    assert not receipt.deleted and api.deletes == 0
    assert receipt.verification == "ownership_refused"


def test_ambiguous_delete_reconciles_before_retry(allocation):
    api = API(allocation)
    api.lost_delete_ack = True
    receipt = run(allocation, api, allow_termination=True)
    assert receipt.deleted and api.deletes == 1


@pytest.mark.parametrize("kind", ["stale", "incomplete"])
def test_direct_404_with_unknown_inventory_is_not_success(allocation, kind):
    api = API(allocation)
    api.inventory_stale = kind == "stale"
    api.incomplete = kind == "incomplete"
    receipt = run(allocation, api, allow_termination=True)
    assert not receipt.deleted and receipt.verification == "pending"


def test_failed_journal_checkpoint_prevents_delete(allocation):
    api = API(allocation)

    def checkpoint(receipt):
        if receipt.delete_requests:
            raise ValueError("CAS conflict")

    with pytest.raises(ValueError, match="CAS"):
        cleanup_owned(
            allocation,
            api,
            load_trusted=lambda: allocation,
            checkpoint=checkpoint,
            allow_termination=True,
        )
    assert api.deletes == 0


def test_changed_run_owner_prevents_delete(allocation):
    api = API(allocation)
    reads = [allocation, allocation, allocation.model_copy(update={"actions_run_id": "456"})]
    with pytest.raises(ValueError, match="changed"):
        cleanup_owned(
            allocation,
            api,
            load_trusted=lambda: reads.pop(0),
            checkpoint=lambda _: None,
            allow_termination=True,
        )
    assert api.deletes == 0
