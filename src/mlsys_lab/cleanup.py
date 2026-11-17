"""Cleanup-only primitive for a known allocation loaded from a trusted journal.

No create, stop, volume deletion or CLI activation. The future controller/reaper
must provide durable journal callbacks and verified GitHub-run ownership handover.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field
from runpod_ci_broker.cleanup import CleanupClient
from runpod_ci_broker.provider import ProviderError

from .budget import CAMPAIGN
from .results import Strict
from .worker import Runtime


class Allocation(Strict):
    runtime: Runtime
    actions_run_id: Annotated[str, Field(pattern=r"^[1-9][0-9]{0,19}$")]
    actions_run_attempt: Annotated[int, Field(ge=1)]
    data_center: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")]
    cloud: Literal["SECURE", "COMMUNITY"]
    create_intent: Literal[True]

    @property
    def name(self) -> str:
        raw = f"{CAMPAIGN}:{self.runtime.job.attempt}"
        return "mlsys-" + hashlib.sha256(raw.encode()).hexdigest()[:24]

    def markers(self) -> dict[str, str]:
        return {
            "MLS_REPOSITORY": self.runtime.job.repository,
            "MLS_CAMPAIGN": CAMPAIGN,
            "MLS_ATTEMPT": self.runtime.job.attempt,
            "MLS_RUN": self.actions_run_id,
            "MLS_RUN_ATTEMPT": str(self.actions_run_attempt),
            "MLS_SOURCE": self.runtime.job.source_sha,
        }


class Receipt(Strict):
    allocation: Allocation
    deleted: bool = False
    verification: Literal["pending", "ownership_refused", "termination_disabled", "absent_twice"]
    delete_requests: Annotated[int, Field(ge=0, le=3)] = 0
    absence_checks: Annotated[int, Field(ge=0, le=2)] = 0


def matches(pod: dict, allocation: Allocation) -> bool:
    runtime = allocation.runtime
    return (
        pod.get("id") == runtime.pod_id
        and pod.get("name") == allocation.name
        and pod.get("image") == runtime.job.image
        and pod.get("mounts") == {}  # Persistent volumes are outside this scope.
        and isinstance(pod.get("gpu"), dict)
        and pod["gpu"].get("id") == runtime.job.gpu
        and type(pod["gpu"].get("count")) is int
        and pod["gpu"]["count"] == 1
        and pod.get("cloud") == allocation.cloud
        and pod.get("dataCenterId") == allocation.data_center
        and isinstance(pod.get("env"), dict)
        and all(pod["env"].get(k) == v for k, v in allocation.markers().items())
    )


def cleanup_owned(
    allocation: Allocation,
    api: CleanupClient,
    *,
    load_trusted: Callable[[], Allocation],
    checkpoint: Callable[[Receipt], None],
    allow_termination: bool = False,
    timeout: float = 30,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Receipt:
    """Re-read trusted identity before DELETE; require two independent absence pairs.

    Callbacks must complete their own CAS/readback within the controller budget.
    Passing an arbitrary local JSON document is not a trusted journal handover.
    Failed checkpoints abort before DELETE; unconfirmed deletion remains pending.
    """
    if not 0 < timeout <= 120 or load_trusted() != allocation:
        raise ValueError("trusted_allocation_required")
    deadline = now() + timeout
    receipt = Receipt(allocation=allocation, verification="pending")
    checkpoint(receipt.model_copy(deep=True))
    for _ in range(5):
        if now() >= deadline:
            break
        try:
            if load_trusted() != allocation:
                raise ValueError("allocation_changed")
            pod = api.get(allocation.runtime.pod_id, deadline - now())
            if pod is None:
                ids = api.active_ids(deadline)
                if now() >= deadline:
                    break
                receipt.absence_checks = (
                    receipt.absence_checks + 1 if allocation.runtime.pod_id not in ids else 0
                )
                if receipt.absence_checks == 2:
                    receipt.deleted, receipt.verification = True, "absent_twice"
                    break
            else:
                receipt.absence_checks = 0
                if not matches(pod, allocation):
                    receipt.verification = "ownership_refused"
                    break
                if not allow_termination:
                    receipt.verification = "termination_disabled"
                    break
                if receipt.delete_requests >= 3:
                    break
                # Checkpoint intent durably before a destructive request. A
                # provider timeout does not prove failure; the next pass GETs.
                receipt.delete_requests += 1
                checkpoint(receipt.model_copy(deep=True))
                if load_trusted() != allocation or now() >= deadline:
                    raise ValueError("allocation_or_deadline_changed")
                api.delete(allocation.runtime.pod_id, deadline - now())
        except ProviderError as exc:
            receipt.absence_checks = 0
            if not (exc.retryable or exc.ambiguous):
                break
        checkpoint(receipt.model_copy(deep=True))
        if now() < deadline:
            sleep(min(1, deadline - now()))
    checkpoint(receipt.model_copy(deep=True))
    return receipt
