"""CAS-backed project reservations. No RunPod operations or automatic initialization.

The broker Store interface supplies SQLite (offline) or GitHub Contents (remote).
Remote deployment/recovery is a separate integration gate. Reservations are retained,
not refunded on successful cleanup or failed creates; all retries consume the same caps.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from runpod_ci_broker.ledger import LedgerError, Store

REPOSITORY = "jasonjiang8866/MLsysZero2Hero"
CAMPAIGN = "mlsys-initial-20261116"
KEY = hashlib.sha256(f"{REPOSITORY}:{CAMPAIGN}".encode()).hexdigest()
MONEY_CAP = 20_000_000  # Integer micro-USD; no binary floating point accounting.
TIME_CAP = 60_000  # Aggregate allocated GPU-seconds, not wall time.
# Keep $2/30 GPU-minutes reserved for recovery, outside normal admissions.
WORK_MONEY_CAP = 18_000_000
WORK_TIME_CAP = TIME_CAP - 30 * 60
ALLOCATIONS = {
    "E00": (1_000_000, 200 * 60),
    "E01": (3_000_000, 9000),
    "E02": (3_000_000, 10800),
    "E03": (3_000_000, 9600),
    "E04": (4_000_000, 9600),
    "E05": (2_000_000, 4800),
    "E06": (2_000_000, 2400),
}
Pos = Annotated[int, Field(gt=0)]
Nat = Annotated[int, Field(ge=0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Quote(Strict):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)
    observed_at: float
    compute_micro_usd_per_gpu_hour: Pos
    storage_micro_usd_per_pod_hour: Nat
    extras_micro_usd: Nat
    gpu_count: Annotated[int, Field(ge=1, le=4)]
    wall_seconds: Annotated[int, Field(ge=1, le=1800)]
    # wall_seconds includes startup, compute, collection and cleanup.

    @property
    def gpu_seconds(self) -> int:
        return self.gpu_count * self.wall_seconds

    @property
    def reserved_micro_usd(self) -> int:
        rate = (
            self.compute_micro_usd_per_gpu_hour * self.gpu_count
            + self.storage_micro_usd_per_pod_hour
        )
        return (rate * self.wall_seconds + 3599) // 3600 + self.extras_micro_usd


class Reservation(Strict):
    experiment: str
    quote: Quote
    reserved_micro_usd: Pos
    gpu_seconds: Pos
    source_sha: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
    image_digest: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    state: Literal["reserved", "creation_unknown", "closed"] = "reserved"
    outcome: Literal["pending", "accepted", "rejected", "not_created"] = "pending"
    cleanup_evidence: str | None = None
    observed_micro_usd: Nat = 0
    observed_gpu_seconds: Nat = 0

    @model_validator(mode="after")
    def consistent(self):
        if (
            self.experiment not in ALLOCATIONS
            or self.reserved_micro_usd != self.quote.reserved_micro_usd
            or self.gpu_seconds != self.quote.gpu_seconds
        ):
            raise ValueError("reservation_invalid")
        if (self.state == "closed") != (self.outcome != "pending"):
            raise ValueError("outcome_invalid")
        if self.state == "closed" and not self.cleanup_evidence:
            raise ValueError("cleanup_evidence_required")
        return self


class State(Strict):
    schema_version: Literal[1] = 1
    repository: Literal["jasonjiang8866/MLsysZero2Hero"] = REPOSITORY
    campaign: Literal["mlsys-initial-20261116"] = CAMPAIGN
    revision: Nat = 0
    reservations: dict[str, Reservation] = {}

    @model_validator(mode="after")
    def identities(self):
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", k) for k in self.reservations):
            raise ValueError("invalid_attempt_id")
        return self


class Budget:
    def __init__(self, store: Store):
        self.store = store

    def load(self) -> tuple[str, State]:
        version, raw = self.store.read(KEY)
        if raw is None or version is None:
            raise LedgerError("budget_missing_no_automatic_reset")
        try:
            return version, State.model_validate(raw)
        except ValueError:
            raise LedgerError("budget_invalid") from None

    def write(self, version: str | None, state: State) -> None:
        state.revision += 1
        document = State.model_validate(state.model_dump()).model_dump()
        try:
            self.store.swap(KEY, version, document)
        except LedgerError as exc:
            if str(exc) != "ledger_write_unknown":
                raise
        # A lost write response is never retried; exact readback must confirm intent.
        _, observed = self.store.read(KEY)
        if observed != document:
            raise LedgerError("budget_write_unconfirmed")

    def initialize(self) -> None:
        """Explicit one-time bootstrap; not exposed in the cloud preflight workflow."""
        version, raw = self.store.read(KEY)
        if version is not None or raw is not None:
            raise LedgerError("budget_already_exists")
        self.write(None, State())

    def reserve(
        self,
        attempt: str,
        experiment: str,
        quote: Quote,
        *,
        now: float,
        source_sha: str,
        image_digest: str,
    ) -> Reservation:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", attempt) or experiment not in ALLOCATIONS:
            raise LedgerError("invalid_experiment")
        if not math.isfinite(now) or not 0 <= now - quote.observed_at <= 120:
            raise LedgerError("quote_stale_or_future")
        version, state = self.load()
        if attempt in state.reservations:
            raise LedgerError("attempt_already_reserved")
        if any(r.state != "closed" for r in state.reservations.values()):
            raise LedgerError("unresolved_resource_blocks_new_work")
        records = list(state.reservations.values())

        def total(rows, money):
            return sum(
                max(r.reserved_micro_usd, r.observed_micro_usd)
                if money
                else max(r.gpu_seconds, r.observed_gpu_seconds)
                for r in rows
            )

        same = [r for r in records if r.experiment == experiment]
        money, seconds = quote.reserved_micro_usd, quote.gpu_seconds
        cap_money, cap_seconds = ALLOCATIONS[experiment]
        if (
            total(records, True) + money > WORK_MONEY_CAP
            or total(records, False) + seconds > WORK_TIME_CAP
            or total(same, True) + money > cap_money
            or total(same, False) + seconds > cap_seconds
        ):
            raise LedgerError("cumulative_budget_exceeded")
        record = Reservation(
            experiment=experiment,
            quote=quote,
            reserved_micro_usd=money,
            gpu_seconds=seconds,
            source_sha=source_sha,
            image_digest=image_digest,
        )
        state.reservations[attempt] = record
        self.write(version, state)
        return record

    def mark_unknown(self, attempt: str) -> None:
        version, state = self.load()
        record = state.reservations[attempt]
        if record.state != "reserved":
            raise LedgerError("invalid_transition")
        record.state = "creation_unknown"
        self.write(version, state)

    def close(
        self,
        attempt: str,
        *,
        outcome: str,
        cleanup_evidence: str,
        observed_micro_usd: int,
        observed_gpu_seconds: int,
    ) -> None:
        """Called only AFTER independent cleanup verification; never a deletion API.

        Evidence reference is required but its truth must be checked by the future
        live controller. This library is not itself a verifier of provider receipts.
        """
        version, state = self.load()
        record = state.reservations[attempt]
        if record.state == "closed" or outcome not in {"accepted", "rejected", "not_created"}:
            raise LedgerError("invalid_transition")
        if (
            not re.fullmatch(r"sha256:[a-f0-9]{64}", cleanup_evidence)
            or record.state == "creation_unknown"
            and outcome == "not_created"
        ):
            raise LedgerError("cleanup_proof_required")
        record.state, record.outcome = "closed", outcome
        record.cleanup_evidence = cleanup_evidence
        record.observed_micro_usd = observed_micro_usd
        record.observed_gpu_seconds = observed_gpu_seconds
        self.write(version, state)
