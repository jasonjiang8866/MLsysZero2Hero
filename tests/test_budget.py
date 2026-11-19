import copy

import pytest
from runpod_ci_broker.ledger import LedgerError, SQLiteStore

from mlsys_lab.budget import ALLOCATIONS, KEY, Budget, Quote

SHA = "a" * 40
IMAGE = "sha256:" + "b" * 64
PROOF = "sha256:" + "c" * 64


def quote(**changes):
    return Quote(
        **{
            "observed_at": 1000.0,
            "compute_micro_usd_per_gpu_hour": 1_000_000,
            "storage_micro_usd_per_pod_hour": 1000,
            "extras_micro_usd": 100_000,
            "gpu_count": 1,
            "wall_seconds": 600,
            **changes,
        }
    )


@pytest.fixture
def budget(tmp_path):
    b = Budget(SQLiteStore(tmp_path / "budget.sqlite3"))
    b.initialize()
    return b


def reserve(b, attempt="attempt-1", experiment="E00", q=None):
    return b.reserve(
        attempt, experiment, q or quote(), now=1010.0, source_sha=SHA, image_digest=IMAGE
    )


def close(b, attempt="attempt-1", **extra):
    b.close(
        attempt,
        outcome="rejected",
        cleanup_evidence=PROOF,
        observed_micro_usd=extra.get("money", 10),
        observed_gpu_seconds=extra.get("seconds", 10),
    )


def test_missing_ledger_never_resets(tmp_path):
    b = Budget(SQLiteStore(tmp_path / "missing.sqlite3"))
    with pytest.raises(LedgerError, match="missing"):
        reserve(b)


def test_initialization_cannot_reset_history(budget):
    reserve(budget)
    with pytest.raises(LedgerError, match="already_exists"):
        budget.initialize()


def test_round_up_and_multigpu_time():
    q = quote(
        gpu_count=2,
        wall_seconds=31,
        compute_micro_usd_per_gpu_hour=1,
        storage_micro_usd_per_pod_hour=1,
        extras_micro_usd=0,
    )
    assert q.gpu_seconds == 62 and q.reserved_micro_usd == 1


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_count": True},
        {"gpu_count": 0},
        {"wall_seconds": -1},
        {"observed_at": float("nan")},
        {"compute_micro_usd_per_gpu_hour": -1},
        {"storage_micro_usd_per_pod_hour": None},
        {"extras_micro_usd": -1},
    ],
)
def test_invalid_quote_rejected(change):
    with pytest.raises(ValueError):
        quote(**change)


@pytest.mark.parametrize("observed", [0.0, 1011.0])
def test_stale_and_future_quotes_block(budget, observed):
    with pytest.raises(LedgerError, match="quote"):
        reserve(budget, q=quote(observed_at=observed))


def test_two_gpus_consume_double_minutes(budget):
    low_rate = quote(
        gpu_count=2, wall_seconds=1800, compute_micro_usd_per_gpu_hour=1, extras_micro_usd=0
    )
    for index in range(3):
        reserve(budget, f"two-{index}", q=low_rate)
        close(budget, f"two-{index}")
    with pytest.raises(LedgerError, match="budget"):
        reserve(budget, "two-3", q=low_rate)


def test_retries_never_reset_experiment_cap(budget):
    cheap = quote(compute_micro_usd_per_gpu_hour=100_000, extras_micro_usd=0)
    for i in range(20):
        reserve(budget, f"try-{i}", q=cheap)
        close(budget, f"try-{i}")
    with pytest.raises(LedgerError, match="budget"):
        reserve(budget, "try-20", q=cheap)
    assert len(budget.load()[1].reservations) == 20


def test_unknown_create_blocks_retries_and_other_experiments(budget):
    reserve(budget)
    budget.mark_unknown("attempt-1")
    with pytest.raises(LedgerError, match="unresolved"):
        reserve(budget, "new", "E01")
    with pytest.raises(LedgerError, match="proof"):
        budget.close(
            "attempt-1",
            outcome="not_created",
            cleanup_evidence=PROOF,
            observed_micro_usd=0,
            observed_gpu_seconds=0,
        )


def test_success_cannot_refund_reservations(budget):
    original = reserve(budget)
    budget.close(
        "attempt-1",
        outcome="accepted",
        cleanup_evidence=PROOF,
        observed_micro_usd=1,
        observed_gpu_seconds=1,
    )
    assert budget.load()[1].reservations["attempt-1"].reserved_micro_usd == (
        original.reserved_micro_usd
    )
    with pytest.raises(LedgerError, match="already_reserved"):
        reserve(budget)


def test_overrun_blocks_next_reservation(budget):
    reserve(budget)
    close(budget, money=20_000_001, seconds=60_001)
    with pytest.raises(LedgerError, match="budget"):
        reserve(budget, "next", "E01")


def test_invalid_close_does_not_mutate_store(budget):
    reserve(budget)
    before = budget.store.read(KEY)
    with pytest.raises(ValueError):
        close(budget, money=-1)
    assert budget.store.read(KEY) == before


def test_corrupt_ledger_blocks(budget):
    version, raw = budget.store.read(KEY)
    raw["repository"] = "someone/else"
    budget.store.swap(KEY, version, raw)
    with pytest.raises(LedgerError, match="invalid"):
        reserve(budget)


def test_compare_and_swap_conflict_prevents_double_reservation(budget):
    version, first = budget.load()
    second = copy.deepcopy(first)
    budget.write(version, first)
    with pytest.raises(LedgerError, match="conflict"):
        budget.write(version, second)


def test_lost_write_ack_reconciles_only_exact_readback(budget, monkeypatch):
    original = budget.store.swap

    def lost_ack(key, version, value):
        original(key, version, value)
        raise LedgerError("ledger_write_unknown")

    monkeypatch.setattr(budget.store, "swap", lost_ack)
    reserve(budget)
    assert len(budget.load()[1].reservations) == 1


def test_lost_write_without_persist_does_not_permit_create(budget, monkeypatch):
    def no_write(*args):
        raise LedgerError("ledger_write_unknown")

    monkeypatch.setattr(budget.store, "swap", no_write)
    with pytest.raises(LedgerError, match="unconfirmed"):
        reserve(budget)
    assert not budget.load()[1].reservations


def test_allocations_preserve_recovery_reserve():
    assert sum(v[0] for v in ALLOCATIONS.values()) == 18_000_000
    assert sum(v[1] for v in ALLOCATIONS.values()) == 58_200
