"""Manual E00 controller and cleanup-only recovery. One POST per reservation."""

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import statistics
import time
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from runpod_ci_broker.export import RunPodResultReader
from runpod_ci_broker.ledger import GitHubContentsStore, LedgerError
from runpod_ci_broker.live import CreateHTTPError, LiveAPI
from runpod_ci_broker.planner import reasons
from runpod_ci_broker.provider import ProviderError, ReadOnlyRunPod

from . import e01_hosted
from .artifact import github_json, uploaded_artifact
from .budget import CAMPAIGN, REPOSITORY, Budget, Quote
from .cleanup import Allocation, cleanup_owned, matches
from .preflight import GPU_IDS, discovery_task
from .transport import collect
from .worker import Job, Runtime


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class Journal:
    def __init__(self, store, run, attempt, experiment="E00"):
        if experiment not in {"E00", "E01"}:
            raise ValueError("unknown_experiment")
        self.experiment = experiment
        if not all(re.fullmatch(r"[1-9][0-9]{0,18}", value) for value in (run, attempt)):
            raise ValueError("invalid_run_identity")
        self.run, self.number = run, int(attempt)
        self.attempt = f"{experiment}-{run}-{attempt}"
        self.key = hashlib.sha256(f"{REPOSITORY}:{CAMPAIGN}:{self.attempt}".encode()).hexdigest()
        self.store, self.budget = store, Budget(store)
        self.version, self.record = store.read(self.key)

    def save(self, **changes):
        document = {**(self.record or {}), **changes}
        try:
            self.store.swap(self.key, self.version, document)
        except LedgerError as exc:
            if str(exc) != "ledger_write_unknown":
                raise
        self.version, self.record = self.store.read(self.key)
        if self.record != document or not self.version:
            raise ValueError("journal_write_unconfirmed")

    def allocation(self, pod_id):
        return Allocation(
            runtime=Runtime(job=Job.model_validate(self.record["job"]), pod_id=pod_id),
            actions_run_id=self.run,
            actions_run_attempt=self.number,
            data_center=self.record["center"],
            cloud="SECURE",
            create_intent=True,
        )

    def trusted(self):
        if self.store.read(self.key) != (self.version, self.record):
            raise ValueError("journal_changed")
        return self.allocation(self.record["pod_id"])


def adopt(j, pod):
    # RunPod can choose a different placement. Only accept the observed center
    # after checking every other ownership fact and the journaled Pod ID if known.
    actual = pod.get("dataCenterId")
    pod_id = pod.get("id")
    if not isinstance(actual, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", actual):
        return False
    if not isinstance(pod_id, str) or (j.record.get("pod_id") and pod_id != j.record["pod_id"]):
        return False
    candidate = j.allocation(pod_id).model_copy(update={"data_center": actual})
    if not matches(pod, candidate):
        return False
    if actual != j.record["center"] or not j.record.get("pod_id"):
        j.save(
            requested_center=j.record.get("requested_center", j.record["center"]),
            center=actual,
            pod_id=pod_id,
        )
    return True


def launch(j, api, catalog, source, image, output, disk_rate, extras):
    if j.record is not None:
        raise ValueError("attempt_already_exists")
    j.budget.load()
    task = discovery_task()
    observed = time.time()
    rows, inventory = catalog.snapshot(task)
    if (
        inventory.get("inventory") != "complete"
        or inventory.get("query_status") != "ok"
        or inventory.get("datacenter_status") != "ok"
        or inventory.get("scope_active_pod_count") != 0
    ):
        raise ValueError("inventory_unverified")
    if any(
        p["env"].get("MLS_CAMPAIGN") == CAMPAIGN and p["status"] != "TERMINATED"
        for p in api.inventory()
    ):
        raise ValueError("existing_campaign_pod")
    eligible = [r for r in rows if r.evidence == "read_only" and not reasons(task, r)]
    if not eligible:
        raise ValueError("no_eligible_capacity")
    chosen = min(eligible, key=lambda r: (GPU_IDS.index(r.gpu_id), r.hourly_usd, r.data_center))
    rate = int(
        (Decimal(str(chosen.hourly_usd)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING)
    )
    quote = Quote(
        observed_at=observed,
        compute_micro_usd_per_gpu_hour=rate,
        storage_micro_usd_per_pod_hour=disk_rate,
        extras_micro_usd=extras,
        gpu_count=1,
        wall_seconds=1200 if j.experiment == "E01" else 600,
    )
    job = Job(
        experiment=j.experiment,
        attempt=j.attempt,
        source_sha=source,
        image_digest=image,
        gpu=chosen.gpu_id,
    )
    centers = sorted(
        {
            r.data_center
            for r in eligible
            if r.gpu_id == job.gpu and r.hourly_usd == chosen.hourly_usd
        }
    )
    j.save(
        job=job.model_dump(), center=chosen.data_center, eligible_centers=centers, phase="prepared"
    )
    j.budget.reserve(
        j.attempt, j.experiment, quote, now=time.time(), source_sha=source, image_digest=image
    )
    # The deterministic name and owner markers are independent of the future Pod ID.
    intent = j.allocation("pending")
    token = secrets.token_hex(32)
    body = {
        "name": intent.name,
        "image": job.image,
        "gpu": {"id": job.gpu, "count": 1, "minCudaVersion": "12.8"},
        "cloud": "SECURE",
        "dataCenterIds": centers,
        "disk": 5,
        "mounts": {},
        "ports": ["8000/http"],
        "startSsh": False,
        "startJupyter": False,
        "globalNetworking": False,
        "env": {
            **intent.markers(),
            "MLS_JOB_JSON": job.model_dump_json(),
            "MLS_RESULT_TOKEN": token,
        },
    }
    if j.experiment == "E01":
        body.update(e01_hosted.launch_options(image))
    j.save(phase="create_intent", started=time.time())
    try:
        pod_id = api.create(body)
    except CreateHTTPError as exc:
        if not exc.ambiguous:
            j.save(phase="create_refused", reason=exc.diagnostic.get("reason_class", exc.code))
        raise
    j.save(phase="allocated", pod_id=pod_id)
    pod = None
    for _ in range(10):
        pod = api.get(pod_id, 5)
        if pod is not None:
            break
        time.sleep(1)
    if not pod or not adopt(j, pod):
        raise ValueError("allocation_unverified")
    cost = pod.get("cost")
    if type(cost) not in (float, int) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("allocated_rate_unverified")
    j.save(hourly_cost=cost)
    if cost > (rate + disk_rate) / 1e6:
        raise ValueError("allocated_rate_exceeded")
    collect(
        j.trusted().runtime,
        RunPodResultReader(pod_id, 8000, token),
        output,
        timeout=900 if j.experiment == "E01" else 240,
    )
    evidence = {}
    if j.experiment == "E01":
        summary = json.loads((output / "summary.json").read_bytes())
        evidence["trace_summary"] = {
            case: {
                **{
                    k: row[k]
                    for k in (
                        "kernels",
                        "median_wall_ms",
                        "kernel_active_us",
                        "inter_kernel_gap_us",
                        "longest_gap_us",
                        "output_sha256",
                    )
                },
                "median_host_cpu_ms": statistics.median(
                    s["host_thread_cpu_ms"] for s in row["samples"]
                ),
            }
            for case, row in summary["results"].items()
        }
    j.save(
        phase="collected",
        manifest_digest=digest((output / "manifest.json").read_bytes()),
        **evidence,
    )


def receipt(j, artifact, identifier, archive_digest, output):
    expected = "sha256:" + archive_digest.removeprefix("sha256:")
    if (
        not re.fullmatch(r"sha256:[0-9a-f]{64}", expected)
        or artifact.get("digest") != expected
        or artifact.get("expired") is not False
        or artifact.get("id") != int(identifier)
        or artifact.get("name") != j.attempt
        or artifact["workflow_run"]["id"] != int(j.run)
        or artifact["workflow_run"]["head_sha"] != j.record["job"]["source_sha"]
        or j.record["phase"] != "collected"
        or digest((output / "manifest.json").read_bytes()) != j.record["manifest_digest"]
    ):
        raise ValueError("artifact_receipt_invalid")
    j.save(artifact_id=int(identifier), archive_digest=expected)


def recover(j, api):
    if not j.record:
        return "no_intent"
    _, state = j.budget.load()
    reservation = state.reservations.get(j.attempt)
    if not reservation:
        if j.record["phase"] == "prepared":
            return "no_intent"
        raise ValueError("reservation_missing")
    if reservation.state == "closed":
        return reservation.outcome
    if j.record["phase"] in ("prepared", "create_refused"):
        outcome, seconds, money = "not_created", 0, 0
    else:
        if not j.record.get("pod_id"):
            candidates = [
                p for p in api.inventory() if p.get("name") == j.allocation("pending").name
            ]
            if len(candidates) != 1 or not adopt(j, candidates[0]):
                raise ValueError("creation_unresolved")
        else:
            pod = api.get(j.record["pod_id"], 5)
            if pod is not None and not adopt(j, pod):
                raise ValueError("ownership_refused")
        proof = cleanup_owned(
            j.trusted(),
            api,
            load_trusted=j.trusted,
            checkpoint=lambda r: j.save(cleanup=r.model_dump()),
            allow_termination=True,
            timeout=90,
        )
        if not proof.deleted:
            raise ValueError("cleanup_unconfirmed")
        seconds = max(1, math.ceil(time.time() - j.record["started"]))
        quote = reservation.quote
        rate = max(
            math.ceil(j.record.get("hourly_cost", 0) * 1e6), quote.compute_micro_usd_per_gpu_hour
        )
        money = (
            (rate + quote.storage_micro_usd_per_pod_hour) * seconds + 3599
        ) // 3600 + quote.extras_micro_usd
        outcome = (
            "accepted"
            if j.record.get("artifact_id")
            and seconds <= quote.wall_seconds
            and money <= reservation.reserved_micro_usd
            else "rejected"
        )
    j.budget.close(
        j.attempt,
        outcome=outcome,
        cleanup_evidence=digest(json.dumps(j.record, sort_keys=True).encode()),
        observed_micro_usd=money,
        observed_gpu_seconds=seconds,
    )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "receipt", "cleanup", "recover"])
    action = parser.parse_args().action
    env = os.environ
    try:
        if env["GITHUB_REPOSITORY"] != REPOSITORY or env["GITHUB_REF"] != "refs/heads/main":
            raise ValueError("trusted_main_required")
        j = Journal(
            GitHubContentsStore(REPOSITORY),
            env["OWNER_RUN"],
            env["OWNER_ATTEMPT"],
            env.get("COURSE_EXPERIMENT", "E00"),
        )
        output = Path("out") / f"{j.experiment.lower()}-bundle"
        if action == "launch":
            launch(
                j,
                LiveAPI(allow_create=True),
                ReadOnlyRunPod(),
                env["GITHUB_SHA"],
                env["IMAGE_DIGEST"],
                output,
                int(env["DISK_MICRO_USD_HOUR"]),
                int(env["EXTRAS_MICRO_USD"]),
            )
        elif action == "receipt":
            receipt(
                j,
                uploaded_artifact(j.run, j.attempt, env["ARTIFACT_ID"], fetch=github_json),
                env["ARTIFACT_ID"],
                env["ARCHIVE_DIGEST"],
                output,
            )
        else:
            result = recover(j, LiveAPI(allow_termination=True))
            print(json.dumps({"attempt": j.attempt, "outcome": result}))
            return 0 if action == "recover" or result == "accepted" else 2
        return 0
    except (ValueError, OSError, KeyError, TypeError, ProviderError) as exc:
        # Never emit provider responses, worker environments or untrusted raw exceptions.
        code = str(exc) if type(exc) in (ValueError, LedgerError, ProviderError) else ""
        print(
            json.dumps(
                {"error": code if re.fullmatch(r"[a-z0-9_]{1,80}", code) else "controller_failed"}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
