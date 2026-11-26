"""Authenticated bounded collection; verified local export is not durable upload."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError

from runpod_ci_broker.export import Reader, RunPodResultReader
from runpod_ci_broker.provider import ProviderError

from . import e01_hosted
from .worker import ExportManifest, Runtime, admit


def publish(staging: Path, destination: Path) -> None:
    """Linux renameat2 publishes atomically and cannot replace an existing directory."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        raise OSError(ctypes.get_errno(), "atomic_publish_failed")
    descriptor = os.open(destination.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def collect(
    runtime: Runtime,
    reader: Reader,
    destination: Path,
    *,
    timeout: float = 180,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not 0 < timeout <= (900 if runtime.job.experiment in {"E01", "E02"} else 300):
        raise ValueError("collection_timeout_invalid")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination_exists")
    if any(p.is_symlink() for p in destination.parents):
        raise ValueError("destination_symlink")
    deadline = now() + timeout

    def remaining() -> float:
        left = deadline - now()
        if left <= 0:
            raise ProviderError("collection_deadline")
        return min(left, 5.0)

    def read(path: str, limit: int) -> bytes:
        data = bytearray()
        with reader.open(path, remaining()) as response:
            method = getattr(response, "read1", response.read)
            while True:
                remaining()
                part = method(min(16384, limit + 1 - len(data)))
                remaining()
                if not part:
                    return bytes(data)
                data.extend(part)
                if len(data) > limit:
                    raise ProviderError("result_too_large")

    def read_result(path: str, limit: int) -> bytes:
        stage = path.rsplit("/", 1)[-1].removesuffix(".json")
        for attempt in range(4):
            try:
                return read(path, limit)
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in {404, 429, 500, 502, 503, 504} or attempt == 3:
                    raise ProviderError(f"result_{stage}_http_{status}") from None
                sleep(min(2**attempt, remaining()))
        raise AssertionError("unreachable")

    trace_bundle = runtime.job.experiment in {"E01", "E02"}
    staging: Path | None = None
    try:
        while True:
            remaining()
            try:
                raw = read("/v1/status", 32 * 1024**2 if trace_bundle else 4096)
                if trace_bundle and raw.startswith(b"PK\x03\x04"):
                    break
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if not trace_bundle:
                        raise
                    # A proxy can briefly serve a non-worker response during startup.
                    value = None
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in {404, 429, 502, 503, 504}:
                    raise ProviderError("readiness_http_refused") from None
                value = None  # Proxy readiness is transient; authentication is not.
            except (URLError, TimeoutError, OSError):
                value = None
            if value is not None:
                if (
                    not isinstance(value, dict)
                    or set(value) != {"phase", "runtime_fingerprint"}
                    or value["runtime_fingerprint"] != runtime.fingerprint()
                    or value["phase"] not in {"starting", "running", "ready", "failed"}
                ):
                    raise ProviderError("readiness_identity_invalid")
                if value["phase"] == "failed":
                    raise ProviderError("worker_failed")
                if value["phase"] == "ready":
                    if trace_bundle:
                        raise ProviderError("e01_ready_without_archive")
                    break
            sleep(min(1, remaining()))
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".course-collect-", dir=destination.parent))
        if trace_bundle:
            try:
                envelope_raw = e01_hosted.unpack(raw, staging, runtime)
            except (ValueError, KeyError, TypeError) as exc:
                # Preserve only known files after validating archive membership and run identity.
                if (staging / "manifest.json").is_file():
                    publish(
                        staging,
                        destination.with_name(runtime.job.experiment.lower() + "-diagnostic"),
                    )
                code = str(exc)
                if code in {
                    "e02_summary_mismatch",
                    "e02_case_mismatch",
                    "e02_trace_mismatch",
                    "e01_summary_mismatch",
                    "e01_trace_mismatch",
                    "e01_timing_invalid",
                    "e01_median_invalid",
                    "e01_file_hash_mismatch",
                    "e01_invalid_zip",
                    "e01_archive_files_invalid",
                    "e01_manifest_too_large",
                    "export_identity_invalid",
                    "e01_archive_size_mismatch",
                }:
                    raise ProviderError(code) from None
                raise ProviderError("e01_archive_invalid") from None
        else:
            envelope_raw = read_result("/v1/manifest", 65536)
            envelope = ExportManifest.model_validate_json(envelope_raw)
            if envelope.runtime != runtime or set(envelope.bundle.files) != {
                "metrics.json",
                "output.json",
            }:
                raise ProviderError("export_identity_invalid")
            for name, metadata in envelope.bundle.files.items():
                data = read_result("/v1/files/" + name, metadata.bytes)
                with (staging / name).open("xb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            (staging / "manifest.json").write_text(envelope.bundle.model_dump_json(indent=2) + "\n")
            admit(staging, runtime)
        (staging / "transfer.json").write_bytes(envelope_raw)
        for name in ("manifest.json", "transfer.json"):
            with (staging / name).open("rb") as handle:
                os.fsync(handle.fileno())
        remaining()
        fd = os.open(staging, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        publish(staging, destination)
    except HTTPError as exc:
        exc.close()
        raise ProviderError("result_http_refused") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError("result_transport_failed") from None
    except (ValueError, KeyError, TypeError):
        raise ProviderError("result_invalid") from None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    try:
        from .results import bounded

        runtime = Runtime.model_validate_json(bounded(args.runtime))
        reader = RunPodResultReader(runtime.pod_id, 8000, os.environ["MLS_RESULT_TOKEN"])
        collect(runtime, reader, args.output, timeout=args.timeout_seconds)
        print('{"kernel_verified":true,"local_export_verified":true,"cloud_acceptance":false}')
        return 0
    except (ValueError, OSError, KeyError, TypeError, ProviderError):
        print('{"status":"collection_failed","cloud_acceptance":false}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
