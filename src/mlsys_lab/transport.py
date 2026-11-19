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
    if not 0 < timeout <= 300:
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

    staging: Path | None = None
    try:
        while True:
            remaining()
            try:
                value = json.loads(read("/v1/status", 4096))
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
                    break
            sleep(min(1, remaining()))
        envelope_raw = read_result("/v1/manifest", 65536)
        envelope = ExportManifest.model_validate_json(envelope_raw)
        if envelope.runtime != runtime or set(envelope.bundle.files) != {
            "metrics.json",
            "output.json",
        }:
            raise ProviderError("export_identity_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".e00-collect-", dir=destination.parent))
        for name, metadata in envelope.bundle.files.items():
            data = read_result("/v1/files/" + name, metadata.bytes)
            with (staging / name).open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        (staging / "manifest.json").write_text(envelope.bundle.model_dump_json(indent=2) + "\n")
        # Recompute numerical results, file hashes and selected-device admission.
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
