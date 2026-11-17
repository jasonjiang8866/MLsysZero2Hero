"""Local HTTP and fake-driver integration only; these are not CUDA/cloud evidence."""

import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from runpod_ci_broker.provider import ProviderError
from test_smoke_results import FakeDriver

from mlsys_lab import transport, worker
from mlsys_lab.cuda_smoke import execute
from mlsys_lab.results import seal, verify

TOKEN = "1" * 64


@pytest.fixture
def runtime():
    return worker.Runtime(
        job=worker.Job(
            attempt="e00-1",
            source_sha="a" * 40,
            image_digest="sha256:" + "b" * 64,
            gpu="NVIDIA GeForce RTX 4090",
        ),
        pod_id="testpod",
    )


@pytest.fixture
def source(tmp_path, runtime):
    root = tmp_path / "source"
    root.mkdir()
    metrics = execute(root, FakeDriver())
    (root / "metrics.json").write_text(json.dumps(metrics))
    seal(root, runtime.job.source_sha, runtime.job.image_digest)
    return root


class Reader:
    def __init__(self, runtime, payloads):
        self.payloads = {
            "/v1/status": json.dumps(
                {"phase": "ready", "runtime_fingerprint": runtime.fingerprint()}
            ).encode(),
            **payloads,
        }
        self.calls = []

    def open(self, path, timeout):
        self.calls.append(path)
        value = self.payloads[path]
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(value)


def test_roundtrip_recomputes_numerics_and_identity(source, runtime, tmp_path):
    reader = Reader(runtime, worker.export(source, runtime))
    target = tmp_path / "saved"
    transport.collect(runtime, reader, target)
    verify(target, runtime.job.source_sha, runtime.job.image_digest)
    assert json.loads((target / "transfer.json").read_text())["runtime"] == runtime.model_dump()
    assert not list(tmp_path.glob(".e00-collect-*"))


@pytest.mark.parametrize("kind", ["corrupt", "oversized", "missing", "wrong_attempt", "wrong_sha"])
def test_failed_collection_never_publishes(source, runtime, tmp_path, kind):
    reader = Reader(runtime, worker.export(source, runtime))
    path = "/v1/files/output.json"
    if kind == "corrupt":
        reader.payloads[path] = b"[0]"
    elif kind == "oversized":
        reader.payloads[path] = b" " * 65537
    elif kind == "missing":
        reader.payloads[path] = HTTPError("hidden", 404, "hidden", {}, None)
    else:
        value = json.loads(reader.payloads["/v1/manifest"])
        if kind == "wrong_attempt":
            value["runtime"]["job"]["attempt"] = "old-attempt"
        else:
            value["bundle"]["source_sha"] = "c" * 40
        reader.payloads["/v1/manifest"] = json.dumps(value).encode()
    with pytest.raises(ProviderError):
        transport.collect(runtime, reader, tmp_path / "saved")
    assert not (tmp_path / "saved").exists()
    assert not list(tmp_path.glob(".e00-collect-*"))


@pytest.mark.parametrize("code", [401, 403, 400])
def test_readiness_auth_failure_is_terminal(source, runtime, tmp_path, code):
    reader = Reader(runtime, worker.export(source, runtime))
    reader.payloads["/v1/status"] = HTTPError("hidden", code, "hidden", {}, None)
    with pytest.raises(ProviderError, match="readiness_http_refused"):
        transport.collect(runtime, reader, tmp_path / "saved")
    assert reader.calls == ["/v1/status"]


def test_proxy_not_ready_retries_before_result_fetch(source, runtime, tmp_path):
    reader = Reader(runtime, worker.export(source, runtime))
    reader.payloads["/v1/status"] = [
        HTTPError("hidden", 404, "hidden", {}, None),
        reader.payloads["/v1/status"],
    ]
    transport.collect(runtime, reader, tmp_path / "saved", sleep=lambda _: None)
    assert reader.calls[:3] == ["/v1/status", "/v1/status", "/v1/manifest"]


def test_readiness_deadline_does_not_loop_forever(source, runtime, tmp_path):
    reader = Reader(runtime, worker.export(source, runtime))
    reader.payloads["/v1/status"] = HTTPError("hidden", 503, "hidden", {}, None)
    clock = [0.0]
    with pytest.raises(ProviderError, match="collection_deadline"):
        transport.collect(
            runtime,
            reader,
            tmp_path / "saved",
            timeout=2,
            now=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        )
    assert clock[0] == 2


@pytest.mark.parametrize("phase", ["failed", "unknown"])
def test_worker_failure_and_bad_phase_rejected(source, runtime, tmp_path, phase):
    reader = Reader(runtime, worker.export(source, runtime))
    reader.payloads["/v1/status"] = json.dumps(
        {"phase": phase, "runtime_fingerprint": runtime.fingerprint()}
    ).encode()
    with pytest.raises(ProviderError):
        transport.collect(runtime, reader, tmp_path / "saved")


def test_pod_identity_mismatch_rejected(source, runtime, tmp_path):
    reader = Reader(runtime, worker.export(source, runtime))
    other = runtime.model_copy(update={"pod_id": "other"})
    with pytest.raises(ProviderError, match="identity"):
        transport.collect(other, reader, tmp_path / "saved")


def test_atomic_publish_cannot_replace_competing_directory(tmp_path):
    stage, target = tmp_path / "stage", tmp_path / "target"
    stage.mkdir()
    (stage / "new").write_text("data")
    target.mkdir()
    with pytest.raises(OSError):
        transport.publish(stage, target)
    assert stage.exists() and not list(target.iterdir())


def test_destination_symlink_refused_before_network(source, runtime, tmp_path):
    reader = Reader(runtime, worker.export(source, runtime))
    (tmp_path / "link").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError):
        transport.collect(runtime, reader, tmp_path / "link" / "new")
    assert not reader.calls


def test_real_local_http_auth_and_closed_paths(source, runtime, tmp_path):
    state = worker.State()
    with worker.server(runtime, state, TOKEN, ("127.0.0.1", 0)) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        try:

            def request(path, token=TOKEN):
                return urlopen(Request(base + path, headers={"Authorization": "Bearer " + token}))

            with pytest.raises(HTTPError) as exc:
                request("/v1/status", "bad")
            assert exc.value.code == 401
            exc.value.close()
            with request("/v1/status") as response:
                assert json.load(response)["phase"] == "starting"
            with pytest.raises(HTTPError) as exc:
                request("/v1/manifest")
            assert exc.value.code == 503
            exc.value.close()
            state.update("ready", worker.export(source, runtime))
            with request("/v1/manifest") as response:
                assert json.load(response)["runtime"] == runtime.model_dump()

            class LocalReader:
                def open(self, path, timeout):
                    return urlopen(
                        Request(base + path, headers={"Authorization": "Bearer " + TOKEN}),
                        timeout=timeout,
                    )

            transport.collect(runtime, LocalReader(), tmp_path / "http-export")
            verify(tmp_path / "http-export", runtime.job.source_sha, runtime.job.image_digest)
            for path in ("/v1/files/../../etc/passwd", "/shell", "/v1/status?token=x"):
                with pytest.raises(HTTPError) as exc:
                    request(path)
                assert exc.value.code == 404
                exc.value.close()
        finally:
            http.shutdown()
            thread.join(timeout=3)


def test_worker_build_revision_checked_before_compute(runtime, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("CUDA must not be invoked")

    monkeypatch.setattr(worker, "run_smoke", forbidden)
    state = worker.State()
    worker.compute(runtime, tmp_path, "c" * 40, state)
    assert json.loads(state.response("/v1/status", runtime)[1])["phase"] == "failed"


def test_worker_rejects_wrong_gpu_even_after_valid_arithmetic(source, runtime):
    other = runtime.model_copy(
        update={"job": runtime.job.model_copy(update={"gpu": "NVIDIA GeForce RTX 3090"})}
    )
    with pytest.raises(ValueError, match="gpu_mismatch"):
        worker.export(source, other)


def test_compute_exposes_failed_phase_after_timeout(runtime, tmp_path, monkeypatch):
    import subprocess

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("fixed_command", 30)

    monkeypatch.setattr(worker, "run_smoke", timeout)
    state = worker.State()
    worker.compute(runtime, tmp_path / "result", runtime.job.source_sha, state)
    assert json.loads(state.response("/v1/status", runtime)[1])["phase"] == "failed"
    assert state.response("/v1/manifest", runtime)[0] == 503


def test_stream_timeout_leaves_no_published_bundle(source, runtime, tmp_path):
    reader = Reader(runtime, worker.export(source, runtime))
    reader.payloads["/v1/files/output.json"] = TimeoutError()
    with pytest.raises(ProviderError, match="transport_failed"):
        transport.collect(runtime, reader, tmp_path / "saved")
    assert not (tmp_path / "saved").exists()
    assert not list(tmp_path.glob(".e00-collect-*"))
