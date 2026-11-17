"""Driver doubles exercise error paths; these tests are NOT GPU evidence."""

import ctypes as c
import json

import pytest
from runpod_ci_broker.cuda_probe import GPUProbeError

from mlsys_lab.cuda_smoke import N, execute, reference
from mlsys_lab.results import seal, verify

SHA = "a" * 40


class FakeDriver:
    def __init__(self, fail=None, corrupt=False, count=1, version=13000):
        self.fail, self.corrupt, self.count, self.version = fail, corrupt, count, version
        self.calls = []

    def call(self, name, types, *args):
        self.calls.append(name)
        if name == self.fail:
            raise GPUProbeError("injected_failure")
        if name == "cuDeviceGetCount":
            c.cast(args[0], c.POINTER(c.c_int))[0] = self.count
        elif name == "cuDriverGetVersion":
            c.cast(args[0], c.POINTER(c.c_int))[0] = self.version
        elif name == "cuDeviceGetName":
            args[0].value = b"NVIDIA GeForce RTX 4090"
        elif name == "cuDeviceTotalMem_v2":
            c.cast(args[0], c.POINTER(c.c_size_t))[0] = 24 * 1024**3
        elif name == "cuDeviceGetAttribute":
            c.cast(args[0], c.POINTER(c.c_int))[0] = 8 if args[1] == 75 else 9
        elif name == "cuModuleLoadData":
            c.cast(args[0], c.POINTER(c.c_void_p))[0] = 1
        elif name == "cuMemcpyDtoH_v2":
            if args[2] == 4096:
                for i in range(1024):
                    args[0][i] = 0x13579BDF
            else:
                assert "cuLaunchKernel" in self.calls
                for i, value in enumerate(reference()):
                    args[0][i] = float("nan") if self.corrupt and i == N - 1 else value


@pytest.fixture
def bundle(tmp_path):
    metrics = execute(tmp_path, FakeDriver())
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    seal(tmp_path, SHA, None)
    return tmp_path


def test_kernel_launch_and_cleanup_sequence(tmp_path):
    driver = FakeDriver()
    metrics = execute(tmp_path, driver)
    assert metrics["elements"] == 257 and metrics["max_abs_error"] == 0
    assert "cuLaunchKernel" in driver.calls
    assert driver.calls[-2:] == ["cuModuleUnload", "cuDevicePrimaryCtxRelease_v2"]


@pytest.mark.parametrize("fail", ["cuModuleLoadData", "cuModuleGetFunction", "cuLaunchKernel"])
def test_failed_kernel_cannot_produce_result(tmp_path, fail):
    driver = FakeDriver(fail=fail)
    with pytest.raises(GPUProbeError):
        execute(tmp_path, driver)
    assert driver.calls[-1] == "cuDevicePrimaryCtxRelease_v2"
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_wrong_device_count_fails_without_cpu_fallback(tmp_path, count):
    driver = FakeDriver(count=count)
    with pytest.raises(GPUProbeError):
        execute(tmp_path, driver)
    assert "cuLaunchKernel" not in driver.calls


def test_partial_block_missing_write_is_detected(tmp_path):
    with pytest.raises(GPUProbeError, match="output"):
        execute(tmp_path, FakeDriver(corrupt=True))


@pytest.mark.parametrize("version,passed", [(12070, False), (12080, True), (13000, True)])
def test_numeric_cuda_floor(tmp_path, version, passed):
    if passed:
        execute(tmp_path, FakeDriver(version=version))
    else:
        with pytest.raises(GPUProbeError, match="below"):
            execute(tmp_path, FakeDriver(version=version))


def test_valid_bundle_is_not_cloud_acceptance(bundle):
    result = verify(bundle, SHA, None)
    assert result.kernel_verified and result.cloud_acceptance == "not_evaluated"


@pytest.mark.parametrize("name", ["output.json", "metrics.json"])
def test_corruption_or_missing_file_rejected(bundle, name):
    (bundle / name).write_text("{}")
    with pytest.raises(ValueError):
        verify(bundle, SHA, None)
    (bundle / name).unlink()
    with pytest.raises(ValueError):
        verify(bundle, SHA, None)


def test_wrong_revision_and_image_rejected(bundle):
    with pytest.raises(ValueError, match="provenance"):
        verify(bundle, "b" * 40, None)
    with pytest.raises(ValueError, match="provenance"):
        verify(bundle, SHA, "sha256:" + "d" * 64)


def test_bad_numerics_cannot_be_resealed(bundle):
    (bundle / "manifest.json").unlink()
    (bundle / "output.json").write_text(json.dumps([0.0] * N))
    with pytest.raises(GPUProbeError):
        seal(bundle, SHA, None)


def test_large_artifact_rejected(bundle):
    (bundle / "output.json").write_bytes(b" " * 65537)
    with pytest.raises(ValueError):
        verify(bundle, SHA, None)


def test_non_object_metrics_rejected_cleanly(bundle):
    (bundle / "manifest.json").unlink()
    (bundle / "metrics.json").write_text("[]")
    with pytest.raises(ValueError, match="metrics_invalid"):
        seal(bundle, SHA, None)


def test_symlink_rejected(bundle, tmp_path_factory):
    alternate = tmp_path_factory.mktemp("other") / "output.json"
    alternate.write_bytes((bundle / "output.json").read_bytes())
    (bundle / "output.json").unlink()
    (bundle / "output.json").symlink_to(alternate)
    with pytest.raises(ValueError):
        verify(bundle, SHA, None)
