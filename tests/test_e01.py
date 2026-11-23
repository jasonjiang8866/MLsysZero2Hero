import pytest

from mlsys_lab.e01 import kernel_timeline


def test_kernel_timeline_merges_overlap_and_excludes_cpu_and_copies():
    events = [
        {"ph": "X", "cat": category, "ts": start, "dur": duration}
        for category, start, duration in [
            ("kernel", 20, 5),
            ("cpu_op", 0, 50),
            ("kernel", 1, 8),
            ("gpu_memcpy", 12, 1),
            ("kernel", 3, 1),
            ("kernel", 7, 4),
        ]
    ]
    assert kernel_timeline(events) == {
        "kernels": 4,
        "span_us": 24,
        "kernel_active_us": 15,
        "inter_kernel_gap_us": 9,
        "longest_gap_us": 9,
    }


def test_no_device_activity_cannot_pass():
    with pytest.raises(ValueError, match="no CUDA kernels"):
        kernel_timeline([{"ph": "X", "cat": "cpu_op", "ts": 0, "dur": 10}])
