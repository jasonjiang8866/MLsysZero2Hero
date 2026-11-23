import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "lessons/l00/dispatch.py"


@pytest.mark.parametrize(
    ("nodes", "compute", "row", "timing"),
    [
        ([1, 2, 3, 4], 6, (4, 256), (16, 22, 16)),
        ([1, 1, 1, 1], 6, (1, 64), (4, 10, 6)),
        ([1, 2, 3, 4], 3, (4, 256), (16, 19, 16)),
        ([1, 1, 1, 1], 3, (1, 64), (4, 7, 4)),
        ([0, 1, 1, 2], 3, (2, 128), (8, 11, 8)),
        ([0, 0, 0, 0], 3, (0, 0), (0, 3, 3)),
    ],
)
def test_lesson_commands(nodes, compute, row, timing):
    command = [
        sys.executable,
        str(SCRIPT),
        "--nodes",
        *map(str, nodes),
        "--compute-ms",
        str(compute),
    ]
    output = subprocess.check_output(command, text=True).splitlines()
    assert output[1:] == [
        f"remote_nodes={row[0]} payload_MiB={row[1]}",
        f"dispatch_ms={timing[0]}",
        f"serial_ms={timing[1]}",
        f"ideal_overlap_ms={timing[2]}",
    ]
