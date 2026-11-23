"""L00 teaching model. Estimate outgoing dispatch payload for one source node."""

import argparse
import math

TOKENS = 8192
HIDDEN = 4096
BYTES_PER_ELEMENT = 2
MIB = 1024**2
LINK_MIB_PER_MS = 16


def estimate(nodes, compute_ms):
    """The four entries give the host nodes of four distinct selected experts."""
    if len(nodes) != 4 or any(node < 0 for node in nodes):
        raise ValueError("Give four nonnegative node IDs (source node = 0).")
    if not math.isfinite(compute_ms) or compute_ms < 0:
        raise ValueError("Compute time must be finite and nonnegative.")
    remote_nodes = len(set(nodes) - {0})
    payload_bytes = TOKENS * HIDDEN * BYTES_PER_ELEMENT * remote_nodes
    dispatch_ms = payload_bytes / MIB / LINK_MIB_PER_MS
    return {
        "remote_nodes": remote_nodes,
        "payload_bytes": payload_bytes,
        "dispatch_ms": dispatch_ms,
        "serial_ms": compute_ms + dispatch_ms,
        "ideal_overlap_ms": max(compute_ms, dispatch_ms),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs=4, default=[1, 2, 3, 4], metavar="ID")
    parser.add_argument("--compute-ms", type=float, default=6.0)
    args = parser.parse_args()
    try:
        row = estimate(args.nodes, args.compute_ms)
    except ValueError as exc:
        parser.error(str(exc))
    print("L00 synthetic model | source node = 0 | one-way dispatch")
    print(f"remote_nodes={row['remote_nodes']} payload_MiB={row['payload_bytes'] / MIB:g}")
    for name in ("dispatch_ms", "serial_ms", "ideal_overlap_ms"):
        print(f"{name}={row[name]:g}")


if __name__ == "__main__":
    main()
