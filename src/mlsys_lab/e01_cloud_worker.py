"""Run the locked course workload inside the public PyTorch image."""

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from .e01 import CASES
from .result_server import State, server


def main():
    job = json.loads(os.environ.pop("MLS_JOB_JSON"))
    pod_id = os.environ["RUNPOD_POD_ID"]
    runtime = {"job": job, "pod_id": pod_id}
    token = os.environ.pop("MLS_RESULT_TOKEN")
    os.environ.pop("RUNPOD_API_KEY", None)
    identity = json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()

    class Identity:
        def fingerprint(self):
            return hashlib.sha256(identity).hexdigest()

    state = State()

    def compute():
        try:
            state.update("running")
            experiment = job["experiment"]
            root = Path("/tmp") / (experiment.lower() + "-results")
            if experiment == "E02":
                from .e02 import trace_names

                files = trace_names()
            else:
                files = [f"{case}.json" for case in CASES]
            # Keep profiler activity on the subprocess main thread.
            subprocess.run(
                [sys.executable, "-m", "mlsys_lab." + experiment.lower(), "--output", str(root)],
                timeout=300 if experiment == "E02" else 120,
                check=True,
            )
            archive = io.BytesIO()
            metadata = {}
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name in ["summary.json", *files]:
                    path = root / name
                    if path.stat().st_size > 32 * 1024**2:
                        raise ValueError("trace_too_large")
                    raw = path.read_bytes()
                    bundle.writestr(name, raw)
                    metadata[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                bundle.writestr(
                    "manifest.json", json.dumps({"runtime": runtime, "files": metadata})
                )
            state.update("ready", {"/v1/status": archive.getvalue()})
        except (ValueError, OSError, subprocess.SubprocessError):
            state.update("failed")

    with server(Identity(), state, token, ("0.0.0.0", 8000)) as http:
        http.timeout = 1
        threading.Thread(target=compute, daemon=True).start()
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            http.handle_request()


if __name__ == "__main__":
    main()
