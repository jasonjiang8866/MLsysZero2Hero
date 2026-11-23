"""Result-only HTTP service shared by the smoke and profiler workers."""

import hmac
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class State:
    """Publish phase and immutable payloads together under a lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._phase = "starting"
        self._payloads: dict[str, bytes] = {}

    def update(self, phase: str, files: dict[str, bytes] | None = None) -> None:
        with self._lock:
            self._phase, self._payloads = phase, files or {}

    def response(self, path: str, runtime) -> tuple[int, bytes]:
        with self._lock:
            if path == "/v1/status" and path not in self._payloads:
                return 200, json.dumps(
                    {"phase": self._phase, "runtime_fingerprint": runtime.fingerprint()}
                ).encode()
            if self._phase != "ready":
                return 503, b"{}"
            data = self._payloads.get(path)
            return (200, data) if data is not None else (404, b"{}")


def server(runtime, state: State, token: str, address: tuple[str, int]) -> HTTPServer:
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        raise ValueError("result_capability_invalid")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, *_args):
            pass  # Headers, tokens and raw paths never enter logs.

        def do_GET(self):
            if not hmac.compare_digest(
                self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()
            ):
                status, data = 401, b"{}"
            else:
                status, data = state.response(self.path, runtime)
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/zip" if data.startswith(b"PK\x03\x04") else "application/json",
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (OSError, TimeoutError):
                pass

    return HTTPServer(address, Handler)
