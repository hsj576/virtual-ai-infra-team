"""Signed control wrapper around the fixed mlx_vlm.server command.

The wrapper exposes a private localhost control socket. A later CLI invocation
must present the random startup signature to ping or stop this exact process.
This proves ownership without depending on `ps` output and avoids ever sending
a signal to an unknown PID.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra_team.service_manager import ServiceSpec, build_mlx_server_command


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--control-state", required=True)
    args = parser.parse_args(argv)

    with open(args.spec, encoding="utf-8") as fh:
        raw = json.load(fh)
    spec = ServiceSpec(**raw)
    command = build_mlx_server_command(spec, args.python)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as log:
        log.write(f"\n=== managed service start: {spec.name} ===\n")
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        class ControlHandler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request = json.loads(self.rfile.readline(8192).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.wfile.write(b'{"ok":false,"error":"invalid request"}\n')
                    return
                if request.get("signature") != args.signature:
                    self.wfile.write(b'{"ok":false,"error":"signature mismatch"}\n')
                    return
                action = request.get("action")
                if action == "ping":
                    payload = {"ok": True, "pid": os.getpid(), "child_pid": proc.pid}
                elif action == "stop":
                    if proc.poll() is None:
                        proc.terminate()
                    payload = {"ok": True, "stopping": True}
                else:
                    payload = {"ok": False, "error": "unknown action"}
                self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))

        control = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ControlHandler)
        control.daemon_threads = True
        control.allow_reuse_address = False
        control_port = int(control.server_address[1])
        _atomic_json(
            args.control_state,
            {
                "host": "127.0.0.1",
                "port": control_port,
                "pid": os.getpid(),
                "signature_hash": __import__("hashlib").sha256(
                    args.signature.encode("utf-8")
                ).hexdigest(),
            },
        )
        control_thread = threading.Thread(target=control.serve_forever, daemon=True)
        control_thread.start()

        def forward(signum, _frame):
            if proc.poll() is None:
                proc.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                code = 124
        finally:
            control.shutdown()
            control.server_close()
            control_thread.join(timeout=2)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
