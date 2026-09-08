"""Lifecycle manager for an MLX-VLM OpenAI-compatible local service.

The public API address remains stable while the internal model recipe can be
restarted. The first implementation uses a short maintenance window rather
than pretending a 48GB Mac can always hold two 27B targets for blue/green
switching.

Safety invariants:
- no root;
- no shell=True;
- refuse an occupied port not owned by this workspace;
- stop only a wrapper that authenticates over a private localhost control channel;
- persist enough state to recover across CLI invocations.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

SERVICES_DIRNAME = os.path.join(".infra-team", "services")
WORKER = os.path.join(os.path.dirname(__file__), "managed_server_worker.py")


class ServiceError(RuntimeError):
    pass


@dataclass
class ServiceSpec:
    name: str
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    draft_model: str | None = None
    draft_kind: str | None = None
    draft_block_size: int | None = None
    max_tokens: int | None = None
    enable_thinking: bool = False
    target_revision: str | None = None
    candidate_manifest_id: str | None = None
    candidate_manifest_hash: str | None = None
    runtime_version: str | None = None
    resolved_model_path: str | None = None

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_mlx_server_command(spec: ServiceSpec, python_executable: str) -> list[str]:
    """Translate a validated ServiceSpec into a fixed command template."""
    if not (1 <= int(spec.port) <= 65535):
        raise ServiceError(f"invalid port: {spec.port}")
    command = [
        python_executable,
        "-m",
        "mlx_vlm.server",
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--model",
        spec.resolved_model_path or spec.model,
    ]
    if spec.draft_model:
        command.extend(["--draft-model", spec.draft_model])
    if spec.draft_kind:
        if spec.draft_kind not in ("dflash", "eagle3", "mtp"):
            raise ServiceError(f"unsupported draft kind: {spec.draft_kind}")
        command.extend(["--draft-kind", spec.draft_kind])
    if spec.draft_block_size is not None:
        block = int(spec.draft_block_size)
        if block < 1 or block > 32:
            raise ServiceError(f"invalid draft block size: {block}")
        command.extend(["--draft-block-size", str(block)])
    if spec.max_tokens is not None:
        command.extend(["--max-tokens", str(int(spec.max_tokens))])
    if spec.enable_thinking:
        command.append("--enable-thinking")
    return command


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    """Write state atomically so interruption cannot leave partial JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def _health(base_url: str, timeout: float = 2.0) -> tuple[bool, str]:
    request = urllib.request.Request(base_url.rstrip("/") + "/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            if 200 <= response.status < 300:
                return True, body[:300]
            return False, f"HTTP {response.status}: {body[:200]}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)


def _port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((check_host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ManagedService:
    """Persistent process state for one named local model service."""

    def __init__(
        self,
        root: str,
        name: str = "default",
        python_executable: str | None = None,
    ) -> None:
        self.root = os.path.abspath(root)
        self.name = name
        self.python_executable = python_executable or sys.executable
        self.directory = os.path.join(self.root, SERVICES_DIRNAME, name)
        self.state_path = os.path.join(self.directory, "state.json")
        self.spec_path = os.path.join(self.directory, "spec.json")
        self.control_state_path = os.path.join(self.directory, "control.json")
        self.server_log = os.path.join(self.directory, "server.log")
        self.wrapper_log = os.path.join(self.directory, "wrapper.log")
        self.active_recipe_path = os.path.join(self.directory, "active_recipe.json")
        self.previous_recipe_path = os.path.join(self.directory, "previous_recipe.json")
        self.recipe_history_dir = os.path.join(self.directory, "recipe_history")

    def _read_state(self) -> dict[str, Any] | None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _write_state(self, state: dict[str, Any]) -> None:
        _atomic_json(self.state_path, state)

    def saved_spec(self) -> ServiceSpec | None:
        """Return the last managed recipe without exposing control capability."""
        state = self._read_state()
        raw = (state or {}).get("spec")
        if not isinstance(raw, dict):
            try:
                with open(self.spec_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return None
        try:
            return ServiceSpec(**raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_json(path: str) -> dict[str, Any] | None:
        try:
            with open(path, encoding="utf-8") as fh:
                value = json.load(fh)
            return value if isinstance(value, dict) else None
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def active_recipe(self) -> dict[str, Any] | None:
        return self._read_json(self.active_recipe_path)

    def previous_recipe(self) -> dict[str, Any] | None:
        return self._read_json(self.previous_recipe_path)

    def recipe_from_spec(
        self,
        spec: ServiceSpec,
        source: str,
        run_id: str | None = None,
        candidate_id: str | None = None,
        online_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "service_name": self.name,
            "candidate_id": candidate_id or ("accelerated" if spec.draft_model else "baseline"),
            "service_spec": spec.to_dict(),
            "target": {
                "model": spec.model,
                "revision": spec.target_revision,
            },
            "runtime": {
                "name": "mlx-vlm",
                "version": spec.runtime_version,
            },
            "accelerator": (
                {
                    "manifest_id": spec.candidate_manifest_id,
                    "manifest_hash": spec.candidate_manifest_hash,
                    "draft_model": spec.draft_model,
                    "draft_kind": spec.draft_kind,
                    "draft_block_size": spec.draft_block_size,
                }
                if spec.draft_model
                else None
            ),
            "source": source,
            "promoted_from_run": run_id,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "online_quality": online_quality,
        }

    def atomically_promote(self, recipe: dict[str, Any]) -> dict[str, Any]:
        """Commit an Active Recipe only after its service is healthy.

        The caller performs online quality verification first. This method
        additionally proves that the currently managed healthy service matches
        the exact recipe before replacing the active pointer.
        """
        if not isinstance(recipe, dict) or recipe.get("schema_version") != 1:
            raise ServiceError("active recipe schema_version must be 1")
        if recipe.get("service_name") != self.name:
            raise ServiceError("active recipe service name mismatch")
        spec_raw = recipe.get("service_spec")
        if not isinstance(spec_raw, dict):
            raise ServiceError("active recipe has no service_spec")
        try:
            spec = ServiceSpec(**spec_raw)
        except (TypeError, ValueError) as exc:
            raise ServiceError(f"active recipe service spec is invalid: {exc}") from exc
        status = self.status()
        if not status.get("healthy"):
            raise ServiceError("cannot promote an active recipe before service is healthy")
        try:
            running_spec = ServiceSpec(**(status.get("spec") or {})).to_dict()
        except (TypeError, ValueError) as exc:
            raise ServiceError("running service has an invalid spec") from exc
        if running_spec != spec.to_dict():
            raise ServiceError("running service does not match the proposed active recipe")

        os.makedirs(self.recipe_history_dir, exist_ok=True)
        existing = self.active_recipe()
        if existing is not None:
            _atomic_json(self.previous_recipe_path, existing)
        history_hash = hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        history_path = os.path.join(
            self.recipe_history_dir,
            f"{time.strftime('%Y%m%d-%H%M%S')}-{history_hash}.json",
        )
        try:
            _atomic_json(history_path, recipe)
            next_path = self.active_recipe_path + ".next"
            _atomic_json(next_path, recipe)
            os.replace(next_path, self.active_recipe_path)
        except OSError as exc:
            raise ServiceError(f"failed to atomically persist active recipe: {exc}") from exc
        return recipe

    @staticmethod
    def _control_request(
        state: dict[str, Any], action: str, timeout: float = 2.0
    ) -> dict[str, Any] | None:
        """Authenticate to the wrapper's private localhost control socket."""
        try:
            host = str(state["control_host"])
            port = int(state["control_port"])
            signature = str(state["signature"])
        except (KeyError, TypeError, ValueError):
            return None
        if host != "127.0.0.1" or not (1 <= port <= 65535) or not signature:
            return None
        payload = json.dumps({"action": action, "signature": signature}) + "\n"
        try:
            with socket.create_connection((host, port), timeout=timeout) as conn:
                conn.sendall(payload.encode("utf-8"))
                conn.settimeout(timeout)
                response = b""
                while not response.endswith(b"\n") and len(response) < 8192:
                    chunk = conn.recv(8192 - len(response))
                    if not chunk:
                        break
                    response += chunk
            parsed = json.loads(response.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else None
        except (OSError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _pid_matches(self, state: dict[str, Any]) -> bool:
        """Ownership proof: only the signed wrapper can answer this ping."""
        response = self._control_request(state, "ping")
        if not response or not response.get("ok"):
            return False
        try:
            return int(response["pid"]) == int(state["pid"])
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _public_state(state: dict[str, Any]) -> dict[str, Any]:
        public = dict(state)
        public.pop("signature", None)
        public.pop("control_port", None)
        public.pop("control_host", None)
        return public

    def status(self) -> dict[str, Any]:
        state = self._read_state()
        if not state:
            return {"name": self.name, "status": "not_deployed", "managed": False}
        managed = self._pid_matches(state)
        spec = state.get("spec") or {}
        base_url = f"http://{spec.get('host', '127.0.0.1')}:{spec.get('port', 8000)}"
        if spec.get("host") in ("0.0.0.0", "::"):
            base_url = f"http://127.0.0.1:{spec.get('port', 8000)}"
        healthy, detail = (
            _health(base_url)
            if managed
            else (False, "signed control channel unavailable")
        )
        # The signature is a capability token for stopping the process. Never
        # expose it through status output, CLI JSON or run artifacts.
        current = self._public_state(state)
        current.update(
            {
                "managed": managed,
                "healthy": healthy,
                "health_detail": detail,
                "status": "running" if managed and healthy else "unhealthy",
                "base_url": base_url,
            }
        )
        return current

    def _wait_health(
        self,
        base_url: str,
        timeout: float,
        process: subprocess.Popen[Any] | None = None,
    ) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout
        last = "not checked"
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                return False, f"server wrapper exited with code {process.returncode}"
            ok, last = _health(base_url, timeout=2.0)
            if ok:
                return True, last
            time.sleep(1.0)
        return False, last

    def _wait_control_ready(
        self,
        process: subprocess.Popen[Any],
        signature: str,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        expected_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ServiceError(
                    f"server wrapper exited before control channel was ready: {process.returncode}"
                )
            try:
                with open(self.control_state_path, encoding="utf-8") as fh:
                    control = json.load(fh)
                if (
                    control.get("signature_hash") == expected_hash
                    and int(control.get("pid")) == process.pid
                    and int(control.get("port")) > 0
                ):
                    return control
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
                pass
            time.sleep(0.1)
        process.terminate()
        raise ServiceError("server wrapper did not establish its signed control channel")

    @staticmethod
    def _resolve_target_snapshot(spec: ServiceSpec) -> ServiceSpec:
        if not spec.target_revision or spec.resolved_model_path:
            return spec
        if Path(spec.model).expanduser().is_dir():
            return replace(spec, resolved_model_path=str(Path(spec.model).expanduser().resolve()))
        try:
            resolved = snapshot_download(
                repo_id=spec.model,
                revision=spec.target_revision,
            )
        except Exception as exc:
            raise ServiceError(
                f"failed to resolve pinned target {spec.model}@{spec.target_revision}: {exc}"
            ) from exc
        return replace(spec, resolved_model_path=str(Path(resolved).resolve()))

    def start(self, spec: ServiceSpec, wait_timeout: float = 900.0) -> dict[str, Any]:
        if spec.name != self.name:
            raise ServiceError(
                f"service name mismatch: manager={self.name}, spec={spec.name}"
            )
        existing = self._read_state()
        if existing and self._pid_matches(existing):
            raise ServiceError(
                f"managed service '{self.name}' is already running; stop it first"
            )
        if _port_open(spec.host, spec.port):
            raise ServiceError(
                f"port {spec.port} is already occupied; refusing to stop or replace an unknown process"
            )

        spec = self._resolve_target_snapshot(spec)
        os.makedirs(self.directory, exist_ok=True)
        _atomic_json(self.spec_path, spec.to_dict())
        # This is our own ephemeral control file inside the project workspace.
        # Remove only this exact file; never touch unknown processes or paths.
        try:
            os.unlink(self.control_state_path)
        except FileNotFoundError:
            pass
        signature = secrets.token_hex(16)
        spec_hash = hashlib.sha256(
            json.dumps(spec.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()
        command = [
            self.python_executable,
            WORKER,
            "--spec",
            self.spec_path,
            "--signature",
            signature,
            "--python",
            self.python_executable,
            "--log",
            self.server_log,
            "--control-state",
            self.control_state_path,
        ]
        wrapper_fh = open(self.wrapper_log, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=wrapper_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            wrapper_fh.close()

        control = self._wait_control_ready(proc, signature)
        state = {
            "name": self.name,
            "status": "starting",
            "pid": proc.pid,
            "signature": signature,
            "control_host": control["host"],
            "control_port": control["port"],
            "spec_hash": spec_hash,
            "spec": spec.to_dict(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "base_url": spec.base_url,
            "server_log": self.server_log,
        }
        self._write_state(state)
        healthy, detail = self._wait_health(spec.base_url, wait_timeout, process=proc)
        if not healthy:
            # Only stop if ownership is still provable.
            if self._pid_matches(state):
                self._terminate_owned(state, timeout=20.0)
            state.update(status="failed", healthy=False, health_detail=detail)
            self._write_state(state)
            raise ServiceError(
                f"service did not become healthy within {wait_timeout}s: {detail}"
            )

        state.update(status="running", healthy=True, health_detail=detail)
        self._write_state(state)
        return self.status()

    def _terminate_owned(self, state: dict[str, Any], timeout: float) -> None:
        if not self._pid_matches(state):
            raise ServiceError(
                "refusing to stop: signed control-channel ownership check failed"
            )
        response = self._control_request(state, "stop", timeout=3.0)
        if not response or not response.get("ok"):
            raise ServiceError("managed wrapper rejected the authenticated stop request")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pid_matches(state):
                return
            time.sleep(0.2)
        raise ServiceError(
            "managed service did not stop after authenticated request; "
            "refusing force-kill automatically"
        )

    def stop(self, timeout: float = 30.0) -> dict[str, Any]:
        state = self._read_state()
        if not state:
            return {"name": self.name, "status": "not_deployed", "stopped": False}
        self._terminate_owned(state, timeout)
        state.update(
            status="stopped",
            healthy=False,
            stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self._write_state(state)
        return dict(self._public_state(state), stopped=True)
