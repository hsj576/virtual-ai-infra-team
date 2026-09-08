"""Environment probe for macOS / Apple Silicon.

Collects a deterministic snapshot of the machine so that any optimization
result can be interpreted (and reproduced) later. Read-only: this module
never mutates system state.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

TIMEOUT = 15


def _sh(cmd: list[str], timeout: int = TIMEOUT) -> str:
    """Run a read-only shell probe, returning stdout or '' on any failure."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _sysctl(key: str) -> str:
    return _sh(["sysctl", "-n", key])


def _chip() -> str:
    chip = _sysctl("machdep.cpu.brand_string")
    return chip or platform.processor() or "unknown"


def _gpu_cores() -> int | None:
    """Parse GPU core count from system_profiler."""
    raw = _sh(["system_profiler", "-json", "SPDisplaysDataType"], timeout=30)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for entry in data.get("SPDisplaysDataType", []):
        for key in ("sppci_cores", "spdisplays_ncores"):
            val = entry.get(key)
            if val:
                try:
                    return int(str(val).strip())
                except ValueError:
                    continue
    return None


def _cpu_cores() -> dict[str, int | None]:
    def as_int(v: str) -> int | None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    # perflevel0 is the high-power cluster, perflevel1 the efficiency cluster.
    # Naming follows sysctl rather than marketing names, which vary by chip.
    return {
        "total": as_int(_sysctl("hw.ncpu")),
        "perflevel0": as_int(_sysctl("hw.perflevel0.logicalcpu")),
        "perflevel1": as_int(_sysctl("hw.perflevel1.logicalcpu")),
    }


def _memory_gb() -> float | None:
    raw = _sysctl("hw.memsize")
    try:
        return round(int(raw) / (1024**3), 2)
    except (TypeError, ValueError):
        return None


def _disk_free_gb(path: str) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 2)
    except OSError:
        return None


def _pkg_version(module: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(module)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _on_ac_power() -> bool | None:
    """Power source matters: battery mode throttles sustained GPU work."""
    raw = _sh(["pmset", "-g", "batt"])
    if not raw:
        return None
    lowered = raw.lower()
    if "'ac power'" in lowered or "ac power" in lowered.split("\n")[0]:
        return True
    if "battery power" in lowered:
        return False
    return None


def _thermal_pressure() -> str | None:
    raw = _sh(["pmset", "-g", "therm"])
    if not raw:
        return None
    for line in raw.splitlines():
        if "CPU_Speed_Limit" in line:
            return line.strip()
    return raw.splitlines()[-1].strip() if raw else None


def _memory_pressure_free_pct() -> int | None:
    raw = _sh(["memory_pressure"])
    for line in raw.splitlines():
        if "System-wide memory free percentage" in line:
            digits = "".join(c for c in line if c.isdigit())
            try:
                return int(digits)
            except ValueError:
                return None
    return None


def _hf_cache_size_gb(repo_id: str) -> float | None:
    """Report cached weight size without triggering a download."""
    cache = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )
    hub = os.path.join(cache, "hub") if not cache.endswith("hub") else cache
    folder = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    if not os.path.isdir(folder):
        return None
    snapshots_root = os.path.join(folder, "snapshots")
    try:
        snapshots = [
            os.path.join(snapshots_root, name)
            for name in os.listdir(snapshots_root)
            if os.path.isdir(os.path.join(snapshots_root, name))
        ]
    except OSError:
        return None
    if not snapshots:
        return None
    # Count the logical files reachable from the newest complete snapshot,
    # not the blob store. Blob stores may contain resumable downloads and
    # several hardlinks to the same bytes, which would wildly over-report.
    snapshot = max(snapshots, key=os.path.getmtime)
    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for root, _dirs, files in os.walk(snapshot):
        for f in files:
            fp = os.path.join(root, f)
            try:
                stat = os.stat(fp, follow_symlinks=True)
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen_inodes:
                continue
            seen_inodes.add(identity)
            total += stat.st_size
    return round(total / (1024**3), 2)


@dataclass
class Environment:
    """Deterministic environment snapshot."""

    chip: str
    cpu_cores: dict[str, int | None]
    gpu_cores: int | None
    unified_memory_gb: float | None
    macos_version: str
    macos_build: str
    arch: str
    python_version: str
    disk_free_gb: float | None
    packages: dict[str, str | None]
    power: dict[str, Any]
    model_cache: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe(models: list[str] | None = None) -> Environment:
    """Collect the environment snapshot. Read-only."""
    return Environment(
        chip=_chip(),
        cpu_cores=_cpu_cores(),
        gpu_cores=_gpu_cores(),
        unified_memory_gb=_memory_gb(),
        macos_version=platform.mac_ver()[0] or _sh(["sw_vers", "-productVersion"]),
        macos_build=_sh(["sw_vers", "-buildVersion"]),
        arch=platform.machine(),
        python_version=platform.python_version(),
        disk_free_gb=_disk_free_gb(os.path.expanduser("~")),
        packages={
            name: _pkg_version(name)
            for name in ("mlx", "mlx-vlm", "mlx-lm", "transformers", "huggingface_hub")
        },
        power={
            "on_ac_power": _on_ac_power(),
            "thermal_pressure": _thermal_pressure(),
            "memory_free_pct": _memory_pressure_free_pct(),
        },
        model_cache={m: _hf_cache_size_gb(m) for m in (models or [])},
    )


if __name__ == "__main__":
    env = probe(
        ["mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2"]
    )
    print(json.dumps(env.to_dict(), indent=2))
