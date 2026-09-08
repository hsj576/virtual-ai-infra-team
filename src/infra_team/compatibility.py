"""Deterministic autonomy policy and candidate preflight checks."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from .candidate_manifest import CandidateManifest


class PreflightError(ValueError):
    """A trusted manifest is incompatible with the approved local policy."""


def _parse_clock(value: str, field_name: str) -> datetime_time:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise PreflightError(f"{field_name} must use HH:MM in 24-hour time")
    hour, minute = value.split(":", 1)
    return datetime_time(int(hour), int(minute))


@dataclass(frozen=True)
class MaintenanceWindow:
    """Local-time maintenance authorization, including cross-midnight windows."""

    enabled: bool = False
    timezone: str = "Asia/Shanghai"
    start: str = "02:00"
    end: str = "05:00"
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        _parse_clock(self.start, "maintenance_window.start")
        _parse_clock(self.end, "maintenance_window.end")
        if self.start == self.end:
            raise PreflightError("maintenance window start and end must differ")
        if not self.weekdays or any(day not in range(7) for day in self.weekdays):
            raise PreflightError("maintenance_window.weekdays must contain integers 0-6")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise PreflightError(
                f"maintenance_window.timezone is unknown: {self.timezone}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timezone": self.timezone,
            "start": self.start,
            "end": self.end,
            "weekdays": list(self.weekdays),
        }

    def allows(self, moment: datetime | None = None) -> bool:
        if not self.enabled:
            return True
        moment = moment or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        local = moment.astimezone(ZoneInfo(self.timezone))
        start = _parse_clock(self.start, "maintenance_window.start")
        end = _parse_clock(self.end, "maintenance_window.end")
        current = local.timetz().replace(tzinfo=None)
        if start < end:
            return local.weekday() in self.weekdays and start <= current < end
        # For a cross-midnight window, the post-midnight segment belongs to
        # the weekday on which the window started.
        if current >= start:
            owner_weekday = local.weekday()
        elif current < end:
            owner_weekday = (local.weekday() - 1) % 7
        else:
            return False
        return owner_weekday in self.weekdays

    def next_start(self, moment: datetime | None = None) -> datetime:
        """Return the next authorized window start as an aware datetime."""
        moment = moment or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        zone = ZoneInfo(self.timezone)
        local = moment.astimezone(zone)
        start = _parse_clock(self.start, "maintenance_window.start")
        if self.allows(moment):
            return local
        for offset in range(0, 8):
            day = local.date() + timedelta(days=offset)
            if day.weekday() not in self.weekdays:
                continue
            candidate = datetime.combine(day, start, tzinfo=zone)
            if candidate > local:
                return candidate
        raise PreflightError("maintenance window has no next start")


@dataclass(frozen=True)
class AutonomyPolicy:
    schema_version: int = 1
    enabled: bool = True
    trusted_registries: tuple[str, ...] = ("builtin",)
    trusted_namespaces: tuple[str, ...] = ("z-lab", "mlx-community", "Qwen")
    allowed_kinds: tuple[str, ...] = ("acceleration_plugin",)
    max_download_gb_per_candidate: float = 20.0
    max_total_candidate_store_gb: float = 60.0
    max_experiment_minutes: int = 45
    max_peak_memory_gb: float = 40.0
    maintenance_window: MaintenanceWindow = field(default_factory=MaintenanceWindow)
    watch_poll_interval_seconds: int = 300
    watch_backoff_base_seconds: int = 60
    watch_backoff_cap_seconds: int = 3600
    watch_no_improvement_recheck_seconds: int = 86400
    notifications_enabled: bool = False
    notifications_macos_notification_center: bool = False
    auto_prepare: bool = True
    auto_experiment: bool = True
    auto_promote_acceleration_plugin: bool = True
    minimum_speedup_percent: float = 10.0
    require_all_quality_gates: bool = True
    require_zero_errors: bool = True
    allow_remote_code: bool = False
    approved_launch_templates: tuple[str, ...] = ("mlx_vlm_dflash2",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "trusted_registries": list(self.trusted_registries),
            "trusted_namespaces": list(self.trusted_namespaces),
            "allowed_kinds": list(self.allowed_kinds),
            "max_download_gb_per_candidate": self.max_download_gb_per_candidate,
            "max_total_candidate_store_gb": self.max_total_candidate_store_gb,
            "max_experiment_minutes": self.max_experiment_minutes,
            "max_peak_memory_gb": self.max_peak_memory_gb,
            "maintenance_window": self.maintenance_window.to_dict(),
            "watch": {
                "poll_interval_seconds": self.watch_poll_interval_seconds,
                "backoff_base_seconds": self.watch_backoff_base_seconds,
                "backoff_cap_seconds": self.watch_backoff_cap_seconds,
                "no_improvement_recheck_seconds": (
                    self.watch_no_improvement_recheck_seconds
                ),
            },
            "notifications": {
                "enabled": self.notifications_enabled,
                "macos_notification_center": (
                    self.notifications_macos_notification_center
                ),
            },
            "auto_prepare": self.auto_prepare,
            "auto_experiment": self.auto_experiment,
            "auto_promote": {
                "acceleration_plugin": self.auto_promote_acceleration_plugin,
                "quantization_variant": False,
                "runtime_upgrade": False,
                "target_model": False,
            },
            "minimum_speedup_percent": self.minimum_speedup_percent,
            "require_all_quality_gates": self.require_all_quality_gates,
            "require_zero_errors": self.require_zero_errors,
            "allow_remote_code": self.allow_remote_code,
            "approved_launch_templates": list(self.approved_launch_templates),
        }

    @property
    def policy_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @property
    def evaluation_policy_hash(self) -> str:
        """Hash only rules that can change experiment or promotion evidence.

        Poll cadence, notification choice and maintenance-window timing must not
        cause a previously validated Manifest to be re-executed.
        """
        payload = self.to_dict()
        payload.pop("watch", None)
        payload.pop("notifications", None)
        payload.pop("maintenance_window", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> "AutonomyPolicy":
        with Path(path).open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise PreflightError("autonomy policy schema_version must be 1")
        allowed = {
            "schema_version",
            "enabled",
            "trusted_registries",
            "trusted_namespaces",
            "allowed_kinds",
            "approved_launch_templates",
            "max_download_gb_per_candidate",
            "max_total_candidate_store_gb",
            "max_experiment_minutes",
            "max_peak_memory_gb",
            "maintenance_window",
            "watch",
            "notifications",
            "auto_prepare",
            "auto_experiment",
            "auto_promote",
            "minimum_speedup_percent",
            "require_all_quality_gates",
            "require_zero_errors",
            "allow_remote_code",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise PreflightError(
                f"autonomy policy contains unknown field(s): {', '.join(sorted(unknown))}"
            )
        auto_promote = raw.get("auto_promote") or {}
        maintenance = raw.get("maintenance_window") or {}
        watch = raw.get("watch") or {}
        notifications = raw.get("notifications") or {}
        if not all(
            isinstance(value, dict)
            for value in (auto_promote, maintenance, watch, notifications)
        ):
            raise PreflightError("autonomy policy nested sections must be objects")
        unknown_promotions = set(auto_promote) - {
            "acceleration_plugin",
            "quantization_variant",
            "runtime_upgrade",
            "target_model",
        }
        if unknown_promotions:
            raise PreflightError("autonomy policy has an unknown promotion kind")
        unknown_window = set(maintenance) - {
            "enabled",
            "timezone",
            "start",
            "end",
            "weekdays",
        }
        if unknown_window:
            raise PreflightError("maintenance_window contains an unknown field")
        unknown_watch = set(watch) - {
            "poll_interval_seconds",
            "backoff_base_seconds",
            "backoff_cap_seconds",
            "no_improvement_recheck_seconds",
        }
        if unknown_watch:
            raise PreflightError("watch contains an unknown field")
        unknown_notifications = set(notifications) - {
            "enabled",
            "macos_notification_center",
        }
        if unknown_notifications:
            raise PreflightError("notifications contains an unknown field")
        if raw.get("allow_remote_code") is not False:
            raise PreflightError("autonomy policy must disable remote code")
        if raw.get("require_all_quality_gates") is not True:
            raise PreflightError("autonomy policy must require all quality gates")
        if raw.get("require_zero_errors") is not True:
            raise PreflightError("autonomy policy must require zero errors")
        if auto_promote.get("target_model") is not False:
            raise PreflightError("L3 policy must not auto-promote target models")
        policy = cls(
            schema_version=1,
            enabled=bool(raw.get("enabled", False)),
            trusted_registries=tuple(raw.get("trusted_registries") or ()),
            trusted_namespaces=tuple(raw.get("trusted_namespaces") or ()),
            allowed_kinds=tuple(raw.get("allowed_kinds") or ()),
            max_download_gb_per_candidate=float(
                raw.get("max_download_gb_per_candidate", 0)
            ),
            max_total_candidate_store_gb=float(
                raw.get("max_total_candidate_store_gb", 0)
            ),
            max_experiment_minutes=int(raw.get("max_experiment_minutes", 0)),
            max_peak_memory_gb=float(raw.get("max_peak_memory_gb", 0)),
            maintenance_window=MaintenanceWindow(
                enabled=bool(maintenance.get("enabled", False)),
                timezone=str(maintenance.get("timezone", "Asia/Shanghai")),
                start=str(maintenance.get("start", "02:00")),
                end=str(maintenance.get("end", "05:00")),
                weekdays=tuple(
                    int(day)
                    for day in maintenance.get(
                        "weekdays", (0, 1, 2, 3, 4, 5, 6)
                    )
                ),
            ),
            watch_poll_interval_seconds=int(
                watch.get("poll_interval_seconds", 300)
            ),
            watch_backoff_base_seconds=int(
                watch.get("backoff_base_seconds", 60)
            ),
            watch_backoff_cap_seconds=int(
                watch.get("backoff_cap_seconds", 3600)
            ),
            watch_no_improvement_recheck_seconds=int(
                watch.get("no_improvement_recheck_seconds", 86400)
            ),
            notifications_enabled=bool(notifications.get("enabled", False)),
            notifications_macos_notification_center=bool(
                notifications.get("macos_notification_center", False)
            ),
            auto_prepare=bool(raw.get("auto_prepare", False)),
            auto_experiment=bool(raw.get("auto_experiment", False)),
            auto_promote_acceleration_plugin=bool(
                auto_promote.get("acceleration_plugin", False)
            ),
            minimum_speedup_percent=float(raw.get("minimum_speedup_percent", 0)),
            require_all_quality_gates=bool(
                raw.get("require_all_quality_gates", False)
            ),
            require_zero_errors=bool(raw.get("require_zero_errors", False)),
            allow_remote_code=bool(raw.get("allow_remote_code", False)),
            approved_launch_templates=tuple(
                raw.get("approved_launch_templates") or ("mlx_vlm_dflash2",)
            ),
        )
        if (
            policy.max_download_gb_per_candidate <= 0
            or policy.max_total_candidate_store_gb <= 0
            or policy.max_peak_memory_gb <= 0
            or policy.minimum_speedup_percent <= 0
        ):
            raise PreflightError("autonomy policy budgets must be positive")
        if (
            policy.watch_poll_interval_seconds <= 0
            or policy.watch_backoff_base_seconds <= 0
            or policy.watch_backoff_cap_seconds < policy.watch_backoff_base_seconds
            or policy.watch_no_improvement_recheck_seconds <= 0
        ):
            raise PreflightError("watch intervals and backoff bounds are invalid")
        if (
            policy.notifications_macos_notification_center
            and not policy.notifications_enabled
        ):
            raise PreflightError(
                "macOS notifications require notifications.enabled=true"
            )
        return policy


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value or "")
    return tuple(int(part) for part in parts[:4])


def current_platform() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") and platform.system() == "Darwin":
        return "apple-silicon"
    return machine or "unknown"


def preflight_candidate(
    manifest: CandidateManifest,
    policy: AutonomyPolicy,
    environment: dict[str, Any],
    registry_name: str = "builtin",
) -> dict[str, Any]:
    """Check trust, compatibility and resource bounds without downloading."""
    checks: list[dict[str, Any]] = []

    def require(condition: bool, name: str, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise PreflightError(detail)

    namespace = manifest.source.repo_id.split("/", 1)[0]
    require(policy.enabled, "policy_enabled", "autonomy policy is disabled")
    require(
        registry_name in policy.trusted_registries,
        "trusted_registry",
        f"registry '{registry_name}' is not trusted",
    )
    require(
        namespace in policy.trusted_namespaces,
        "trusted_namespace",
        f"repository namespace '{namespace}' is not trusted",
    )
    require(
        manifest.kind in policy.allowed_kinds,
        "allowed_kind",
        f"candidate kind '{manifest.kind}' is not allowed",
    )
    require(
        manifest.launch_template in policy.approved_launch_templates,
        "launch_template",
        f"launch template '{manifest.launch_template}' is not approved",
    )
    require(
        not manifest.allow_remote_code and not policy.allow_remote_code,
        "remote_code",
        "remote code is not allowed",
    )
    require(
        manifest.review_status == "approved",
        "license_review",
        f"license review status is '{manifest.review_status}'",
    )
    require(
        manifest.status == "active",
        "manifest_status",
        f"manifest status is '{manifest.status}'",
    )
    require(
        manifest.estimated_download_gb <= policy.max_download_gb_per_candidate,
        "download_budget",
        "candidate exceeds per-candidate download budget",
    )
    require(
        manifest.max_peak_memory_gb <= policy.max_peak_memory_gb,
        "memory_budget",
        "candidate exceeds peak memory budget",
    )
    require(
        manifest.compatibility.target_model
        == "mlx-community/Qwen3.8-27B-4bit",
        "target_model",
        "candidate target model is incompatible",
    )
    require(
        manifest.compatibility.runtime == "mlx-vlm",
        "runtime",
        "candidate runtime is incompatible",
    )
    runtime_version = str((environment.get("packages") or {}).get("mlx-vlm") or "")
    require(
        _version_tuple(runtime_version)
        >= _version_tuple(manifest.compatibility.runtime_min_version),
        "runtime_version",
        (
            f"mlx-vlm {runtime_version or 'unknown'} is older than "
            f"{manifest.compatibility.runtime_min_version}"
        ),
    )
    actual_platform = (
        "apple-silicon"
        if environment.get("arch") == "arm64"
        and str(environment.get("chip") or "").startswith("Apple")
        else current_platform()
    )
    require(
        manifest.compatibility.platform == actual_platform,
        "platform",
        f"candidate requires {manifest.compatibility.platform}, got {actual_platform}",
    )
    disk_free = environment.get("disk_free_gb")
    if disk_free is not None:
        require(
            float(disk_free) >= manifest.estimated_download_gb + 2.0,
            "disk_free",
            "insufficient free disk for candidate preparation",
        )
    return {
        "passed": True,
        "manifest_id": manifest.id,
        "manifest_hash": manifest.manifest_hash,
        "policy_hash": policy.policy_hash,
        "checks": checks,
    }
