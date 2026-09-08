"""Durable local Recipe Memory backed by SQLite.

Immutable run bundles remain the evidence source. This database is only a
queryable index and prior store: every row points back to the run/evolution
artifacts that produced it, and every prior still requires local validation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .candidate_registry import ResolvedCandidate

SCHEMA_VERSION = 2


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def environment_payload(
    environment: dict[str, Any],
    target_model: str,
    target_revision: str | None,
) -> dict[str, Any]:
    packages = environment.get("packages") or {}
    return {
        "chip": environment.get("chip"),
        "gpu_cores": environment.get("gpu_cores"),
        "unified_memory_gb": environment.get("unified_memory_gb"),
        "os": environment.get("macos_version") or environment.get("os_version"),
        "runtime_name": "mlx-vlm",
        "runtime_version": packages.get("mlx-vlm"),
        "mlx_version": packages.get("mlx"),
        "target_model": target_model,
        "target_revision": target_revision,
    }


def environment_fingerprint(
    environment: dict[str, Any],
    target_model: str,
    target_revision: str | None,
) -> str:
    return hashlib.sha256(
        _canonical(environment_payload(environment, target_model, target_revision)).encode(
            "utf-8"
        )
    ).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


class RecipeMemory:
    """SQLite index of environments, candidate outcomes and promotions."""

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).resolve()
        self.path = (
            root_path
            if root_path.suffix == ".db"
            else root_path / ".infra-team" / "memory" / "recipes.db"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _migrate(self) -> None:
        with self._connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Recipe Memory schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version < 1:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS environments (
                        fingerprint TEXT PRIMARY KEY,
                        chip TEXT,
                        gpu_cores INTEGER,
                        unified_memory_gb REAL,
                        os_version TEXT,
                        runtime_name TEXT NOT NULL,
                        runtime_version TEXT,
                        mlx_version TEXT,
                        target_model TEXT NOT NULL,
                        target_revision TEXT,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_used_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS candidate_runs (
                        run_id TEXT NOT NULL,
                        candidate_id TEXT NOT NULL,
                        evolution_id TEXT,
                        environment_fingerprint TEXT NOT NULL,
                        target_model TEXT NOT NULL,
                        target_revision TEXT,
                        manifest_id TEXT,
                        manifest_hash TEXT,
                        source_revision TEXT,
                        candidate_params_json TEXT NOT NULL,
                        benchmark_suite TEXT,
                        quality_pass INTEGER NOT NULL,
                        quality_json TEXT NOT NULL,
                        performance_json TEXT NOT NULL,
                        speedup_percent REAL,
                        accepted INTEGER NOT NULL,
                        qualified INTEGER NOT NULL,
                        failure_stage TEXT,
                        failure_reason TEXT,
                        runtime_version TEXT,
                        artifact_path TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, candidate_id),
                        FOREIGN KEY (environment_fingerprint)
                            REFERENCES environments(fingerprint)
                    );
                    CREATE INDEX IF NOT EXISTS idx_candidate_prior
                        ON candidate_runs(
                            target_model, manifest_hash, candidate_id,
                            environment_fingerprint, created_at DESC
                        );
                    CREATE TABLE IF NOT EXISTS promotions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL UNIQUE,
                        evolution_id TEXT,
                        environment_fingerprint TEXT NOT NULL,
                        previous_recipe_json TEXT NOT NULL,
                        selected_recipe_json TEXT NOT NULL,
                        promoted_at TEXT NOT NULL,
                        online_verified INTEGER NOT NULL,
                        rolled_back INTEGER NOT NULL DEFAULT 0,
                        rollback_reason TEXT,
                        active_seconds REAL,
                        FOREIGN KEY (environment_fingerprint)
                            REFERENCES environments(fingerprint)
                    );
                    CREATE TABLE IF NOT EXISTS evolution_runs (
                        evolution_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        trigger_json TEXT NOT NULL,
                        artifact_path TEXT NOT NULL,
                        supervisor_run_id TEXT,
                        error TEXT,
                        started_at TEXT NOT NULL,
                        completed_at TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    """
                )
            if version < 2:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS watch_states (
                        context_key TEXT PRIMARY KEY,
                        service_name TEXT NOT NULL,
                        manifest_id TEXT NOT NULL,
                        manifest_hash TEXT NOT NULL,
                        environment_fingerprint TEXT NOT NULL,
                        policy_hash TEXT NOT NULL,
                        snapshot_hash TEXT,
                        status TEXT NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at TEXT,
                        last_evolution_id TEXT,
                        error_class TEXT,
                        last_error TEXT,
                        first_seen_at TEXT NOT NULL,
                        last_attempt_at TEXT,
                        completed_at TEXT,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_watch_due
                        ON watch_states(status, next_attempt_at, updated_at);
                    CREATE TABLE IF NOT EXISTS notification_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        template_id TEXT NOT NULL,
                        fields_hash TEXT NOT NULL,
                        sent INTEGER NOT NULL,
                        detail TEXT,
                        created_at TEXT NOT NULL
                    );
                    PRAGMA user_version = 2;
                    """
                )

    def _upsert_environment(
        self,
        conn: sqlite3.Connection,
        environment: dict[str, Any],
        target_model: str,
        target_revision: str | None,
    ) -> str:
        payload = environment_payload(environment, target_model, target_revision)
        fingerprint = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        now = _now()
        conn.execute(
            """
            INSERT INTO environments(
                fingerprint, chip, gpu_cores, unified_memory_gb, os_version,
                runtime_name, runtime_version, mlx_version, target_model,
                target_revision, payload_json, created_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET last_used_at=excluded.last_used_at
            """,
            (
                fingerprint,
                payload["chip"],
                payload["gpu_cores"],
                payload["unified_memory_gb"],
                payload["os"],
                payload["runtime_name"],
                payload["runtime_version"],
                payload["mlx_version"],
                target_model,
                target_revision,
                _canonical(payload),
                now,
                now,
            ),
        )
        return fingerprint

    @staticmethod
    def _target_revision(
        resolved_candidates: dict[str, ResolvedCandidate],
    ) -> str | None:
        return next(
            (
                candidate.target_revision
                for candidate in resolved_candidates.values()
                if candidate.target_revision
            ),
            None,
        )

    def record_supervisor_summary(
        self,
        summary: dict[str, Any],
        resolved_candidates: dict[str, ResolvedCandidate],
        evolution_id: str | None = None,
        previous_recipe: dict[str, Any] | None = None,
        selected_recipe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = str(summary.get("run_id") or "")
        if not run_id:
            raise ValueError("supervisor summary has no run_id")
        environment = summary.get("environment") or {}
        target_model = next(
            (
                str((item.get("spec") or {}).get("target_model"))
                for item in [summary.get("baseline") or {}, *(summary.get("candidates") or [])]
                if (item.get("spec") or {}).get("target_model")
            ),
            "unknown",
        )
        target_revision = self._target_revision(resolved_candidates)
        verdict = summary.get("verdict") or {}
        evaluations = {
            item.get("id"): item
            for item in verdict.get("evaluations") or []
            if isinstance(item, dict)
        }
        rows = [summary.get("baseline") or {}, *(summary.get("candidates") or [])]
        written = 0
        with self._connect() as conn:
            fingerprint = self._upsert_environment(
                conn, environment, target_model, target_revision
            )
            for result in rows:
                candidate_id = result.get("id") or (result.get("spec") or {}).get("id")
                if not candidate_id:
                    continue
                resolved = resolved_candidates.get(str(candidate_id))
                quality = result.get("quality") or {}
                performance = result.get("performance") or {}
                evaluation = evaluations.get(candidate_id) or {}
                quality_pass = bool(
                    evaluation.get("quality_pass", quality.get("quality_pass", False))
                )
                qualified = bool(
                    evaluation.get(
                        "qualified",
                        result.get("ok") and quality_pass and not result.get("error"),
                    )
                )
                accepted = bool(
                    evaluation.get("accepted")
                    or (
                        verdict.get("accepted")
                        and verdict.get("selected_id") == candidate_id
                    )
                )
                failure_reason = result.get("error")
                disqualifications = evaluation.get("disqualifications") or []
                if not failure_reason and disqualifications:
                    failure_reason = "; ".join(map(str, disqualifications))
                failure_stage = None
                if not result.get("ok", False):
                    failure_stage = "candidate_execution"
                elif not quality_pass:
                    failure_stage = "quality_gate"
                elif not qualified:
                    failure_stage = "qualification"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO candidate_runs(
                        run_id, candidate_id, evolution_id, environment_fingerprint,
                        target_model, target_revision, manifest_id, manifest_hash,
                        source_revision, candidate_params_json, benchmark_suite,
                        quality_pass, quality_json, performance_json,
                        speedup_percent, accepted, qualified, failure_stage,
                        failure_reason, runtime_version, artifact_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(candidate_id),
                        evolution_id,
                        fingerprint,
                        target_model,
                        target_revision,
                        resolved.manifest_id if resolved else None,
                        resolved.manifest_hash if resolved else None,
                        resolved.source_revision if resolved else None,
                        _canonical(result.get("spec") or {}),
                        resolved.benchmark_suite if resolved else None,
                        int(quality_pass),
                        _canonical(quality),
                        _canonical(performance),
                        evaluation.get("speedup_percent")
                        if evaluation.get("speedup_percent") is not None
                        else (
                            verdict.get("speedup_percent")
                            if verdict.get("selected_id") == candidate_id
                            else 0.0
                        ),
                        int(accepted),
                        int(qualified),
                        failure_stage,
                        failure_reason,
                        (environment.get("packages") or {}).get("mlx-vlm"),
                        summary.get("path"),
                        _now(),
                    ),
                )
                written += 1

            transition = summary.get("service_transition") or {}
            promotion_written = bool(
                verdict.get("accepted")
                and verdict.get("selected_id") != "baseline"
                and transition.get("status") == "serving_selected"
            )
            if promotion_written:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO promotions(
                        run_id, evolution_id, environment_fingerprint,
                        previous_recipe_json, selected_recipe_json, promoted_at,
                        online_verified, rolled_back, rollback_reason, active_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)
                    """,
                    (
                        run_id,
                        evolution_id,
                        fingerprint,
                        _canonical(previous_recipe or {}),
                        _canonical(selected_recipe or {}),
                        _now(),
                        int(
                            bool(
                                (transition.get("post_restart_quality") or {}).get(
                                    "quality_pass"
                                )
                            )
                            or transition.get("status") == "serving_selected"
                        ),
                    ),
                )
        return {
            "environment_fingerprint": fingerprint,
            "candidate_runs_written": written,
            "promotion_written": promotion_written,
            "run_id": run_id,
        }

    def record_evolution(
        self,
        evolution_id: str,
        status: str,
        trigger: dict[str, Any],
        artifact_path: str,
        supervisor_run_id: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_runs(
                    evolution_id, status, trigger_json, artifact_path,
                    supervisor_run_id, error, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evolution_id) DO UPDATE SET
                    status=excluded.status,
                    supervisor_run_id=excluded.supervisor_run_id,
                    error=excluded.error,
                    completed_at=excluded.completed_at
                """,
                (
                    evolution_id,
                    status,
                    _canonical(trigger),
                    artifact_path,
                    supervisor_run_id,
                    error,
                    started_at or now,
                    now,
                ),
            )

    def query_prior(
        self,
        environment: dict[str, Any],
        target_model: str,
        target_revision: str | None,
        resolved_candidates: dict[str, ResolvedCandidate],
    ) -> dict[str, Any]:
        fingerprint = environment_fingerprint(
            environment, target_model, target_revision
        )
        ranking: list[dict[str, Any]] = []
        with self._connect() as conn:
            for candidate in resolved_candidates.values():
                if candidate.id == "baseline":
                    continue
                row = conn.execute(
                    """
                    SELECT cr.*, e.chip, e.gpu_cores, e.unified_memory_gb
                    FROM candidate_runs cr
                    JOIN environments e ON e.fingerprint=cr.environment_fingerprint
                    WHERE cr.candidate_id=? AND cr.manifest_hash=?
                      AND cr.environment_fingerprint=?
                    ORDER BY cr.created_at DESC LIMIT 1
                    """,
                    (candidate.id, candidate.manifest_hash, fingerprint),
                ).fetchone()
                match = "exact"
                if row is None:
                    row = conn.execute(
                        """
                        SELECT cr.*, e.chip, e.gpu_cores, e.unified_memory_gb
                        FROM candidate_runs cr
                        JOIN environments e ON e.fingerprint=cr.environment_fingerprint
                        WHERE cr.candidate_id=? AND cr.manifest_hash=?
                          AND cr.target_model=?
                        ORDER BY cr.created_at DESC LIMIT 1
                        """,
                        (candidate.id, candidate.manifest_hash, target_model),
                    ).fetchone()
                    match = "similar" if row is not None else "none"
                if row is None:
                    ranking.append(
                        {
                            "candidate_id": candidate.id,
                            "manifest_id": candidate.manifest_id,
                            "manifest_hash": candidate.manifest_hash,
                            "match": "none",
                            "recommendation": "experiment",
                            "score": 0,
                        }
                    )
                    continue
                quality_pass = bool(row["quality_pass"])
                accepted = bool(row["accepted"])
                if match == "exact" and not quality_pass:
                    recommendation = "skip_same_version_quality_failure"
                    score = -100
                elif match == "exact" and accepted:
                    recommendation = "prioritize_revalidation"
                    score = 100
                elif match == "exact":
                    recommendation = "revalidate_exact_environment"
                    score = 50
                else:
                    recommendation = "use_as_prior_and_revalidate"
                    score = 25 if accepted else 5
                ranking.append(
                    {
                        "candidate_id": candidate.id,
                        "manifest_id": candidate.manifest_id,
                        "manifest_hash": candidate.manifest_hash,
                        "match": match,
                        "recommendation": recommendation,
                        "score": score,
                        "run_id": row["run_id"],
                        "quality_pass": quality_pass,
                        "accepted": accepted,
                        "qualified": bool(row["qualified"]),
                        "speedup_percent": row["speedup_percent"],
                        "failure_stage": row["failure_stage"],
                        "failure_reason": row["failure_reason"],
                        "runtime_version": row["runtime_version"],
                    }
                )
        ranking.sort(
            key=lambda item: (
                -item["score"],
                -(item.get("speedup_percent") or 0.0),
                item["candidate_id"],
            )
        )
        return {
            "environment_fingerprint": fingerprint,
            "target_model": target_model,
            "target_revision": target_revision,
            "ranking": ranking,
            "requires_local_validation": True,
        }

    def mark_promotion_rolled_back(
        self,
        run_id: str,
        reason: str,
        active_seconds: float | None = None,
    ) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE promotions
                SET rolled_back=1, rollback_reason=?, active_seconds=?
                WHERE run_id=?
                """,
                (reason, active_seconds, run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"promotion not found for run: {run_id}")

    def latest_manifest_outcome(
        self,
        manifest_hash: str,
        environment_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Summarize exact-environment evidence for pre-P4 dedup bootstrap."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    MAX(accepted) AS any_accepted,
                    MIN(quality_pass) AS all_quality_pass,
                    MAX(quality_pass) AS any_quality_pass,
                    MAX(created_at) AS last_seen_at,
                    COUNT(*) AS result_count
                FROM candidate_runs
                WHERE manifest_hash=? AND environment_fingerprint=?
                """,
                (manifest_hash, environment_fingerprint),
            ).fetchone()
        if row is None or int(row["result_count"] or 0) == 0:
            return None
        return {
            "accepted": bool(row["any_accepted"]),
            "all_quality_pass": bool(row["all_quality_pass"]),
            "any_quality_pass": bool(row["any_quality_pass"]),
            "last_seen_at": row["last_seen_at"],
            "result_count": int(row["result_count"]),
        }

    def get_watch_state(self, context_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM watch_states WHERE context_key=?", (context_key,)
            ).fetchone()
        return dict(row) if row is not None else None

    def record_watch_state(
        self,
        *,
        context_key: str,
        service_name: str,
        manifest_id: str,
        manifest_hash: str,
        environment_fingerprint: str,
        policy_hash: str,
        snapshot_hash: str | None,
        status: str,
        attempt_count: int,
        now: str,
        next_attempt_at: str | None = None,
        last_evolution_id: str | None = None,
        error_class: str | None = None,
        last_error: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO watch_states(
                    context_key, service_name, manifest_id, manifest_hash,
                    environment_fingerprint, policy_hash, snapshot_hash,
                    status, attempt_count, next_attempt_at, last_evolution_id,
                    error_class, last_error, first_seen_at, last_attempt_at,
                    completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(context_key) DO UPDATE SET
                    snapshot_hash=excluded.snapshot_hash,
                    status=excluded.status,
                    attempt_count=excluded.attempt_count,
                    next_attempt_at=excluded.next_attempt_at,
                    last_evolution_id=excluded.last_evolution_id,
                    error_class=excluded.error_class,
                    last_error=excluded.last_error,
                    last_attempt_at=excluded.last_attempt_at,
                    completed_at=excluded.completed_at,
                    updated_at=excluded.updated_at
                """,
                (
                    context_key,
                    service_name,
                    manifest_id,
                    manifest_hash,
                    environment_fingerprint,
                    policy_hash,
                    snapshot_hash,
                    status,
                    max(0, int(attempt_count)),
                    next_attempt_at,
                    last_evolution_id,
                    error_class,
                    last_error,
                    now,
                    now,
                    completed_at,
                    now,
                ),
            )

    def list_watch_states(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM watch_states ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_notification(
        self,
        template_id: str,
        fields_hash: str,
        sent: bool,
        detail: str | None = None,
        created_at: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notification_events(
                    template_id, fields_hash, sent, detail, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    template_id,
                    fields_hash,
                    int(bool(sent)),
                    detail,
                    created_at or _now(),
                ),
            )

    def _count(self, table: str) -> int:
        if table not in {
            "environments",
            "candidate_runs",
            "promotions",
            "evolution_runs",
            "watch_states",
            "notification_events",
        }:
            raise ValueError("invalid memory table")
        with self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def show(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "environments": self._count("environments"),
            "candidate_runs": self._count("candidate_runs"),
            "promotions": self._count("promotions"),
            "evolution_runs": self._count("evolution_runs"),
            "watch_states": self._count("watch_states"),
            "notification_events": self._count("notification_events"),
            "recent_evolutions": self.list_evolutions(limit=5),
            "recent_promotions": self.list_promotions(limit=5),
            "recent_watch_states": self.list_watch_states(limit=5),
        }

    def list_evolutions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_runs ORDER BY completed_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [
            {
                **dict(row),
                "trigger": _loads(row["trigger_json"]),
            }
            for row in rows
        ]

    def list_promotions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM promotions ORDER BY promoted_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [
            {
                **dict(row),
                "online_verified": bool(row["online_verified"]),
                "rolled_back": bool(row["rolled_back"]),
                "previous_recipe": _loads(row["previous_recipe_json"]),
                "selected_recipe": _loads(row["selected_recipe_json"]),
            }
            for row in rows
        ]
