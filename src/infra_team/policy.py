"""Policy: validate untrusted plans against one resolved registry snapshot.

Candidate facts come from strict, project-maintained manifests. Executable
capabilities stay code-owned in LAUNCH_TEMPLATES. A model may select an ID from
the snapshot; it cannot introduce a repository, command or launch parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .runner import CandidateSpec

if TYPE_CHECKING:
    from .candidate_registry import ResolvedCandidate
    from .compatibility import AutonomyPolicy

TARGET_MODEL = "mlx-community/Qwen3.8-27B-4bit"
TARGET_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"

# Code-owned capabilities, not candidate facts. Manifests may reference one of
# these names, but cannot supply argv, modules or shell fragments.
LAUNCH_TEMPLATES: dict[str, dict[str, Any]] = {
    "mlx_vlm_baseline": {
        "draft_kind": None,
        "allow_block_size": False,
    },
    "mlx_vlm_dflash2": {
        "draft_kind": "dflash",
        "allow_block_size": True,
    },
}

ALLOWED_BLOCK_SIZES = (2, 3, 4, 5, 6, 7, 8, 10, 12)
LIMITS = {
    "max_memory_gb": (8.0, 44.0),
    "minimum_speedup_percent": (1.0, 100.0),
    "repeats": (1, 10),
}


class PolicyError(Exception):
    """Raised when a plan cannot be safely executed."""


@dataclass
class PolicyDecision:
    """Audit record of what was allowed and what was refused."""

    approved: list[str]
    refused: list[dict[str, str]]
    notes: list[str]
    manifest_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "refused": self.refused,
            "notes": self.notes,
            "manifest_hashes": self.manifest_hashes,
        }


def _clamp(name: str, value: float) -> float:
    lo, hi = LIMITS[name]
    return max(lo, min(hi, value))


def builtin_resolved_candidates() -> dict[str, "ResolvedCandidate"]:
    """Load the installed, validated builtin registry for legacy optimize."""
    from .candidate_registry import CandidateRegistry

    return CandidateRegistry.builtin().resolve_all()


def candidate_summaries(
    resolved_candidates: dict[str, "ResolvedCandidate"] | None = None,
) -> list[dict[str, Any]]:
    resolved = (
        builtin_resolved_candidates()
        if resolved_candidates is None
        else resolved_candidates
    )
    return [
        {
            "id": candidate.id,
            "description": candidate.description,
            "manifest_id": candidate.manifest_id,
            "manifest_hash": candidate.manifest_hash,
            "source_repo_id": candidate.source_repo_id,
            "source_revision": candidate.source_revision,
            "prepared": bool(candidate.local_model_path) or not candidate.requires_prepare,
        }
        for candidate in resolved.values()
    ]


def validate_plan(
    plan: dict[str, Any],
    repeats: int = 3,
    resolved_candidates: dict[str, "ResolvedCandidate"] | None = None,
) -> tuple[list[CandidateSpec], PolicyDecision]:
    """Turn an untrusted plan into fixed CandidateSpecs.

    Candidate IDs must exist in the exact discovery snapshot supplied by the
    caller. Unknown IDs are refused. The baseline is always injected first.
    """
    if not isinstance(plan, dict):
        raise PolicyError("plan must be a JSON object")
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise PolicyError("plan.experiments must be a non-empty list")

    resolved = (
        builtin_resolved_candidates()
        if resolved_candidates is None
        else resolved_candidates
    )
    if "baseline" not in resolved:
        raise PolicyError("resolved candidate snapshot has no baseline")

    approved: list[str] = []
    refused: list[dict[str, str]] = []
    notes: list[str] = []
    for item in experiments:
        proposed_hash = None
        if isinstance(item, str):
            cand_id, reason = item, ""
        elif isinstance(item, dict):
            cand_id = item.get("id") or item.get("candidate") or ""
            reason = str(item.get("reason", ""))
            proposed_hash = item.get("manifest_hash")
        else:
            refused.append({"id": str(item), "why": "unrecognised entry type"})
            continue
        if cand_id not in resolved:
            refused.append(
                {
                    "id": str(cand_id),
                    "why": "not in candidate whitelist",
                    "reason_given": reason,
                }
            )
            continue
        if proposed_hash is not None and proposed_hash != resolved[cand_id].manifest_hash:
            refused.append(
                {
                    "id": str(cand_id),
                    "why": "manifest hash does not match discovery snapshot",
                    "reason_given": reason,
                }
            )
            continue
        if cand_id in approved:
            notes.append(f"duplicate candidate '{cand_id}' collapsed")
            continue
        approved.append(cand_id)

    if "baseline" not in approved:
        approved.insert(0, "baseline")
        notes.append("baseline injected: required as the comparison reference")
    else:
        approved.remove("baseline")
        approved.insert(0, "baseline")
    if len(approved) == 1:
        raise PolicyError("plan contained no executable candidate beyond the baseline")

    repeats = int(_clamp("repeats", repeats))
    specs: list[CandidateSpec] = []
    manifest_hashes: dict[str, str] = {}
    final_approved: list[str] = []
    for cand_id in approved:
        candidate = resolved[cand_id]
        capability = LAUNCH_TEMPLATES.get(candidate.launch_template)
        if capability is None:
            refused.append(
                {"id": cand_id, "why": "manifest launch template is not executable"}
            )
            continue
        if candidate.draft_kind != capability["draft_kind"]:
            refused.append(
                {"id": cand_id, "why": "manifest launch parameters do not match template"}
            )
            continue
        block = candidate.draft_block_size
        if block is not None and (
            not capability["allow_block_size"] or block not in ALLOWED_BLOCK_SIZES
        ):
            refused.append({"id": cand_id, "why": f"block size {block} out of range"})
            continue
        specs.append(candidate.to_spec(repeats))
        final_approved.append(cand_id)
        manifest_hashes[cand_id] = candidate.manifest_hash

    if not specs or specs[0].id != "baseline":
        raise PolicyError("baseline launch template was not executable")
    if len(specs) == 1:
        raise PolicyError("no executable candidate remained after policy validation")
    return specs, PolicyDecision(
        approved=final_approved,
        refused=refused,
        notes=notes,
        manifest_hashes=manifest_hashes,
    )


def sanitise_acceptance(plan: dict[str, Any]) -> dict[str, float]:
    """Clamp planner-supplied thresholds into a safe range."""
    raw = plan.get("acceptance_policy") or {}
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        if "max_memory_gb" in raw:
            try:
                out["max_memory_gb"] = _clamp(
                    "max_memory_gb", float(raw["max_memory_gb"])
                )
            except (TypeError, ValueError):
                pass
        if "minimum_speedup_percent" in raw:
            try:
                out["minimum_speedup_percent"] = _clamp(
                    "minimum_speedup_percent",
                    float(raw["minimum_speedup_percent"]),
                )
            except (TypeError, ValueError):
                pass
    return out


def resolve_acceptance_policy(
    plan: dict[str, Any],
    autonomy_policy: "AutonomyPolicy",
    resolved_candidates: dict[str, "ResolvedCandidate"],
    approved_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge planner, operator and Manifest gates without allowing downgrades."""
    planner = sanitise_acceptance(plan)
    candidate_manifests = [
        resolved_candidates[candidate_id]
        for candidate_id in approved_ids
        if candidate_id != "baseline" and candidate_id in resolved_candidates
    ]
    manifest_memory_limits = [
        candidate.max_peak_memory_gb for candidate in candidate_manifests
    ]
    manifest_speedup_limits = [
        candidate.minimum_speedup_percent for candidate in candidate_manifests
    ]
    manifest_error_limits = [
        candidate.manifest.max_error_rate
        for candidate in candidate_manifests
        if candidate.manifest is not None
    ]
    manifest_quality_required = any(
        candidate.manifest is not None
        and candidate.manifest.require_all_quality_gates
        for candidate in candidate_manifests
    )

    max_memory_candidates = [
        planner.get("max_memory_gb", 40.0),
        autonomy_policy.max_peak_memory_gb,
        *manifest_memory_limits,
    ]
    minimum_speedup_candidates = [
        planner.get("minimum_speedup_percent", 5.0),
        autonomy_policy.minimum_speedup_percent,
        *manifest_speedup_limits,
    ]
    max_error_candidates = [*manifest_error_limits]
    if autonomy_policy.require_zero_errors:
        max_error_candidates.append(0.0)

    effective = {
        "quality_must_pass": bool(
            autonomy_policy.require_all_quality_gates or manifest_quality_required
        ),
        "max_memory_gb": min(max_memory_candidates),
        "minimum_speedup_percent": max(minimum_speedup_candidates),
        "max_error_rate": min(max_error_candidates) if max_error_candidates else 0.0,
    }
    sources = {
        "planner": planner,
        "autonomy_policy": {
            "quality_must_pass": autonomy_policy.require_all_quality_gates,
            "max_memory_gb": autonomy_policy.max_peak_memory_gb,
            "minimum_speedup_percent": autonomy_policy.minimum_speedup_percent,
            "max_error_rate": 0.0 if autonomy_policy.require_zero_errors else None,
        },
        "manifests": {
            candidate.id: {
                "quality_must_pass": (
                    candidate.manifest.require_all_quality_gates
                    if candidate.manifest is not None
                    else None
                ),
                "max_memory_gb": candidate.max_peak_memory_gb,
                "minimum_speedup_percent": candidate.minimum_speedup_percent,
                "max_error_rate": (
                    candidate.manifest.max_error_rate
                    if candidate.manifest is not None
                    else None
                ),
            }
            for candidate in candidate_manifests
        },
        "rule": {
            "quality_must_pass": "logical OR",
            "max_memory_gb": "minimum",
            "minimum_speedup_percent": "maximum",
            "max_error_rate": "minimum",
        },
        "effective": effective,
    }
    return effective, sources


def default_plan() -> dict[str, Any]:
    """Safe fallback expressed only in builtin registry candidate IDs."""
    return {
        "hypothesis": (
            "DFlash2 block-diffusion drafting may reduce target decode steps "
            "on Apple Silicon; compare its pinned native configuration with "
            "a shorter block using the same serving path."
        ),
        "experiments": [
            {"id": "baseline", "reason": "reference measurement"},
            {
                "id": "dflash2_default",
                "reason": "pinned manifest native configuration",
            },
            {
                "id": "dflash2_block4",
                "reason": "shorter block may fit the GPU verify step better",
            },
        ],
        "acceptance_policy": {
            "quality_pass": True,
            "minimum_speedup_percent": 5.0,
            "max_memory_gb": 40.0,
        },
        "source": "default_rule_based_plan",
    }
