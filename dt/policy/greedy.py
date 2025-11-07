#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/policy/greedy.py — Baseline greedy planner for Fabric DT.

What it does
------------
- Plans a single job (sequential stages) by scanning all feasible nodes per stage
  and choosing the one with the lowest score.
- Score combines compute_time + transfer_time (+ optional risk/energy weights).
- Optional BanditPolicy: pick an execution format per (stage,node) before scoring.
- Optional reservation step via DTState.reserve(), with 'dry_run' toggle.

Key API
-------
planner = GreedyPlanner(state, cost_model, bandit=None, cfg=None)
result  = planner.plan_job(job, dry_run=False)

Result shape
------------
{
  "job_id": str,
  "assignments": {stage_id: node_name, ...},
  "per_stage": [
     {"id": stage_id, "node": node_name, "format": "native|cuda|wasm|...",
      "compute_ms": float, "xfer_ms": float, "energy_kj": float, "risk": float,
      "score": float, "reservation_id": "res-..." (if not dry_run), "infeasible": bool, "reason"?: str}
  ],
  "reservations": [{"node": "...", "reservation_id": "res-..."}],
  "latency_ms": float,
  "energy_kj": float,
  "risk": float,
  "infeasible": bool
}

Notes
-----
- This module is framework-agnostic; `dt/api.py` can import and use it directly.
- No external dependencies.

"""
from dt.policy.rl_stub import RLPolicy
RL = RLPolicy(persist_path="sim/rl_state.json")

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

try:
    from dt.state import DTState, safe_float
    from dt.cost_model import CostModel, merge_stage_details
except Exception:  # pragma: no cover
    DTState = object  # type: ignore
    CostModel = object  # type: ignore
    def safe_float(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d
    def merge_stage_details(primary, cost):  # type: ignore
        return (cost or []) or (primary or [])

# Optional bandit
try:
    from dt.policy.bandit import BanditPolicy
except Exception:  # pragma: no cover
    BanditPolicy = None  # type: ignore


DEFAULT_CFG = {
    # scoring = compute_ms + xfer_ms + risk_weight*risk + energy_weight*energy_kj
    "risk_weight": 10.0,          # converts 0..1 risk → "ms-like" penalty
    "energy_weight": 0.0,         # set >0 to trade some latency for energy
    "prefer_locality_bonus_ms": 0.0,  # subtract this if stage stays on prev node
    "require_format_match": False,    # if True, node must support stage.allowed_formats
}


def _supports_formats(node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
    allowed = set(stage.get("allowed_formats") or [])
    disallowed = set(stage.get("disallowed_formats") or [])
    fmts = set(node.get("formats_supported") or [])
    if disallowed & fmts:
        return False
    if not allowed:
        return True
    return bool(fmts & allowed)


def _fits(state: DTState, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
    if (node.get("dyn") or {}).get("down", False):
        return False
    caps = state._effective_caps(node)
    res = stage.get("resources") or {}
    need_cpu  = safe_float(res.get("cpu_cores"), 0.0)
    need_mem  = safe_float(res.get("mem_gb"), 0.0)
    need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)
    if caps["free_cpu_cores"] + 1e-9 < need_cpu:  return False
    if caps["free_mem_gb"]   + 1e-9 < need_mem:   return False
    if caps["free_gpu_vram_gb"] + 1e-9 < need_vram: return False
    return True


class GreedyPlanner:
    def __init__(
        self,
        state: DTState,
        cost_model: CostModel,
        bandit: Optional["BanditPolicy"] = None,
        cfg: Optional[Dict[str, float]] = None,
    ):
        self.state = state
        self.cm = cost_model
        self.bandit = bandit
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}

    # --------- core scoring ---------

    def _choose_format(self, stage: Dict[str, Any], node: Dict[str, Any]) -> Optional[str]:
        if self.bandit is None:
            # If formats are specified and node supports them, keep as-is; else let CM handle penalties.
            allowed = stage.get("allowed_formats")
            if allowed:
                fmts = set(node.get("formats_supported") or [])
                # prefer intersection if exists; else just pick first allowed to constrain evaluation
                inter = [f for f in allowed if f in fmts]
                if inter:
                    return inter[0]
            return None  # no override
        # Ask bandit for a single best format
        return self.bandit.choose_format(stage, node)

    def _score_candidate(
        self,
        stage: Dict[str, Any],
        node_name: str,
        prev_node: Optional[str],
        prefer_locality_bonus_ms: float,
        risk_weight: float,
        energy_weight: float,
        require_format_match: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        node = self.state.nodes_by_name[node_name]

        # (Optional) hard format feasibility
        if require_format_match and not _supports_formats(node, stage):
            return float("inf"), {"reason": "format_mismatch"}

        # Pick evaluation format (bandit or heuristic)
        fmt_override = self._choose_format(stage, node)
        stage_eval = dict(stage)
        if fmt_override is not None:
            stage_eval["allowed_formats"] = [fmt_override]

        # Times, energy, risk
        comp_ms = self.cm.compute_time_ms(stage_eval, node)
        xfer_ms = 0.0 if prev_node in (None, node_name) else self.cm.transfer_time_ms(
            prev_node, node_name, safe_float(stage.get("size_mb"), 10.0)
        )
        energy  = self.cm.energy_kj(stage_eval, node, comp_ms)
        risk    = self.cm.risk_score(stage_eval, node)

        # Greedy score
        score = comp_ms + xfer_ms + risk_weight * risk + energy_weight * energy

        # Locality preference (keep stages on same node if ties)
        if prev_node and prev_node == node_name and prefer_locality_bonus_ms > 0:
            score -= prefer_locality_bonus_ms

        metrics = {
            "format": fmt_override,
            "compute_ms": round(comp_ms, 3),
            "xfer_ms": round(xfer_ms, 3),
            "energy_kj": round(energy, 5),
            "risk": round(risk, 4),
            "score": round(score, 3),
        }
        return score, metrics

    # --------- public: plan a job ---------

    def plan_job(self, job: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        stages: List[Dict[str, Any]] = job.get("stages") or []
        if not stages:
            return {
                "job_id": job.get("id"),
                "assignments": {},
                "per_stage": [],
                "reservations": [],
                "latency_ms": 0.0,
                "energy_kj": 0.0,
                "risk": 0.0,
                "infeasible": True,
                "reason": "no_stages",
            }

        risk_w   = float(self.cfg["risk_weight"])
        energy_w = float(self.cfg["energy_weight"])
        loc_bonus = float(self.cfg["prefer_locality_bonus_ms"])
        require_fmt = bool(self.cfg["require_format_match"])

        assignments: Dict[str, str] = {}
        per_stage: List[Dict[str, Any]] = []
        reservations: List[Dict[str, str]] = []

        prev_node: Optional[str] = None
        infeasible = False

        for st in stages:
            sid = st.get("id")
            if not sid:
                per_stage.append({"infeasible": True, "reason": "missing_stage_id"})
                infeasible = True
                prev_node = None
                continue

            best_name = None
            best_score = float("inf")
            best_metrics: Dict[str, Any] = {}

            # Scan candidates
            for name, node in self.state.nodes_by_name.items():
                if not _fits(self.state, node, st):
                    continue
                sc, met = self._score_candidate(
                    st, name, prev_node,
                    prefer_locality_bonus_ms=loc_bonus,
                    risk_weight=risk_w,
                    energy_weight=energy_w,
                    require_format_match=require_fmt,
                )
                if sc < best_score:
                    best_score = sc
                    best_name = name
                    best_metrics = met

            if best_name is None or best_score == float("inf"):
                per_stage.append({"id": sid, "node": None, "infeasible": True, "reason": "no_feasible_node"})
                infeasible = True
                prev_node = None
                continue

            # Try reservation unless dry_run
            res_id = None
            if not dry_run:
                res = st.get("resources") or {}
                req = {
                    "node": best_name,
                    "cpu_cores": safe_float(res.get("cpu_cores"), 0.0),
                    "mem_gb": safe_float(res.get("mem_gb"), 0.0),
                    "gpu_vram_gb": safe_float(res.get("gpu_vram_gb"), 0.0),
                }
                res_id = self.state.reserve(req)
                if res_id is None:
                    per_stage.append({"id": sid, "node": best_name, "infeasible": True, "reason": "reservation_failed"})
                    infeasible = True
                    prev_node = None
                    continue
                reservations.append({"node": best_name, "reservation_id": res_id})

            rec = {"id": sid, "node": best_name, "reservation_id": res_id, **best_metrics}
            per_stage.append(rec)
            assignments[sid] = best_name
            prev_node = best_name

        # End-to-end cost using CM (adds up compute+xfer & aggregates)
        job_cost = self.cm.job_cost(job, assignments)
        merged_per_stage = merge_stage_details(per_stage, job_cost.get("per_stage") or [])
        out = {
            "job_id": job.get("id"),
            "assignments": assignments,
            "per_stage": merged_per_stage,
            "reservations": reservations,
            "latency_ms": job_cost.get("latency_ms", float("inf")),
            "energy_kj": job_cost.get("energy_kj", 0.0),
            "risk": job_cost.get("risk", 1.0),
            "infeasible": infeasible or (job_cost.get("latency_ms") == float("inf")),
        }
        return out

