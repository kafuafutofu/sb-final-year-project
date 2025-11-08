"""Federated and fault-tolerant planner for the Fabric DT.

The :class:`FederatedPlanner` extends the baseline greedy policy with
network-aware scoring, federation-aware load balancing, and explicit
fallback placements so that the DT can survive correlated failures and
link partitions.

Key features
------------
* Penalises federations that are already saturated or degraded so new
  work is steered towards healthier domains.
* Accounts for link loss/latency when chaining stages, preferring nodes
  connected through resilient paths.
* Emits fallback assignments per-stage so operators can pre-warm or
  quickly fail over when chaos events hit a zone.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from dt.cost_model import CostModel, clamp, merge_stage_details
from dt.state import DTState, safe_float

ModeConfig = Dict[str, Any]

DEFAULT_MODES: Dict[str, ModeConfig] = {
    "resilient": {
        "redundancy": 2,
        "risk_weight": 220.0,
        "load_weight": 380.0,
        "spread_weight": 210.0,
        "network_weight": 240.0,
        "resilience_weight": 250.0,
        "prefer_prev_bonus": 15.0,
    },
    "network-aware": {
        "redundancy": 1,
        "risk_weight": 200.0,
        "load_weight": 260.0,
        "spread_weight": 140.0,
        "network_weight": 300.0,
        "resilience_weight": 190.0,
        "prefer_prev_bonus": 12.0,
    },
    "federated": {
        "redundancy": 3,
        "risk_weight": 210.0,
        "load_weight": 360.0,
        "spread_weight": 260.0,
        "network_weight": 230.0,
        "resilience_weight": 240.0,
        "prefer_prev_bonus": 10.0,
    },
}


def _mode_key(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode in DEFAULT_MODES:
        return mode
    if mode in ("fault-tolerant", "ft", "failover"):
        return "resilient"
    if mode in ("balanced", "load-balance", "load-balanced"):
        return "network-aware"
    return "resilient"


class FederatedPlanner:
    def __init__(self, state: DTState, cost_model: CostModel):
        self.state = state
        self.cm = cost_model

    # --------------------- helpers ---------------------

    def _supports_formats(self, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        allowed = stage.get("allowed_formats") or []
        if not allowed:
            return True
        fmts = set(node.get("formats_supported") or [])
        return any(fmt in fmts for fmt in allowed)

    def _fits(self, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        if (node.get("dyn") or {}).get("down", False):
            return False
        eff = node.get("effective") or {}
        res = stage.get("resources") or {}
        need_cpu = safe_float(res.get("cpu_cores"), 0.0)
        need_mem = safe_float(res.get("mem_gb"), 0.0)
        need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)
        if eff.get("free_cpu_cores", 0.0) + 1e-9 < need_cpu:
            return False
        if eff.get("free_mem_gb", 0.0) + 1e-9 < need_mem:
            return False
        if eff.get("free_gpu_vram_gb", 0.0) + 1e-9 < need_vram:
            return False
        return self._supports_formats(node, stage)

    def _choose_format(self, stage: Dict[str, Any], node: Dict[str, Any]) -> Optional[str]:
        allowed = stage.get("allowed_formats") or []
        if not allowed:
            return None
        fmts = node.get("formats_supported") or []
        for fmt in allowed:
            if fmt in fmts:
                return fmt
        return allowed[0] if allowed else None

    def _projected_load(
        self,
        entry: Dict[str, Any],
        need_cpu: float,
        need_mem: float,
        need_vram: float,
    ) -> float:
        loads: List[float] = []
        total_cpu = safe_float(entry.get("total_cpu_cores"), 0.0)
        if total_cpu > 0:
            free_cpu = max(0.0, safe_float(entry.get("free_cpu_cores"), 0.0) - need_cpu)
            loads.append(clamp((total_cpu - free_cpu) / max(total_cpu, 1e-6), 0.0, 1.0))

        total_mem = safe_float(entry.get("total_mem_gb"), 0.0)
        if total_mem > 0:
            free_mem = max(0.0, safe_float(entry.get("free_mem_gb"), 0.0) - need_mem)
            loads.append(clamp((total_mem - free_mem) / max(total_mem, 1e-6), 0.0, 1.0))

        total_vram = safe_float(entry.get("total_gpu_vram_gb"), 0.0)
        if total_vram > 0:
            free_vram = max(0.0, safe_float(entry.get("free_gpu_vram_gb"), 0.0) - need_vram)
            loads.append(clamp((total_vram - free_vram) / max(total_vram, 1e-6), 0.0, 1.0))

        if not loads:
            return 0.0
        return sum(loads) / len(loads)

    def _consume_resources(
        self,
        node: Dict[str, Any],
        fed_entry: Dict[str, Any],
        need_cpu: float,
        need_mem: float,
        need_vram: float,
    ) -> None:
        eff = node.setdefault("effective", {})
        eff["free_cpu_cores"] = max(0.0, safe_float(eff.get("free_cpu_cores"), 0.0) - need_cpu)
        eff["free_mem_gb"] = max(0.0, safe_float(eff.get("free_mem_gb"), 0.0) - need_mem)
        eff["free_gpu_vram_gb"] = max(0.0, safe_float(eff.get("free_gpu_vram_gb"), 0.0) - need_vram)

        fed_entry["free_cpu_cores"] = max(
            0.0, safe_float(fed_entry.get("free_cpu_cores"), 0.0) - need_cpu
        )
        fed_entry["free_mem_gb"] = max(
            0.0, safe_float(fed_entry.get("free_mem_gb"), 0.0) - need_mem
        )
        fed_entry["free_gpu_vram_gb"] = max(
            0.0, safe_float(fed_entry.get("free_gpu_vram_gb"), 0.0) - need_vram
        )
        # Recompute load factor for next stages
        fed_entry["load_factor"] = self._projected_load(fed_entry, 0.0, 0.0, 0.0)

    def _score_candidate(
        self,
        stage: Dict[str, Any],
        node_name: str,
        node: Dict[str, Any],
        federation: str,
        fed_entry: Dict[str, Any],
        prev_node: Optional[str],
        used_federations: Counter,
        mode_cfg: ModeConfig,
    ) -> Tuple[float, Dict[str, Any]]:
        res = stage.get("resources") or {}
        need_cpu = safe_float(res.get("cpu_cores"), 0.0)
        need_mem = safe_float(res.get("mem_gb"), 0.0)
        need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)

        projected_load = self._projected_load(fed_entry, need_cpu, need_mem, need_vram)

        fmt_override = self._choose_format(stage, node)
        stage_eval = dict(stage)
        if fmt_override:
            stage_eval["allowed_formats"] = [fmt_override]

        comp_ms = self.cm.compute_time_ms(stage_eval, node)
        energy_kj = self.cm.energy_kj(stage_eval, node, comp_ms)

        if prev_node in (None, node_name):
            xfer_ms = 0.0
            link_metrics = {"loss_pct": 0.0, "down": False, "rtt_ms": 0.0}
        else:
            xfer_ms = self.cm.transfer_time_ms(
                prev_node, node_name, safe_float(stage.get("size_mb"), 10.0)
            )
            link_metrics = self.state.effective_link_between(prev_node, node_name)

        link_loss = safe_float(link_metrics.get("loss_pct"), 0.0)
        risk = self.cm.risk_score(stage_eval, node, link_loss_pct=link_loss)

        load_penalty = mode_cfg["load_weight"] * projected_load
        spread_penalty = mode_cfg["spread_weight"] * used_federations[federation]
        network_penalty = mode_cfg["network_weight"] * (
            (1.0 if link_metrics.get("down") else 0.0)
            + clamp(link_loss / 10.0, 0.0, 1.0)
        )
        resilience_penalty = mode_cfg["resilience_weight"] * (
            safe_float(fed_entry.get("down_fraction"), 0.0)
            + safe_float(fed_entry.get("hot_fraction"), 0.0)
        )
        risk_penalty = mode_cfg["risk_weight"] * risk

        score = (
            comp_ms
            + xfer_ms
            + load_penalty
            + spread_penalty
            + network_penalty
            + resilience_penalty
            + risk_penalty
        )

        if prev_node and prev_node == node_name:
            score -= mode_cfg["prefer_prev_bonus"]

        metrics = {
            "format": fmt_override,
            "compute_ms": round(comp_ms, 3),
            "xfer_ms": round(xfer_ms, 3),
            "energy_kj": round(energy_kj, 5),
            "risk": round(risk, 4),
            "score": round(score, 3),
            "load_penalty_ms": round(load_penalty, 3),
            "network_penalty_ms": round(network_penalty, 3),
            "resilience_penalty_ms": round(resilience_penalty, 3),
            "projected_load": round(projected_load, 4),
            "link_loss_pct": round(link_loss, 4),
        }

        return score, metrics

    # --------------------- public ---------------------

    def plan_job(
        self,
        job: Dict[str, Any],
        dry_run: bool = False,
        mode: str = "resilient",
    ) -> Dict[str, Any]:
        stages = job.get("stages") or []
        if not stages:
            return {
                "job_id": job.get("id"),
                "assignments": {},
                "per_stage": [],
                "reservations": [],
                "shadow_assignments": {},
                "latency_ms": 0.0,
                "energy_kj": 0.0,
                "risk": 0.0,
                "strategy": mode,
                "dry_run": dry_run,
                "infeasible": True,
                "reason": "no_stages",
                "ts": int(time.time() * 1000),
            }

        cfg = DEFAULT_MODES[_mode_key(mode)]

        nodes = self.state.nodes_for_planner()
        fed_overview = self.state.federations_overview()
        fed_stats_map = {
            entry["name"]: dict(entry)
            for entry in (fed_overview.get("federations") or [])
        }
        node_to_fed = fed_overview.get("node_federations") or {}

        assignments: Dict[str, str] = {}
        shadow_assignments: Dict[str, List[str]] = {}
        per_stage: List[Dict[str, Any]] = []
        reservations: List[Dict[str, str]] = []

        used_federations: Counter = Counter()
        prev_node: Optional[str] = None
        infeasible = False
        fallback_crossfed = 0

        for stage in stages:
            sid = stage.get("id")
            if not sid:
                continue

            res = stage.get("resources") or {}
            need_cpu = safe_float(res.get("cpu_cores"), 0.0)
            need_mem = safe_float(res.get("mem_gb"), 0.0)
            need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)

            candidates: List[Tuple[float, Dict[str, Any], str, Dict[str, Any]]] = []

            for node_name, node in nodes.items():
                if not self._fits(node, stage):
                    continue
                federation = node_to_fed.get(node_name) or self.state.federation_for_node(node_name) or "global"
                fed_entry = fed_stats_map.setdefault(
                    federation,
                    {
                        "name": federation,
                        "total_cpu_cores": safe_float((node.get("caps") or {}).get("max_cpu_cores"), 0.0),
                        "free_cpu_cores": safe_float((node.get("effective") or {}).get("free_cpu_cores"), 0.0),
                        "total_mem_gb": safe_float((node.get("caps") or {}).get("ram_gb"), 0.0),
                        "free_mem_gb": safe_float((node.get("effective") or {}).get("free_mem_gb"), 0.0),
                        "total_gpu_vram_gb": safe_float((node.get("caps") or {}).get("gpu_vram_gb"), 0.0),
                        "free_gpu_vram_gb": safe_float((node.get("effective") or {}).get("free_gpu_vram_gb"), 0.0),
                        "down_fraction": 0.0,
                        "hot_fraction": 0.0,
                        "load_factor": 0.0,
                    },
                )

                score, metrics = self._score_candidate(
                    stage,
                    node_name,
                    node,
                    federation,
                    fed_entry,
                    prev_node,
                    used_federations,
                    cfg,
                )
                candidates.append((score, metrics, node_name, fed_entry))

            if not candidates:
                infeasible = True
                per_stage.append(
                    {
                        "id": sid,
                        "node": None,
                        "infeasible": True,
                        "reason": "no_feasible_node",
                    }
                )
                prev_node = None
                continue

            candidates.sort(key=lambda item: item[0])
            best_score, best_metrics, best_name, best_fed_entry = candidates[0]
            best_node = nodes.get(best_name, {})
            best_fed_name = node_to_fed.get(best_name) or best_fed_entry.get("name") or "global"

            fallback_nodes: List[str] = []
            fallback_feds: List[str] = []
            redundancy = max(1, int(cfg["redundancy"]))
            target_fallbacks = max(0, redundancy - 1)

            if target_fallbacks > 0:
                for candidate in candidates[1:]:
                    cand_name = candidate[2]
                    cand_fed = node_to_fed.get(cand_name) or candidate[3].get("name") or "global"
                    fallback_nodes.append(cand_name)
                    fallback_feds.append(cand_fed)
                    if cand_fed != best_fed_name:
                        fallback_crossfed += 1
                    if len(fallback_nodes) >= target_fallbacks:
                        break

            shadow_assignments[sid] = fallback_nodes

            res_id = None
            assigned = True
            if not dry_run:
                req = {
                    "node": best_name,
                    "cpu_cores": need_cpu,
                    "mem_gb": need_mem,
                    "gpu_vram_gb": need_vram,
                }
                res_id = self.state.reserve(req)
                if res_id is None:
                    assigned = False
                    infeasible = True

            if assigned:
                assignments[sid] = best_name
                if res_id:
                    reservations.append({"node": best_name, "reservation_id": res_id})
                self._consume_resources(best_node, best_fed_entry, need_cpu, need_mem, need_vram)
                used_federations[best_fed_name] += 1
                prev_node = best_name
            else:
                prev_node = None

            per_stage.append(
                {
                    "id": sid,
                    "node": best_name if assigned else None,
                    "reservation_id": res_id,
                    "federation": best_fed_name,
                    "fallbacks": fallback_nodes,
                    "fallback_federations": fallback_feds,
                    "infeasible": not assigned,
                    **best_metrics,
                }
            )

        cost = self.cm.job_cost(job, assignments)
        merged = merge_stage_details(per_stage, cost.get("per_stage"))
        ddl = safe_float(job.get("deadline_ms"), 0.0)
        slo_penalty = self.cm.slo_penalty(ddl, cost.get("latency_ms", float("inf"))) if ddl > 0 else 0.0

        unique_feds = {
            node_to_fed.get(node) or self.state.federation_for_node(node) or "global"
            for node in assignments.values()
        }
        spread = len(unique_feds) / max(1, len(stages))
        fallback_ratio = sum(1 for v in shadow_assignments.values() if v) / max(1, len(stages))
        crossfed_ratio = fallback_crossfed / max(1, len(stages))

        projected_feds = []
        for entry in fed_stats_map.values():
            projected_feds.append(
                {
                    "name": entry.get("name"),
                    "free_cpu_cores": round(safe_float(entry.get("free_cpu_cores"), 0.0), 4),
                    "free_mem_gb": round(safe_float(entry.get("free_mem_gb"), 0.0), 4),
                    "free_gpu_vram_gb": round(safe_float(entry.get("free_gpu_vram_gb"), 0.0), 4),
                    "load_factor": round(safe_float(entry.get("load_factor"), 0.0), 4),
                }
            )
        projected_feds.sort(key=lambda x: x["name"] or "")

        result = {
            "job_id": job.get("id"),
            "assignments": assignments,
            "reservations": reservations,
            "shadow_assignments": shadow_assignments,
            "per_stage": merged,
            "latency_ms": cost.get("latency_ms"),
            "energy_kj": cost.get("energy_kj"),
            "risk": cost.get("risk"),
            "deadline_ms": ddl or None,
            "slo_penalty": slo_penalty,
            "infeasible": infeasible or (cost.get("latency_ms") == float("inf")),
            "strategy": mode,
            "dry_run": dry_run,
            "federation_spread": round(spread, 4),
            "federations_in_use": sorted(unique_feds),
            "resilience_score": round(fallback_ratio, 4),
            "cross_federation_fallback_ratio": round(crossfed_ratio, 4),
            "projected_federations": projected_feds,
            "ts": int(time.time() * 1000),
        }

        return result

