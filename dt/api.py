#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/api.py — Flask API for the Fabric Digital Twin

Endpoints
---------
GET  /health
GET  /snapshot
POST /observe            { payload: {type: "node"|"link", ...} }
POST /plan               { job: {...}, dry_run?: bool, strategy?: "greedy"|"cheapest-energy" }
POST /plan_batch         { jobs: [ {...}, ... ], dry_run?: bool, strategy?: ... }
POST /release            { releases: [ {node: "...", reservation_id: "..."} ] }

Run
---
export FLASK_APP=dt.api:app
flask run -h 0.0.0.0 -p 8080

or:

python3 -m dt.api --host 0.0.0.0 --port 8080
"""

from __future__ import annotations
import argparse
import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

from .state import DTState, safe_float
from .cost_model import CostModel, merge_stage_details

# -----------------------------------
# App singletons
# -----------------------------------

STATE = DTState()  # loads nodes/, topology, starts watcher
CM = CostModel(STATE)

RECENT_PLANS: Deque[Dict[str, Any]] = deque(maxlen=200)

app = Flask(__name__)


# -----------------------------------
# Helpers
# -----------------------------------


def _ok(data: Any, status: int = 200):
    return jsonify({"ok": True, "data": data}), status


def _err(msg: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": msg, **extra}), status


def _free_caps(node: Dict[str, Any]) -> Dict[str, float]:
    return STATE._effective_caps(node)


def _supports_formats(node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
    allowed = set(stage.get("allowed_formats") or [])
    disallowed = set(stage.get("disallowed_formats") or [])
    fmts = set(node.get("formats_supported") or [])
    if disallowed & fmts:
        return False
    if not allowed:
        return True
    return bool(fmts & allowed)


def _fits(node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
    if (node.get("dyn") or {}).get("down", False):
        return False
    caps = _free_caps(node)
    res = stage.get("resources") or {}
    need_cpu = safe_float(res.get("cpu_cores"), 0.0)
    need_mem = safe_float(res.get("mem_gb"), 0.0)
    need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)
    if caps["free_cpu_cores"] + 1e-9 < need_cpu:
        return False
    if caps["free_mem_gb"] + 1e-9 < need_mem:
        return False
    if caps["free_gpu_vram_gb"] + 1e-9 < need_vram:
        return False
    return _supports_formats(node, stage)


def _choose_node_for_stage(
    stage: Dict[str, Any], prev_node: Optional[str], strategy: str = "greedy"
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Return (node_name, metrics). If infeasible, returns (None, {...}).
    """
    best_name = None
    best_score = float("inf")
    best_metrics: Dict[str, Any] = {}

    for name, node in STATE.nodes_by_name.items():
        if not _fits(node, stage):
            continue

        # Times & risk
        comp_ms = CM.compute_time_ms(stage, node)
        xfer_ms = (
            0.0
            if prev_node in (None, name)
            else CM.transfer_time_ms(
                prev_node, name, safe_float(stage.get("size_mb"), 10.0)
            )
        )
        risk = CM.risk_score(stage, node)

        if strategy == "cheapest-energy":
            # Prefer nodes that minimize energy; tie-break with latency
            energy = CM.energy_kj(stage, node, comp_ms)
            score = energy * 10.0 + comp_ms * 0.01 + xfer_ms * 0.01 + risk * 0.1
        else:
            # Greedy: latency first, add small risk tax
            score = comp_ms + xfer_ms + risk * 10.0

        if score < best_score:
            best_score = score
            best_name = name
            best_metrics = {
                "compute_ms": round(comp_ms, 3),
                "xfer_ms": round(xfer_ms, 3),
                "risk": round(risk, 4),
                "score": round(score, 3),
            }

    if best_name is None:
        return None, {"reason": "no_feasible_node"}
    return best_name, best_metrics


def _reserve_stage(node_name: str, stage: Dict[str, Any]) -> Optional[str]:
    res = stage.get("resources") or {}
    req = {
        "node": node_name,
        "cpu_cores": safe_float(res.get("cpu_cores"), 0.0),
        "mem_gb": safe_float(res.get("mem_gb"), 0.0),
        "gpu_vram_gb": safe_float(res.get("gpu_vram_gb"), 0.0),
    }
    return STATE.reserve(req)


# -----------------------------------
# Routes
# -----------------------------------


@app.get("/health")
def health():
    return _ok({"ts": STATE.snapshot()["ts"]})


@app.get("/snapshot")
def snapshot():
    return _ok(STATE.snapshot())


@app.post("/observe")
def observe():
    if not request.is_json:
        return _err("expected JSON body")
    try:
        payload = request.get_json()
        STATE.apply_observation(payload)
        return _ok({"applied": True})
    except Exception as e:
        return _err(f"observe failed: {e}")


@app.post("/plan")
def plan():
    """
    Plan a single job.
    Body:
    {
      "job": { id, deadline_ms?, stages:[ {id, size_mb?, resources{cpu_cores,mem_gb,gpu_vram_gb}, allowed_formats?, ...}, ...] },
      "strategy": "greedy"|"cheapest-energy",
      "dry_run": false
    }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    job = body.get("job")
    if not job:
        return _err("missing 'job'")

    strategy = (body.get("strategy") or "greedy").lower().strip()
    dry_run = bool(body.get("dry_run", False))

    stages: List[Dict[str, Any]] = job.get("stages") or []
    if not stages:
        return _err("job.stages is empty")

    assignments: Dict[str, str] = {}
    reservations: List[Dict[str, str]] = []
    per_stage: List[Dict[str, Any]] = []
    prev_node: Optional[str] = None

    infeasible = False

    for st in stages:
        sid = st.get("id")
        if not sid:
            return _err("each stage must have an 'id'")
        chosen, metrics = _choose_node_for_stage(st, prev_node, strategy=strategy)
        if chosen is None:
            per_stage.append({"id": sid, "node": None, "infeasible": True, **metrics})
            infeasible = True
            prev_node = None
            continue

        # Optionally reserve
        res_id = None
        if not dry_run:
            res_id = _reserve_stage(chosen, st)
            if res_id is None:
                # race or capacity changed; mark infeasible
                per_stage.append(
                    {
                        "id": sid,
                        "node": chosen,
                        "infeasible": True,
                        "reason": "reservation_failed",
                    }
                )
                infeasible = True
                prev_node = None
                continue
            reservations.append({"node": chosen, "reservation_id": res_id})

        per_stage.append(
            {"id": sid, "node": chosen, "reservation_id": res_id, **metrics}
        )
        assignments[sid] = chosen
        prev_node = chosen

    # Compute end-to-end cost with current assignments
    cost = CM.job_cost(job, assignments)
    cost_per_stage = cost.get("per_stage") or []
    merged_per_stage = merge_stage_details(per_stage, cost_per_stage)
    ddl = safe_float(job.get("deadline_ms"), 0.0)
    penalty = CM.slo_penalty(ddl, cost["latency_ms"]) if ddl > 0 else 0.0

    resp = {
        "job_id": job.get("id"),
        "assignments": assignments,
        "reservations": reservations,
        "per_stage": merged_per_stage,
        "latency_ms": cost["latency_ms"],
        "energy_kj": cost["energy_kj"],
        "risk": cost["risk"],
        "deadline_ms": ddl or None,
        "slo_penalty": penalty,
        "infeasible": infeasible or (cost["latency_ms"] == float("inf")),
        "strategy": strategy,
        "dry_run": dry_run,
        "ts": int(time.time() * 1000),
    }
    RECENT_PLANS.appendleft(resp)
    return _ok(resp)


@app.get("/plans")
def plans():
    return _ok(list(RECENT_PLANS))


@app.post("/plan_batch")
def plan_batch():
    """
    Plan multiple jobs in one call.
    Body:
    {
      "jobs": [ { job }, ... ],
      "strategy": "...",
      "dry_run": false
    }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    jobs = body.get("jobs") or []
    if not jobs:
        return _err("missing 'jobs'")

    strategy = (body.get("strategy") or "greedy").lower().strip()
    dry_run = bool(body.get("dry_run", False))

    results = []
    for j in jobs:
        # Reuse the logic by faking a request-local plan
        tmp_req = {"job": j, "strategy": strategy, "dry_run": dry_run}
        with app.test_request_context(json=tmp_req):
            plan_result = app.view_functions["plan"]()  # type: ignore
            if isinstance(plan_result, tuple):
                resp_obj, status = plan_result
            else:
                resp_obj = plan_result
                status = getattr(resp_obj, "status_code", 200)

            data = None
            if hasattr(resp_obj, "get_json"):
                data = resp_obj.get_json(silent=True)
            if data is None:
                try:
                    data = json.loads(resp_obj.get_data(as_text=True) or "{}")
                except Exception:
                    data = {}

            if status >= 400 or not data.get("ok", False):
                return resp_obj, status

            payload = data.get("data")
            if payload is None:
                return _err("plan returned no data", status=500)

            results.append(payload)
    return _ok({"results": results})


@app.post("/release")
def release():
    """
    Release reservations.
    Body: { releases: [ { node: "ws-001", reservation_id: "res-0000001" }, ... ] }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    rels = body.get("releases") or []
    done = []
    for r in rels:
        node = r.get("node")
        rid = r.get("reservation_id")
        if node and rid:
            ok = STATE.release(node, rid)
            done.append({"node": node, "reservation_id": rid, "released": bool(ok)})
    return _ok({"released": done})


# -----------------------------------
# CLI entrypoint
# -----------------------------------


def main():
    ap = argparse.ArgumentParser(description="Fabric DT API")
    ap.add_argument("--host", default=os.environ.get("FABRIC_API_HOST", "127.0.0.1"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("FABRIC_API_PORT", "8080"))
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
