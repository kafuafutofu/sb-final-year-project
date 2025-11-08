#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/dashboard.py — Fabric DT web dashboard (full, single file).

Features
--------
- Overview: totals, quick stats
- Nodes table: status, arch, formats, effective/free capacities, health flags
- Links table: speed/RTT/jitter/loss/ECN
- Recent plans: per-stage node, latency/energy/risk, strategy, dry_run
- Actions:
  • Submit job JSON for planning (dry run / with reservations)
  • Run demo job
  • Apply observation (node/link changes)
  • Refresh snapshot

Run
---
python3 -m ui.dashboard --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request, make_response

# --- DT imports ---
from dt.state import DTState, safe_float
from dt.cost_model import CostModel
from dt.policy.greedy import GreedyPlanner
from dt.policy.resilient import FederatedPlanner

try:
    from dt.policy.bandit import BanditPolicy
except Exception:
    BanditPolicy = None  # optional

# ----------------- App singletons -----------------

app = Flask(__name__)

REMOTE_BASE = os.environ.get("FABRIC_DT_REMOTE")
REMOTE_TIMEOUT = float(os.environ.get("FABRIC_DT_REMOTE_TIMEOUT", "10.0"))
REMOTE_LABEL = "local DT (embedded)"
SESSION = requests.Session()

STATE: Optional[DTState] = None
CM: Optional[CostModel] = None
BANDIT = None
PLANNER: Optional[GreedyPlanner] = None
FED_PLANNER: Optional[FederatedPlanner] = None

RECENT_PLANS: deque = deque(maxlen=50)


def _configure_runtime(remote: Optional[str], timeout: float) -> None:
    global STATE, CM, BANDIT, PLANNER, FED_PLANNER, REMOTE_BASE, REMOTE_TIMEOUT, REMOTE_LABEL

    REMOTE_BASE = remote.rstrip("/") if remote else None
    REMOTE_TIMEOUT = timeout
    REMOTE_LABEL = (
        f"remote API @ {REMOTE_BASE}" if REMOTE_BASE else "local DT (embedded)"
    )

    if REMOTE_BASE:
        STATE = None
        CM = None
        BANDIT = None
        PLANNER = None
        FED_PLANNER = None
    else:
        if STATE is None:
            STATE = DTState()
            CM = CostModel(STATE)
            BANDIT = (
                BanditPolicy(persist_path="sim/bandit_state.json")
                if BanditPolicy
                else None
            )
            PLANNER = GreedyPlanner(
                STATE,
                CM,
                bandit=BANDIT,
                cfg={
                    "risk_weight": 10.0,
                    "energy_weight": 0.0,
                    "prefer_locality_bonus_ms": 0.5,
                    "require_format_match": False,
                },
            )
            FED_PLANNER = FederatedPlanner(STATE, CM)


_configure_runtime(REMOTE_BASE, REMOTE_TIMEOUT)


# ----------------- Helpers -----------------


def _ok(data: Any, status: int = 200):
    return jsonify({"ok": True, "data": data}), status


def _err(msg: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": msg, **extra}), status


def _remote_url(path: str) -> str:
    if not REMOTE_BASE:
        raise RuntimeError("remote base not configured")
    if not path.startswith("/"):
        path = "/" + path
    return f"{REMOTE_BASE}{path}"


def _proxy_remote(method: str, path: str, payload: Optional[Dict[str, Any]] = None):
    if not REMOTE_BASE:
        raise RuntimeError("remote base not configured")
    try:
        kwargs: Dict[str, Any] = {"timeout": REMOTE_TIMEOUT}
        if payload is not None:
            kwargs["json"] = payload
        resp = SESSION.request(method.upper(), _remote_url(path), **kwargs)
    except requests.RequestException as exc:
        return _err(f"remote request failed: {exc}", status=502)

    try:
        data = resp.json()
    except ValueError:
        return _err(
            f"remote returned non-JSON response (status {resp.status_code})", status=502
        )

    return make_response(data, resp.status_code)


def _demo_job() -> Dict[str, Any]:
    return {
        "id": f"demo-{int(time.time())}",
        "deadline_ms": 5000,
        "stages": [
            {
                "id": "ingest",
                "type": "io",
                "size_mb": 40,
                "resources": {"cpu_cores": 1, "mem_gb": 1},
                "allowed_formats": ["native", "wasm"],
            },
            {
                "id": "prep",
                "type": "preproc",
                "size_mb": 60,
                "resources": {"cpu_cores": 2, "mem_gb": 2},
                "allowed_formats": ["native", "cuda", "wasm"],
            },
            {
                "id": "mlp",
                "type": "mlp",
                "size_mb": 100,
                "resources": {"cpu_cores": 4, "mem_gb": 4, "gpu_vram_gb": 2},
                "allowed_formats": ["cuda", "native"],
            },
        ],
    }


# ----------------- JSON APIs -----------------


@app.get("/api/health")
def api_health():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/health")
    return _ok({"ts": STATE.snapshot()["ts"]})


@app.get("/api/snapshot")
def api_snapshot():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/snapshot")
    return _ok(STATE.snapshot())


@app.get("/api/plans")
def api_plans():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/plans")
    return _ok(list(RECENT_PLANS))


@app.post("/api/plan")
def api_plan():
    """
    Body:
    {
      "job": { ... }   # job JSON
      "dry_run": true|false,
      "strategy": "greedy"  # (ignored here; GreedyPlanner used)
    }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    job = body.get("job")
    if not job:
        return _err("missing 'job'")
    dry = bool(body.get("dry_run", True))
    strategy = body.get("strategy") or "greedy"
    normalized = strategy.lower().strip()
    if REMOTE_BASE:
        payload = {
            "job": job,
            "dry_run": dry,
            "strategy": strategy,
        }
        return _proxy_remote("POST", "/plan", payload)

    if normalized in {"resilient", "network-aware", "federated", "balanced", "fault-tolerant"} and FED_PLANNER is not None:
        res = FED_PLANNER.plan_job(job, dry_run=dry, mode=normalized)
    else:
        res = PLANNER.plan_job(job, dry_run=dry)
    # SLO penalty if deadline
    ddl = safe_float(job.get("deadline_ms"), 0.0)
    if ddl > 0:
        res["slo_penalty"] = CM.slo_penalty(ddl, res.get("latency_ms", 0.0))
        res["deadline_ms"] = ddl
    res["strategy"] = strategy
    res["dry_run"] = dry
    res["ts"] = int(time.time() * 1000)
    RECENT_PLANS.appendleft(res)
    return _ok(res)


@app.post("/api/plan_demo")
def api_plan_demo():
    job = _demo_job()
    payload_in = request.get_json() if request.is_json else {}
    dry = bool((payload_in or {}).get("dry_run", True))
    strategy = (payload_in or {}).get("strategy") or "greedy"
    normalized = strategy.lower().strip()
    if REMOTE_BASE:
        payload = {"job": job, "dry_run": dry, "strategy": strategy}
        return _proxy_remote("POST", "/plan", payload)

    if normalized in {"resilient", "network-aware", "federated", "balanced", "fault-tolerant"} and FED_PLANNER is not None:
        res = FED_PLANNER.plan_job(job, dry_run=dry, mode=normalized)
    else:
        res = PLANNER.plan_job(job, dry_run=dry)
    ddl = safe_float(job.get("deadline_ms"), 0.0)
    if ddl > 0:
        res["slo_penalty"] = CM.slo_penalty(ddl, res.get("latency_ms", 0.0))
        res["deadline_ms"] = ddl
    res["strategy"] = strategy
    res["dry_run"] = dry
    res["ts"] = int(time.time() * 1000)
    RECENT_PLANS.appendleft(res)
    return _ok(res)


@app.post("/api/observe")
def api_observe():
    """
    Body: same shape as dt/state.apply_observation payload:
    { "payload": { "type":"node", "node":"name", "changes": { "down": true, "thermal_derate": 0.3 } } }
    or
    { "payload": { "type":"link", "key":"a|b", "changes": { "loss_pct": 2.0 } } }
    """
    if not request.is_json:
        return _err("expected JSON body")
    try:
        payload = request.get_json()
        if REMOTE_BASE:
            return _proxy_remote("POST", "/observe", payload)

        STATE.apply_observation(payload)
        # Optionally persist overrides so state watcher picks them up across restarts
        try:
            STATE.write_overrides()
        except Exception:
            pass
        return _ok({"applied": True})
    except Exception as e:
        return _err(f"observe failed: {e}")


# ----------------- HTML UI -----------------

_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Fabric DT Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:,">
<style>
:root {
  --bg: #0b0f14;
  --panel: #121822;
  --muted: #9bb0c9;
  --muted2: #6c7a8a;
  --text: #e7eef7;
  --accent: #6fc1ff;
  --good: #2ecc71;
  --warn: #f1c40f;
  --bad: #e74c3c;
  --chip: #1a2330;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
header {
  padding: 16px 20px; background: linear-gradient(180deg, #0f1722 0%, #0b0f14 100%);
  border-bottom: 1px solid #1c2430;
  position: sticky; top: 0; z-index: 10;
}
h1 { margin: 0; font-size: 20px; letter-spacing: 0.5px; }
small { color: var(--muted2); }
.container { padding: 16px 20px; display: grid; grid-template-columns: 1fr; gap: 16px; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 1080px) { .grid-2 { grid-template-columns: 1fr; } }

.card {
  background: var(--panel); border: 1px solid #1a2533; border-radius: 12px;
  padding: 14px; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
.card h2 { margin: 0 0 10px 0; font-size: 16px; color: #cfe7ff; letter-spacing: .3px; }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.kpi { background: var(--chip); padding: 10px 12px; border-radius: 10px; border: 1px solid #243140; }
.kpi .v { font-weight: 700; }
.btn {
  appearance: none; border: 1px solid #2a3a4f; background: #192434; color: var(--text);
  padding: 8px 12px; border-radius: 8px; cursor: pointer; font-weight: 600;
}
.btn:hover { border-color: #3f5876; }
.btn.primary { background: #13314d; border-color: #2c5b86; color: #cfe7ff; }
.btn.bad { background: #3a1010; border-color: #5b1a1a; color: #ffbbbb; }
.btn.good { background: #103a28; border-color: #1b5e40; color: #bbffde; }

table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #202b38; vertical-align: top; }
th { color: #a9c3e1; font-weight: 700; position: sticky; top: 60px; background: #111825; }
tr:hover { background: #0f1520; }
.tag { background: #1c2736; display: inline-block; padding: 3px 8px; border-radius: 10px; margin: 2px; font-size: 12px; color: #c7d8ec; border: 1px solid #2b3a4f; }
.badge { padding: 2px 6px; border-radius: 8px; font-weight: 700; font-size: 12px; }
.badge.good { background: #103a28; color: #8ff0c1; border: 1px solid #1e5d42; }
.badge.warn { background: #3a2f10; color: #f8e38a; border: 1px solid #6e5a1a; }
.badge.bad  { background: #3a1010; color: #ffb4b4; border: 1px solid #5e1b1b; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; color: #bcd0e6; }
textarea, input, select {
  width: 100%; background: #0e1420; color: var(--text); border: 1px solid #243140; border-radius: 8px;
  padding: 10px 12px; outline: none;
}
textarea:focus, input:focus { border-color: #3a5575; }
footer { color: var(--muted2); text-align: center; padding: 12px; }
hr { border: none; border-top: 1px solid #1f2a39; margin: 12px 0; }
.small { font-size: 12px; color: var(--muted2); }
.right { text-align: right; }
.flex { display: flex; gap: 8px; align-items: center; }
.remote-label { color: var(--muted); margin-top: 4px; }
.dag-wrap { margin-top: 12px; padding-top: 8px; border-top: 1px solid #1f2a39; }
.dag-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.stage-card { background: var(--chip); border: 1px solid #2a3648; border-radius: 10px; padding: 10px 12px; min-width: 120px; }
.stage-card.bad { background: #2a1414; border-color: #5e1b1b; }
.stage-card .node { font-weight: 700; font-size: 14px; }
.stage-card .fmt { color: #8fbef6; font-size: 12px; margin-top: 4px; display: block; }
.stage-card .metrics { font-size: 12px; color: var(--muted2); margin-top: 4px; }
.stage-arrow { font-size: 18px; color: var(--muted2); }
.topology-canvas { width: 100%; height: 420px; position: relative; }
.topology-canvas svg { width: 100%; height: 100%; }
.topology-legend { margin-top: 8px; font-size: 12px; color: var(--muted2); display: flex; gap: 12px; flex-wrap: wrap; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.legend-dot.up { background: #2ecc71; border: 1px solid #1e5d42; }
.legend-dot.derate { background: #f1c40f; border: 1px solid #6e5a1a; }
.legend-dot.down { background: #e74c3c; border: 1px solid #5e1b1b; }
.legend-dot.assignment { border: 2px solid var(--accent); border-radius: 50%; width: 12px; height: 12px; }
.legend-dot.fallback { border: 2px dashed #f39c12; background: transparent; }
.legend-dot.lossy { background: #f39c12; border: 1px solid #925208; }
.node-label { fill: #cfe7ff; font-size: 11px; pointer-events: none; text-anchor: middle; }
.link { stroke: #243140; stroke-width: 1.8px; stroke-linecap: round; opacity: 0.8; }
.node-core { stroke: #1a2533; stroke-width: 1.5px; }
.node-ring { fill: none; stroke-width: 3px; opacity: 0.8; }
.node-shadow { fill: none; stroke: #f39c12; stroke-width: 2px; opacity: 0.7; stroke-dasharray: 5 4; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
</head>
<body>
<header>
  <h1>Fabric Digital Twin — <small>Cluster Dashboard</small></h1>
  <div class="remote-label">Mode: __REMOTE_LABEL__</div>
</header>

<div class="container">

  <div class="card">
    <h2>Overview</h2>
    <div class="row">
      <div class="kpi">Nodes: <span id="k_nodes" class="v">—</span></div>
      <div class="kpi">Links: <span id="k_links" class="v">—</span></div>
      <div class="kpi">Federations: <span id="k_feds" class="v">—</span></div>
      <div class="kpi">Last Snapshot: <span id="k_ts" class="v">—</span></div>
      <div class="kpi">Max Fed Load: <span id="k_fedload" class="v">—</span></div>
      <div class="kpi">Plan Spread: <span id="k_spread" class="v">—</span></div>
      <div class="kpi">Fallback Coverage: <span id="k_resilience" class="v">—</span></div>
      <div class="kpi">Down: <span id="k_down" class="v">—</span></div>
      <div class="kpi">Reservations: <span id="k_resv" class="v">—</span></div>
      <div class="kpi">CPU used: <span id="k_cpu" class="v">—</span></div>
      <button class="btn" onclick="refresh()">Refresh</button>
      <button class="btn good" onclick="runDemo(true)">Dry-run Demo</button>
      <button class="btn primary" onclick="runDemo(false)">Reserve Demo</button>
    </div>
  </div>

  <div class="card">
    <h2>Fabric Topology</h2>
    <div class="topology-canvas" id="topology_canvas"><div class="small">Loading topology…</div></div>
    <div class="topology-legend">
      <span><span class="legend-dot up"></span> healthy</span>
      <span><span class="legend-dot derate"></span> thermal derate</span>
      <span><span class="legend-dot down"></span> down</span>
      <span><span class="legend-dot fallback"></span> fallback-ready</span>
      <span><span class="legend-dot assignment"></span> latest plan assignment</span>
      <span><span class="legend-dot lossy"></span> lossy / degraded link</span>
    </div>
  </div>

  <div class="grid-2">

    <div class="card">
      <h2>Nodes</h2>
      <div class="row" style="margin-bottom:8px;">
        <input id="nodeFilter" placeholder="Filter by name/arch/label..." oninput="renderNodes()" />
        <select id="archFilter" onchange="renderNodes()">
          <option value="">Arch: All</option>
          <option>x86_64</option>
          <option>arm64</option>
          <option>riscv64</option>
        </select>
      </div>
      <div style="max-height: 420px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Name / Class</th>
              <th>Arch & Formats</th>
              <th>CPU (free/max)</th>
              <th>Mem (free/max GB)</th>
              <th>VRAM (free/max GB)</th>
              <th>Health</th>
            </tr>
          </thead>
          <tbody id="nodes_tbody"></tbody>
        </table>
      </div>
      <div class="small">Tip: Health badges reflect <span class="mono">dyn.down</span>, <span class="mono">thermal_derate</span>, and recent reservations.</div>
    </div>

    <div class="card">
      <h2>Links</h2>
      <div style="max-height: 420px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th>Peers</th>
              <th>Speed (Gbps)</th>
              <th>RTT / Jitter (ms)</th>
              <th>Loss (%)</th>
              <th>ECN</th>
            </tr>
          </thead>
          <tbody id="links_tbody"></tbody>
        </table>
      </div>
    </div>

  </div>

  <div class="grid-2">

    <div class="card">
      <h2>Federations</h2>
      <div style="max-height: 320px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Load</th>
              <th>Down / Total</th>
              <th>Reservations</th>
              <th>Avg loss (%)</th>
              <th>Avg trust</th>
            </tr>
          </thead>
          <tbody id="feds_tbody"></tbody>
        </table>
      </div>
      <div class="small">Load blends CPU, memory, and accelerator utilisation across each federation.</div>
    </div>

    <div class="card">
      <h2>Federation Links</h2>
      <div style="max-height: 320px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>A</th>
              <th>B</th>
              <th>Links</th>
              <th>Down</th>
              <th>Min speed (Gbps)</th>
              <th>Avg RTT (ms)</th>
              <th>Max loss (%)</th>
            </tr>
          </thead>
          <tbody id="fedlinks_tbody"></tbody>
        </table>
      </div>
      <div class="small">Aggregated cross-federation health synthesised from topology links and overrides.</div>
    </div>

  </div>

  <div class="grid-2">

    <div class="card">
      <h2>Plan a Job</h2>
      <div class="row">
        <label class="flex"><input id="dryRun" type="checkbox" checked /> Dry run</label>
        <select id="strategySelect">
          <option value="greedy">Strategy: Greedy</option>
          <option value="cheapest-energy">Strategy: Cheapest energy</option>
          <option value="resilient">Strategy: Resilient</option>
          <option value="network-aware">Strategy: Network aware</option>
          <option value="federated">Strategy: Federated</option>
          <option value="balanced">Strategy: Balanced</option>
        </select>
        <button class="btn primary" onclick="submitPlan()">Plan</button>
      </div>
      <textarea id="jobJson" rows="14" placeholder='Paste your job JSON here...'></textarea>
      <div class="small">Your job should contain: <span class="mono">id</span>, optional <span class="mono">deadline_ms</span>, and <span class="mono">stages[]</span>.</div>
      <div class="small">Uncheck “Dry run” to reserve capacity. Active reservations glow with a cyan halo on the topology map. Resilient and federated strategies surface fallbacks and projected federation load in the plan table.</div>
    </div>

    <div class="card">
      <h2>Recent Plans</h2>
      <div style="max-height: 420px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Job</th>
          <th>Latency (ms)</th>
          <th>Energy (kJ)</th>
          <th>Risk</th>
          <th>Spread / Federations</th>
          <th>Fallback Coverage</th>
          <th>Cross-fed Fallback</th>
          <th>Infeasible</th>
          <th>Stages</th>
            </tr>
          </thead>
          <tbody id="plans_tbody"></tbody>
        </table>
      </div>
      <div class="dag-wrap" id="plan_graph"></div>
    </div>

  </div>

  <div class="card">
    <h2>Apply Observation (node/link)</h2>
    <div class="row" style="margin-bottom:8px;">
      <button class="btn" onclick="applyObs()">Apply</button>
    </div>
    <textarea id="obsJson" rows="8" placeholder='{
  "payload": { "type": "node", "node": "ws-001", "changes": {"down": true, "thermal_derate": 0.3} }
}'></textarea>
    <div class="small">This mirrors the shape used by <span class="mono">sim/chaos.py</span> and merges into runtime <span class="mono">dyn</span> fields.</div>
  </div>

  <footer>Fabric DT — live simulator UI</footer>
</div>

<script>
let SNAP = null;
let LAST_PLAN = null;
let TOPO_SIM = null;
let TOPO_RESIZE = null;

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts || {});
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || 'request failed');
  return j.data;
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return (Number(x) * 100).toFixed(0) + '%';
}
function fmt(x, d=2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return Number(x).toFixed(d);
}
function ts(ms) { const d = new Date(ms); return d.toLocaleString(); }

function badge(s, cls) { return `<span class="badge ${cls}">${s}</span>`; }
function tag(s) { return `<span class="tag">${s}</span>`; }

function renderOverview() {
  if (!SNAP) return;
  const nodes = SNAP.nodes || [];
  const links = SNAP.links || [];
  const federations = SNAP.federations || [];
  document.getElementById('k_nodes').textContent = nodes.length;
  document.getElementById('k_links').textContent = links.length;
  document.getElementById('k_ts').textContent = ts(SNAP.ts);
  const kFeds = document.getElementById('k_feds');
  if (kFeds) kFeds.textContent = federations.length;
  const down = nodes.filter(n => (n.dyn||{}).down).length;
  const reservations = nodes.reduce((acc, n) => {
    const dyn = n.dyn || {};
    const res = dyn.reservations ? Object.keys(dyn.reservations).length : 0;
    return acc + res;
  }, 0);
  let usedCpu = 0;
  let maxCpu = 0;
  nodes.forEach(n => {
    const eff = n.effective || {};
    const maxC = Number(eff.max_cpu_cores || 0);
    const free = Number(eff.free_cpu_cores || 0);
    if (maxC > 0) {
      usedCpu += Math.max(0, maxC - free);
      maxCpu += maxC;
    }
  });
  const maxFedLoad = federations.length ? Math.max(...federations.map(fed => Number(fed.load_factor || 0))) : null;
  const fedLoadEl = document.getElementById('k_fedload');
  if (fedLoadEl) fedLoadEl.textContent = maxFedLoad !== null ? fmtPct(maxFedLoad) : '—';
  const spread = LAST_PLAN && LAST_PLAN.federation_spread !== undefined && LAST_PLAN.federation_spread !== null ? LAST_PLAN.federation_spread : null;
  const spreadEl = document.getElementById('k_spread');
  if (spreadEl) spreadEl.textContent = spread !== null ? fmt(spread, 2) : '—';
  const resilience = LAST_PLAN && LAST_PLAN.resilience_score !== undefined && LAST_PLAN.resilience_score !== null ? LAST_PLAN.resilience_score : null;
  const resilienceEl = document.getElementById('k_resilience');
  if (resilienceEl) resilienceEl.textContent = resilience !== null ? fmtPct(resilience) : '—';
  document.getElementById('k_down').textContent = down;
  document.getElementById('k_resv').textContent = reservations;
  document.getElementById('k_cpu').textContent = maxCpu > 0 ? fmtPct(usedCpu / maxCpu) : '—';
}

function renderNodes() {
  if (!SNAP) return;
  const q = (document.getElementById('nodeFilter').value || '').toLowerCase();
  const arch = document.getElementById('archFilter').value || '';
  const tb = document.getElementById('nodes_tbody');
  tb.innerHTML = '';
  const fedMap = SNAP.node_federations || {};
  SNAP.nodes.forEach(n => {
    const eff = n.effective || {};
    const dyn = n.dyn || {};
    const labels = n.labels || {};
    const archMatch = (!arch || (n.arch||'')===arch);
    const fed = fedMap[n.name] || labels.federation || labels.zone || '';
    const hay = (n.name+' '+(n.class||'')+' '+(n.arch||'')+' '+fed+' '+JSON.stringify(labels)).toLowerCase();
    if (!archMatch) return;
    if (q && !hay.includes(q)) return;

    let health = '';
    if (dyn.down) health += badge('DOWN', 'bad') + ' ';
    if (dyn.thermal_derate && dyn.thermal_derate>0) health += badge('DERATE '+Math.round(dyn.thermal_derate*100)+'%', 'warn') + ' ';
    const resv = (dyn.reservations && Object.keys(dyn.reservations).length) ? Object.keys(dyn.reservations).length : 0;
    if (resv>0) health += badge(`${resv} resv`, 'good');

    const fmts = (n.formats_supported||[]).map(tag).join(' ');
    let labelPairs = Object.entries(labels);
    if (fed) {
      labelPairs = [["federation", fed], ...labelPairs];
    }
    const lbls = labelPairs.map(([k,v]) => tag(`${k}:${v}`)).join(' ');

    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td><div class="mono">${n.name}</div><div class="small">${n.class||''}</div></td>
        <td><div>${n.arch||'—'}</div><div>${fmts||''}</div></td>
        <td><div>${fmt(eff.free_cpu_cores,2)} / ${fmt(eff.max_cpu_cores,2)}</div></td>
        <td><div>${fmt(eff.free_mem_gb,2)} / ${fmt(eff.max_mem_gb,2)}</div></td>
        <td><div>${fmt(eff.free_gpu_vram_gb,2)} / ${fmt(eff.max_gpu_vram_gb,2)}</div></td>
        <td>${health||''}<div class="small">${lbls||''}</div></td>
      </tr>
    `);
  });
}

function renderLinks() {
  if (!SNAP) return;
  const tb = document.getElementById('links_tbody');
  tb.innerHTML = '';
  SNAP.links.forEach(l => {
    const e = l.effective || {};
    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td class="mono">${l.key}</td>
        <td>${l.a} ↔ ${l.b}</td>
        <td>${fmt(e.speed_gbps,2)}</td>
        <td>${fmt(e.rtt_ms,1)} / ${fmt(e.jitter_ms,1)}</td>
        <td>${fmt(e.loss_pct,2)}</td>
        <td>${e.ecn ? 'Yes' : 'No'}</td>
      </tr>
    `);
  });
}

function renderFederations() {
  const tb = document.getElementById('feds_tbody');
  if (!tb) return;
  tb.innerHTML = '';
  const feds = SNAP && SNAP.federations ? SNAP.federations : [];
  if (!feds.length) {
    tb.innerHTML = '<tr><td colspan="6" class="small">No federation data.</td></tr>';
    return;
  }
  feds.forEach(fed => {
    const nodes = (fed.nodes || []).join(', ');
    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td><div class="mono">${fed.name || '—'}</div><div class="small">${nodes || '—'}</div></td>
        <td>${fmtPct(fed.load_factor)}</td>
        <td>${fed.down_nodes || 0} / ${(fed.nodes || []).length}</td>
        <td>${fed.reservations || 0}</td>
        <td>${fed.avg_loss_pct !== null && fed.avg_loss_pct !== undefined ? fmt(fed.avg_loss_pct, 2) : '—'}</td>
        <td>${fed.avg_trust !== null && fed.avg_trust !== undefined ? fmt(fed.avg_trust, 2) : '—'}</td>
      </tr>
    `);
  });
}

function renderFederationLinks() {
  const tb = document.getElementById('fedlinks_tbody');
  if (!tb) return;
  tb.innerHTML = '';
  const edges = SNAP && SNAP.federation_links ? SNAP.federation_links : [];
  if (!edges.length) {
    tb.innerHTML = '<tr><td colspan="7" class="small">No federation links available.</td></tr>';
    return;
  }
  edges.forEach(e => {
    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td>${e.a || '—'}</td>
        <td>${e.b || '—'}</td>
        <td>${e.links || 0}</td>
        <td>${e.down_links || 0}</td>
        <td>${e.min_speed_gbps !== null && e.min_speed_gbps !== undefined ? fmt(Number(e.min_speed_gbps), 2) : '—'}</td>
        <td>${e.avg_rtt_ms !== null && e.avg_rtt_ms !== undefined ? fmt(Number(e.avg_rtt_ms), 2) : '—'}</td>
        <td>${e.max_loss_pct !== null && e.max_loss_pct !== undefined ? fmt(Number(e.max_loss_pct), 2) : '—'}</td>
      </tr>
    `);
  });
}

function destroyTopology() {
  if (TOPO_SIM) {
    TOPO_SIM.stop();
    TOPO_SIM = null;
  }
}

function renderTopology() {
  const wrap = document.getElementById('topology_canvas');
  if (!wrap) return;
  if (!SNAP || !SNAP.nodes || SNAP.nodes.length === 0) {
    destroyTopology();
    wrap.innerHTML = '<div class="small">No topology data yet.</div>';
    return;
  }

  const assignments = new Map();
  if (LAST_PLAN && Array.isArray(LAST_PLAN.per_stage)) {
    LAST_PLAN.per_stage.forEach(stage => {
      if (!stage || stage.infeasible || !stage.node) return;
      const prev = assignments.get(stage.node) || {count: 0, stages: []};
      prev.count += 1;
      if (stage.id) prev.stages.push(stage.id);
      assignments.set(stage.node, prev);
    });
  }

  const fallbackAssignments = new Map();
  if (LAST_PLAN && Array.isArray(LAST_PLAN.per_stage)) {
    LAST_PLAN.per_stage.forEach(stage => {
      if (!stage || !Array.isArray(stage.fallbacks)) return;
      stage.fallbacks.forEach(name => {
        if (!name) return;
        const prev = fallbackAssignments.get(name) || {count: 0, stages: []};
        prev.count += 1;
        if (stage.id) prev.stages.push(stage.id);
        fallbackAssignments.set(name, prev);
      });
    });
  }

  const fedMap = SNAP.node_federations || {};
  const nodes = SNAP.nodes.map(n => {
    const eff = n.effective || {};
    const dyn = n.dyn || {};
    const maxCpu = Number(eff.max_cpu_cores || 0);
    const freeCpu = Number(eff.free_cpu_cores || 0);
    const maxMem = Number(eff.max_mem_gb || 0);
    const freeMem = Number(eff.free_mem_gb || 0);
    const res = dyn.reservations ? Object.keys(dyn.reservations).length : 0;
    const assign = assignments.get(n.name);
    const fallback = fallbackAssignments.get(n.name);
    return {
      id: n.name,
      arch: n.arch,
      class: n.class,
      labels: n.labels || {},
      down: Boolean(dyn.down),
      derate: Number(dyn.thermal_derate || 0),
      cpuCap: maxCpu,
      cpuUsed: Math.max(0, maxCpu - freeCpu),
      memCap: maxMem,
      memUsed: Math.max(0, maxMem - freeMem),
      reservations: res,
      assignCount: assign ? assign.count : 0,
      assignStages: assign ? assign.stages : [],
      shadowCount: fallback ? fallback.count : 0,
      shadowStages: fallback ? fallback.stages : [],
      federation: fedMap[n.name] || (n.labels || {}).federation || (n.labels || {}).zone || '—',
    };
  });

  const lookup = new Map(nodes.map(n => [n.id, n]));
  const links = (SNAP.links || [])
    .filter(l => lookup.has(l.a) && lookup.has(l.b))
    .map(l => {
      const eff = l.effective || {};
      return {
        source: l.a,
        target: l.b,
        down: Boolean(eff.down),
        speed: Number(eff.speed_gbps || 0),
        loss: Number(eff.loss_pct || 0),
        rtt: Number(eff.rtt_ms || 0),
        jitter: Number(eff.jitter_ms || 0),
        key: l.key,
      };
    });

  const width = wrap.clientWidth || 720;
  const height = Math.max(360, Math.min(760, 180 + nodes.length * 14));
  wrap.innerHTML = '';

  const svg = d3
    .select(wrap)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g');

  const linkWidth = l => 1.5 + Math.log1p(Math.max(0.2, l.speed));
  const linkColor = l => {
    if (l.down) return '#e74c3c';
    if (l.loss >= 2.0 || l.jitter >= 2.0) return '#f39c12';
    return '#2c3f57';
  };

  const linkGroup = g.append('g').attr('class', 'links');
  const link = linkGroup
    .selectAll('line')
    .data(links)
    .enter()
    .append('line')
    .attr('class', 'link')
    .attr('stroke', linkColor)
    .attr('stroke-width', linkWidth);

  link.append('title').text(l => {
    const parts = [
      `${l.source} ↔ ${l.target}`,
      `speed: ${fmt(l.speed, 2)} Gbps`,
      `rtt: ${fmt(l.rtt, 1)} ms`,
      `loss: ${fmt(l.loss, 2)} %`,
    ];
    if (l.down) parts.push('status: DOWN');
    return parts.join('\n');
  });

  const nodeRadius = d => {
    const cpu = Math.max(0, d.cpuCap);
    const mem = Math.max(0, d.memCap);
    const cpuTerm = Math.log10(cpu + 1) * 10;
    const memTerm = Math.log10(mem + 1) * 4;
    return 12 + Math.min(22, cpuTerm + memTerm);
  };

  const nodeColor = d => {
    if (d.down) return '#e74c3c';
    if (d.derate > 0.01) return '#f1c40f';
    const pct = d.cpuCap > 0 ? Math.min(1, d.cpuUsed / d.cpuCap) : 0;
    return d3.interpolateBlues(0.3 + pct * 0.6);
  };

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      'link',
      d3
        .forceLink(links)
        .id(d => d.id)
        .distance(l => {
          const base = 160;
          const speed = Math.max(0.2, l.speed || 0.2);
          return base / Math.sqrt(speed);
        })
        .strength(0.6)
    )
    .force('charge', d3.forceManyBody().strength(-260))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 16));

  const drag = sim => {
    function dragstarted(event, d) {
      if (!event.active) sim.alphaTarget(0.2).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) sim.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    return d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended);
  };

  const nodesGroup = g.append('g').attr('class', 'nodes');
  const node = nodesGroup
    .selectAll('g')
    .data(nodes)
    .enter()
    .append('g')
    .call(drag(simulation));

  node
    .append('circle')
    .attr('class', 'node-shadow')
    .attr('r', d => nodeRadius(d) + 9)
    .style('display', d => (d.shadowCount > 0 ? 'block' : 'none'));

  node
    .append('circle')
    .attr('class', 'node-ring')
    .attr('r', d => nodeRadius(d) + 5)
    .attr('stroke', d => (d.assignCount > 0 ? '#6fc1ff' : '#2ecc71'))
    .attr('stroke-dasharray', d => (d.assignCount > 0 ? null : '6 4'))
    .style('display', d => (d.assignCount > 0 || d.reservations > 0 ? 'block' : 'none'));

  node
    .append('circle')
    .attr('class', 'node-core')
    .attr('r', d => nodeRadius(d))
    .attr('fill', nodeColor);

  node
    .append('text')
    .attr('class', 'node-label')
    .attr('dy', 4)
    .text(d => d.id);

  node.append('title').text(d => {
    const status = d.down
      ? 'status: DOWN'
      : d.derate > 0.01
      ? `thermal derate: ${fmtPct(Math.min(1, d.derate))}`
      : 'status: healthy';
    const res = `reservations: ${d.reservations}`;
    const assign = d.assignCount
      ? `latest plan stages: ${d.assignStages.join(', ')}`
      : 'latest plan stages: none';
    const fallback = d.shadowCount
      ? `fallback stages: ${d.shadowStages.join(', ')}`
      : 'fallback stages: none';
    return [
      d.id,
      `${d.arch || '—'} ${d.class || ''}`.trim(),
      status,
      `cpu ${fmt(d.cpuUsed, 1)} / ${fmt(d.cpuCap, 1)} cores`,
      `mem ${fmt(d.memUsed, 1)} / ${fmt(d.memCap, 1)} GB`,
      `federation: ${d.federation}`,
      res,
      assign,
      fallback,
    ]
      .filter(Boolean)
      .join('\n');
  });

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  destroyTopology();
  TOPO_SIM = simulation;
}

function renderPlanGraph() {
  const wrap = document.getElementById('plan_graph');
  if (!wrap) return;
  if (!LAST_PLAN) {
    wrap.innerHTML = '<div class="small">No plans yet.</div>';
    return;
  }
  const per = LAST_PLAN.per_stage || [];
  if (!per.length) {
    wrap.innerHTML = '<div class="small">Plan contains no stages.</div>';
    return;
  }
  const parts = per.map(s => {
    const node = s.node || '—';
    const fmtBadge = s.format ? `<span class="fmt">${s.format}</span>` : '';
    const cls = s.infeasible ? 'stage-card bad' : 'stage-card';
    const metrics = `<div class="metrics">c:${fmt(s.compute_ms,1)} ms • x:${fmt(s.xfer_ms,1)} ms</div>`;
    const reason = s.infeasible && s.reason ? `<div class="small">${s.reason}</div>` : '';
    const badgeHtml = s.infeasible ? `<div class="badge bad">Blocked</div>` : '';
    const fallbackNodes = Array.isArray(s.fallbacks) ? s.fallbacks : [];
    const fallbackFeds = Array.isArray(s.fallback_federations) ? s.fallback_federations : [];
    const fallbackPairs = fallbackNodes.map((name, idx) => {
      const fed = fallbackFeds[idx];
      return `<span class="mono">${name}</span>${fed ? tag(`fed:${fed}`) : ''}`;
    }).join(' ');
    const fallback = fallbackPairs
      ? `<div class="small">Fallback: ${fallbackPairs}</div>`
      : '';
    const fed = s.federation ? `<div class="small">Federation: ${s.federation}</div>` : '';
    return `<div class="${cls}"><div class="mono">${s.id||'?'}</div><div class="node">${node}</div>${fmtBadge}${metrics}${fallback}${fed}${badgeHtml}${reason}</div>`;
  }).join('<div class="stage-arrow">→</div>');
  wrap.innerHTML = `<div class="dag-row">${parts}</div>`;
  const spreadStr = LAST_PLAN && LAST_PLAN.federation_spread !== undefined && LAST_PLAN.federation_spread !== null ? ` • Spread ${fmt(LAST_PLAN.federation_spread,2)}` : '';
  const resilienceStr = LAST_PLAN && LAST_PLAN.resilience_score !== undefined && LAST_PLAN.resilience_score !== null ? ` • Resilience ${fmtPct(LAST_PLAN.resilience_score)}` : '';
  wrap.insertAdjacentHTML('beforeend', `<div class="small" style="margin-top:6px;">Latency ${fmt(LAST_PLAN.latency_ms,1)} ms • Energy ${fmt(LAST_PLAN.energy_kj,3)} kJ • Risk ${fmt(LAST_PLAN.risk,3)}${spreadStr}${resilienceStr}</div>`);
}

function renderPlans() {
  const tb = document.getElementById('plans_tbody');
  tb.innerHTML = '';
  fetchJSON('/api/plans').then(data => {
    LAST_PLAN = data.length ? data[0] : null;
    data.forEach(p => {
      const stages = (p.per_stage||[]).map(s => {
        const nodeName = s.node || '—';
        const fmtTag = s.format ? tag(s.format) : '';
        const inf = s.infeasible ? badge('X','bad') : '';
        const fallbackNodes = Array.isArray(s.fallbacks) ? s.fallbacks : [];
        const fallbackFeds = Array.isArray(s.fallback_federations) ? s.fallback_federations : [];
        const fallbackPairs = fallbackNodes.map((name, idx) => {
          const fed = fallbackFeds[idx];
          const fedTag = fed ? tag(`fed:${fed}`) : '';
          return `<span class="mono">${name}</span>${fedTag}`;
        }).join(' ');
        const fallbackHtml = fallbackPairs ? `<div class="small">Fallback: ${fallbackPairs}</div>` : '';
        const fedTag = s.federation ? `<div class="small">Federation: ${s.federation}</div>` : '';
        const reason = s.infeasible && s.reason ? `<div class="small">${s.reason}</div>` : '';
        return `<div class="small"><span class="mono">${s.id||'?'}</span> → <b>${nodeName}</b> ${fmtTag} ${inf} <span class="mono">c:${fmt(s.compute_ms,1)}ms</span> <span class="mono">x:${fmt(s.xfer_ms,1)}ms</span>${fallbackHtml}${fedTag}${reason}<\/div>`;
      }).join('');
      const spreadVal = p.federation_spread !== null && p.federation_spread !== undefined ? fmt(p.federation_spread, 2) : '—';
      const feds = (p.federations_in_use || []).map(tag).join(' ');
      const spreadCell = `${spreadVal}${feds ? `<div class="small">${feds}</div>` : ''}`;
      const resilienceVal = p.resilience_score !== null && p.resilience_score !== undefined ? fmtPct(p.resilience_score) : '—';
      const crossVal = p.cross_federation_fallback_ratio !== null && p.cross_federation_fallback_ratio !== undefined ? fmtPct(p.cross_federation_fallback_ratio) : '—';
      tb.insertAdjacentHTML('beforeend', `
        <tr>
          <td><div class="mono">${p.job_id||'—'}</div><div class="small">${p.strategy||'greedy'} ${p.dry_run?'(dry)':''}</div></td>
          <td>${fmt(p.latency_ms,1)}</td>
          <td>${fmt(p.energy_kj,3)}</td>
          <td>${fmt(p.risk,3)}</td>
          <td>${spreadCell}</td>
          <td>${resilienceVal}</td>
          <td>${crossVal}</td>
          <td>${p.infeasible ? badge('Yes','bad') : badge('No','good')}</td>
          <td>${stages}</td>
        </tr>
      `);
    });
    renderPlanGraph();
    renderTopology();
    renderOverview();
  }).catch(e => {
    tb.innerHTML = `<tr><td colspan="9" class="small">No plans yet.</td></tr>`;
    LAST_PLAN = null;
    renderPlanGraph();
    renderTopology();
    renderOverview();
  });
}

async function refresh() {
  try {
    const data = await fetchJSON('/api/snapshot');
    SNAP = data;
    renderOverview();
    renderNodes();
    renderLinks();
    renderFederations();
    renderFederationLinks();
    renderTopology();
    renderPlans();
  } catch (e) {
    console.error(e);
    alert('Failed to refresh snapshot: '+e.message);
  }
}

async function submitPlan() {
  const txt = document.getElementById('jobJson').value.trim();
  if (!txt) { alert('Paste a job JSON first'); return; }
  let job;
  try { job = JSON.parse(txt); } catch(e) { alert('Invalid JSON: '+e.message); return; }
  const dry = document.getElementById('dryRun').checked;
  const strategy = document.getElementById('strategySelect').value || 'greedy';
  try {
    await fetchJSON('/api/plan', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({job: job, dry_run: dry, strategy})});
    await refresh();
  } catch(e) {
    alert('Plan failed: '+e.message);
  }
}

async function runDemo(dry=true) {
  const strategy = document.getElementById('strategySelect').value || 'greedy';
  try {
    await fetchJSON('/api/plan_demo', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({dry_run: dry, strategy})});
    await refresh();
  } catch(e) {
    alert('Demo failed: '+e.message);
  }
}

async function applyObs() {
  const txt = document.getElementById('obsJson').value.trim();
  if (!txt) { alert('Paste an observation JSON'); return; }
  let payload;
  try { payload = JSON.parse(txt); } catch(e) { alert('Invalid JSON: '+e.message); return; }
  try {
    await fetchJSON('/api/observe', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload)});
    await refresh();
  } catch(e) {
    alert('Observation failed: '+e.message);
  }
}

window.addEventListener('resize', () => {
  if (TOPO_RESIZE) clearTimeout(TOPO_RESIZE);
  TOPO_RESIZE = setTimeout(() => {
    renderTopology();
  }, 200);
});

setInterval(refresh, 4000);
window.addEventListener('load', refresh);
</script>

</body>
</html>
"""


@app.get("/")
def index():
    resp = make_response(_INDEX_HTML.replace("__REMOTE_LABEL__", REMOTE_LABEL))
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


# ----------------- CLI entry -----------------


def main():
    ap = argparse.ArgumentParser(description="Fabric DT Dashboard")
    ap.add_argument("--host", default=os.environ.get("FABRIC_UI_HOST", "127.0.0.1"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("FABRIC_UI_PORT", "8090"))
    )
    ap.add_argument(
        "--remote", help="Remote Fabric DT API base URL (e.g. http://127.0.0.1:8080)"
    )
    ap.add_argument(
        "--remote-timeout",
        type=float,
        default=REMOTE_TIMEOUT,
        help="Timeout in seconds when contacting the remote API",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.remote is not None or args.remote_timeout != REMOTE_TIMEOUT:
        target_remote = args.remote if args.remote is not None else REMOTE_BASE
        _configure_runtime(target_remote, args.remote_timeout)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
