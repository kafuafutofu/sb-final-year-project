#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/run_plan.py — submit jobs to the Fabric DT (local or remote).

Usage
-----
# Local (imports dt.* directly)
python3 -m planner.run_plan --job jobs/jobs_10.yaml --dry-run

# Remote (use if dt/api.py is running on another process/machine)
python3 -m planner.run_plan --remote http://127.0.0.1:8080 --job jobs/jobs_10.yaml --dry-run

# Single job file, save result JSON
python3 -m planner.run_plan --job jobs/demo.yaml --out /tmp/plan.json

Options
-------
--job PATH            YAML file containing a single job object or a list under 'jobs'
--dry-run             Plan without reserving capacity (default: True)
--strategy STR        greedy (default) or cheapest-energy (local and remote)
--remote URL          If provided, POSTs to {URL}/plan or {URL}/plan_batch
--repeat N            Repeat planning N times (useful for Monte-Carlo learning with bandit; local mode)
--out PATH            Save full JSON result(s) here
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Optional pretty console
try:
    from rich.console import Console
    from rich.table import Table
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None  # type: ignore

# Local DT imports (guarded so remote mode can run without deps)
def _try_import_local():
    try:
        from dt.state import DTState
        from dt.cost_model import CostModel
        from dt.policy.greedy import GreedyPlanner
        try:
            from dt.policy.bandit import BanditPolicy
        except Exception:
            BanditPolicy = None  # type: ignore
        return DTState, CostModel, GreedyPlanner, BanditPolicy
    except Exception as e:
        return None, None, None, None

def load_yaml(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_jobs(obj: Any) -> List[Dict[str, Any]]:
    # Accept either:
    #  - {id, stages[]} (single job)
    #  - {"jobs": [ ... ]}
    #  - [ {...}, {...} ]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "jobs" in obj and isinstance(obj["jobs"], list):
            return obj["jobs"]
        # assume single job object
        return [obj]
    raise ValueError("YAML must be a job object, a list of jobs, or a dict with 'jobs': [...]")

def print_summary(results: List[Dict[str, Any]]):
    if not RICH:
        # Minimal stdout
        for r in results:
            feasible = not r.get("infeasible")
            print(
                f"\nJob {r.get('job_id')}: latency={r.get('latency_ms')}ms  energy={r.get('energy_kj')}kJ  risk={r.get('risk')}  feasible={feasible}"
            )
            for s in (r.get("per_stage") or []):
                print(f"  - {s.get('id')} → {s.get('node')}  fmt={s.get('format')}  c={s.get('compute_ms')}ms  x={s.get('xfer_ms')}ms  risk={s.get('risk')}")
        return

    # Pretty table
    tbl = Table(title="Plan Summary", show_lines=False)
    tbl.add_column("Job", style="bold")
    tbl.add_column("Latency (ms)", justify="right")
    tbl.add_column("Energy (kJ)", justify="right")
    tbl.add_column("Risk", justify="right")
    tbl.add_column("Feasible", justify="center")
    tbl.add_column("Stages", style="dim")

    for r in results:
        stages_str = "\n".join(
            f"[dim]{s.get('id')}[/dim] → [b]{s.get('node') or '—'}[/b] "
            f"{('['+str(s.get('format'))+']') if s.get('format') else ''} "
            f"(c:{s.get('compute_ms')}ms, x:{s.get('xfer_ms')}ms)"
            for s in (r.get("per_stage") or [])
        )
        tbl.add_row(
            str(r.get("job_id") or "—"),
            f"{r.get('latency_ms')}",
            f"{r.get('energy_kj')}",
            f"{r.get('risk')}",
            "✅" if not r.get("infeasible") else "❌",
            stages_str or "—",
        )
    console.print(tbl)  # type: ignore

def plan_remote(
    base_url: str,
    jobs: List[Dict[str, Any]],
    dry_run: bool,
    strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    import requests  # only needed in remote mode
    base = base_url.rstrip("/")

    if len(jobs) == 1:
        payload = {"job": jobs[0], "dry_run": dry_run}
        if strategy:
            payload["strategy"] = strategy
        r = requests.post(f"{base}/plan", json=payload, timeout=60)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"remote /plan error: {j}")
        return [j["data"]]
    else:
        payload = {"jobs": jobs, "dry_run": dry_run}
        if strategy:
            payload["strategy"] = strategy
        r = requests.post(f"{base}/plan_batch", json=payload, timeout=120)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"remote /plan_batch error: {j}")
        # /plan_batch wraps results under data.results
        data = j["data"]
        if isinstance(data, dict) and "results" in data:
            return list(data["results"])
        # fallback
        return list(data)

def plan_local(jobs: List[Dict[str, Any]], dry_run: bool, strategy: str, repeat: int) -> List[Dict[str, Any]]:
    DTState, CostModel, GreedyPlanner, BanditPolicy = _try_import_local()
    if DTState is None:
        raise RuntimeError("Local DT modules not importable. Did you run from project root and install requirements?")

    state = DTState()
    cm = CostModel(state)
    bandit = BanditPolicy(persist_path="sim/bandit_state.json") if BanditPolicy else None
    planner = GreedyPlanner(state, cm, bandit=bandit, cfg={
        "risk_weight": 10.0,
        "energy_weight": 0.0 if strategy == "greedy" else 0.1,
        "prefer_locality_bonus_ms": 0.5,
        "require_format_match": False,
    })

    results: List[Dict[str, Any]] = []
    for job in jobs:
        last = None
        times = max(1, int(repeat))
        for _ in range(times):
            res = planner.plan_job(job, dry_run=dry_run)
            # store last; bandit (if any) will learn across repeats
            last = res
        if last is not None:
            results.append(last)
    return results

def main():
    ap = argparse.ArgumentParser(description="Fabric DT — run planner over job YAML(s)")
    ap.add_argument("--job", required=True, help="Path to job YAML (single job, list of jobs, or {jobs: [...]})")
    ap.add_argument("--dry-run", action="store_true", help="Plan without reserving")
    ap.add_argument("--remote", default=None, help="Base URL of dt/api (e.g., http://127.0.0.1:8080)")
    ap.add_argument("--strategy", default="greedy", choices=["greedy", "cheapest-energy"], help="Scoring preference")
    ap.add_argument("--repeat", type=int, default=1, help="Local-only: repeat planning N times (bandit learning)")
    ap.add_argument("--out", default=None, help="Write JSON results to this path")
    args = ap.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"error: job file not found: {job_path}", file=sys.stderr)
        sys.exit(2)

    try:
        obj = load_yaml(job_path)
        jobs = ensure_jobs(obj)
    except Exception as e:
        print(f"error: failed to load/parse job YAML: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        if args.remote:
            results = plan_remote(args.remote, jobs, dry_run=bool(args.dry_run), strategy=args.strategy)
        else:
            results = plan_local(jobs, dry_run=bool(args.dry_run), strategy=args.strategy, repeat=args.repeat)
    except Exception as e:
        print(f"error: planning failed: {e}", file=sys.stderr)
        sys.exit(1)

    print_summary(results)

    if args.out:
        outp = Path(args.out)
        try:
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(results, indent=2), encoding="utf-8")
            if RICH:
                console.print(f"[green]Saved results →[/green] {outp}")  # type: ignore
            else:
                print(f"Saved results -> {outp}")
        except Exception as e:
            print(f"warn: failed to write --out file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

