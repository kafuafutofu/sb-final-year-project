#!/usr/bin/env python3
"""
Chaos engine for Fabric simulator.

- Loads topology.yaml (validated against schemas/topology.schema.yaml).
- Selects a scenario (or uses top-level chaos).
- Builds an event schedule with start/end (for events with duration).
- Applies/reverts overrides either:
    a) by writing to sim/overrides.json (default), or
    b) by POSTing to a DT observe endpoint (--dt http://127.0.0.1:5055/observe).

Usage:
  python3 sim/chaos.py --topology topology.yaml --scenario SCENARIO_NAME --speed 10 --run
  python3 sim/chaos.py --topology topology.yaml --dry-run
  python3 sim/chaos.py --topology topology.yaml --dt http://127.0.0.1:5055/observe --run

The overrides.json format:
{
  "links": { "A|B": { "speed_gbps": ..., "rtt_ms": ..., "down": true, ... }, ... },
  "nodes": { "node-001": { "down": true, "power_cap_w": 120, "thermal_derate": 0.3, ... }, ... }
}

Your DT (dt/state.py) should periodically read sim/overrides.json and merge into live state.
"""

import argparse, time, json, threading, signal, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

try:
    import requests
except Exception:
    requests = None  # optional

# ----------------------------- Config -----------------------------

DEFAULT_OVERRIDES_PATH = Path(__file__).parent / "overrides.json"
DEFAULT_NODES_DIR = Path(__file__).resolve().parent.parent / "nodes"

# Event kinds we support and their allowed fields
LINK_KINDS = {"link_degrade", "link_loss_spike", "link_down", "link_up"}
NODE_KINDS = {"node_kill", "node_recover", "power_cap", "thermal_derate", "clock_skew", "packet_dup", "packet_reorder"}
GROUP_KINDS = {"zone_blackout", "zone_recover", "federation_partition"}

# ----------------------------- Helpers -----------------------------

def link_key(a: str, b: str) -> str:
    """Stable undirected key."""
    return "|".join(sorted([a, b]))

def now_ms() -> int:
    return int(time.time() * 1000)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ----------------------------- Data classes -----------------------------

@dataclass(order=True)
class ChaosEvent:
    sort_index: float = field(init=False, repr=False)
    at_s: float
    kind: str
    duration_s: float = 0.0
    a: Optional[str] = None
    b: Optional[str] = None
    node: Optional[str] = None
    label: Optional[str] = None
    value: Optional[str] = None
    value_b: Optional[str] = None
    # Link modifiers
    speed_gbps: Optional[float] = None
    rtt_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    loss_pct: Optional[float] = None
    ecn: Optional[bool] = None
    # Node modifiers
    power_cap_w: Optional[float] = None
    thermal_derate: Optional[float] = None
    skew_ms: Optional[float] = None
    # Packets (logical)
    packet_dup: Optional[float] = None
    packet_reorder: Optional[float] = None

    def __post_init__(self):
        self.sort_index = self.at_s

    def end_time(self) -> Optional[float]:
        return (self.at_s + self.duration_s) if self.duration_s and self.duration_s > 0 else None

# ----------------------------- I/O -----------------------------

def load_topology(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def scenario_by_name(topology: Dict[str, Any], name: str) -> Dict[str, Any]:
    for sc in (topology.get("scenarios") or []):
        if sc.get("name") == name:
            return sc
    raise ValueError(f"Scenario '{name}' not found in topology.")

def collect_chaos_events(topology: Dict[str, Any], scenario: Optional[str]) -> List[ChaosEvent]:
    base_events = topology.get("chaos") or []
    sc_events: List[Dict[str, Any]] = []
    if scenario:
        sc = scenario_by_name(topology, scenario)
        sc_events = sc.get("chaos") or []
    raw = base_events + sc_events
    events: List[ChaosEvent] = []
    for e in raw:
        kind = e["kind"]
        at_s = float(e["at_s"])
        duration_s = float(e.get("duration_s", 0.0) or 0.0)
        ce = ChaosEvent(
            at_s=at_s,
            kind=kind,
            duration_s=duration_s,
            a=e.get("a"),
            b=e.get("b"),
            node=e.get("node"),
            label=e.get("label"),
            value=e.get("value"),
            value_b=e.get("value_b") or e.get("other"),
            speed_gbps=e.get("speed_gbps"),
            rtt_ms=e.get("rtt_ms"),
            jitter_ms=e.get("jitter_ms"),
            loss_pct=e.get("loss_pct"),
            ecn=e.get("ecn"),
            power_cap_w=e.get("power_cap_w"),
            thermal_derate=e.get("thermal_derate"),
            skew_ms=e.get("skew_ms"),
            packet_dup=e.get("packet_dup"),
            packet_reorder=e.get("packet_reorder"),
        )
        events.append(ce)
        # For bounded events, we inject an implicit "revert" marker
        if ce.end_time() is not None and kind in (LINK_KINDS | NODE_KINDS | GROUP_KINDS):
            # Add an internal revert event
            revert = ChaosEvent(
                at_s=ce.end_time(),
                kind=f"__revert__::{kind}",
                duration_s=0.0,
                a=ce.a,
                b=ce.b,
                node=ce.node,
                label=ce.label,
                value=ce.value,
                value_b=ce.value_b,
            )
            events.append(revert)
    events.sort()
    return events

# ----------------------------- Overrides store -----------------------------


def load_nodes_index(nodes_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load node descriptors for label-based chaos events."""
    index: Dict[str, Dict[str, Any]] = {}
    try:
        for path in sorted(Path(nodes_dir).glob("*.yaml")):
            data = yaml.safe_load(path.read_text())
            name = data.get("name")
            if name:
                index[name] = data
    except Exception:
        index = {}
    return index

class OverridesStore:
    """Backs overrides.json AND (optionally) posts to DT."""
    def __init__(self, path: Path, dt_endpoint: Optional[str] = None):
        self.path = path
        self.dt_endpoint = dt_endpoint
        self.state = {"links": {}, "nodes": {}}
        # Preload existing state if present
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
        except Exception:
            self.state = {"links": {}, "nodes": {}}

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

    def _post_dt(self, payload: Dict[str, Any], action: str):
        if not self.dt_endpoint or not requests:
            return
        try:
            requests.post(self.dt_endpoint, json={"action": action, "payload": payload}, timeout=2.0)
        except Exception:
            # Non-fatal in demo; DT may be offline
            pass

    # ----- Link ops -----
    def link_apply(self, a: str, b: str, changes: Dict[str, Any]):
        k = link_key(a, b)
        cur = self.state["links"].get(k, {})
        cur.update(changes)
        self.state["links"][k] = cur
        self._write()
        self._post_dt({"type": "link", "key": k, "changes": changes}, "apply")

    def link_revert(self, a: str, b: str, fields: List[str]):
        k = link_key(a, b)
        cur = self.state["links"].get(k, {})
        for f in fields:
            if f in cur:
                del cur[f]
        # Cleanup empty entries
        if cur:
            self.state["links"][k] = cur
        else:
            self.state["links"].pop(k, None)
        self._write()
        self._post_dt({"type": "link", "key": k, "fields": fields}, "revert")

    # ----- Node ops -----
    def node_apply(self, node: str, changes: Dict[str, Any]):
        cur = self.state["nodes"].get(node, {})
        cur.update(changes)
        self.state["nodes"][node] = cur
        self._write()
        self._post_dt({"type": "node", "node": node, "changes": changes}, "apply")

    def node_revert(self, node: str, fields: List[str]):
        cur = self.state["nodes"].get(node, {})
        for f in fields:
            if f in cur:
                del cur[f]
        if cur:
            self.state["nodes"][node] = cur
        else:
            self.state["nodes"].pop(node, None)
        self._write()
        self._post_dt({"type": "node", "node": node, "fields": fields}, "revert")

# ----------------------------- Engine -----------------------------

class ChaosEngine:
    def __init__(
        self,
        store: OverridesStore,
        speed: float = 1.0,
        verbose: bool = True,
        nodes_index: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.store = store
        self.speed = max(0.01, speed)
        self.verbose = verbose
        self._stop = threading.Event()
        self.nodes_index = nodes_index or {}
        self.label_index = self._build_label_index()

    def stop(self):
        self._stop.set()

    def log(self, msg: str):
        if self.verbose:
            print(f"[chaos] {msg}")

    # -------- label helpers --------

    def _build_label_index(self) -> Dict[str, Dict[str, List[str]]]:
        idx: Dict[str, Dict[str, List[str]]] = {}
        for name, node in self.nodes_index.items():
            labels = dict(node.get("labels") or {})
            if "federation" not in labels:
                for key in ("zone", "site", "region"):
                    if labels.get(key):
                        labels.setdefault("federation", labels[key])
                        break
            for key, val in labels.items():
                if val is None:
                    continue
                sval = str(val)
                bucket = idx.setdefault(key, {})
                names = bucket.setdefault(sval, [])
                names.append(name)
        for val_map in idx.values():
            for key in list(val_map.keys()):
                val_map[key] = sorted(set(val_map[key]))
        return idx

    def _nodes_for(self, label: Optional[str], value: Optional[str]) -> List[str]:
        if not label or value is None:
            return []
        label = str(label)
        value = str(value)
        return list(self.label_index.get(label, {}).get(value, []))

    # -------- grouped actions --------

    def _apply_zone_blackout(self, ev: ChaosEvent):
        nodes = self._nodes_for(ev.label, ev.value)
        if not nodes:
            self.log(f"SKIP zone_blackout {ev.label}={ev.value}: no nodes")
            return
        for node in nodes:
            self.store.node_apply(node, {"down": True})
        self.log(f"ZONE BLACKOUT {ev.label}={ev.value} ({len(nodes)} nodes)")

    def _apply_zone_recover(self, ev: ChaosEvent):
        nodes = self._nodes_for(ev.label, ev.value)
        if not nodes:
            return
        for node in nodes:
            self.store.node_revert(node, ["down"])
        self.log(f"ZONE RECOVER {ev.label}={ev.value} ({len(nodes)} nodes)")

    def _apply_federation_partition(self, ev: ChaosEvent):
        label = ev.label or "federation"
        value_a = ev.value
        value_b = ev.value_b
        if value_a is None or value_b is None:
            self.log("SKIP federation_partition: missing value")
            return
        group_a = self._nodes_for(label, value_a)
        group_b = self._nodes_for(label, value_b)
        if not group_a or not group_b:
            self.log(
                f"SKIP federation_partition {label}:{value_a}<->{value_b}: group missing"
            )
            return

        fields: Dict[str, Any] = {}
        if ev.speed_gbps is not None:
            fields["speed_gbps"] = max(0.0, ev.speed_gbps)
        if ev.rtt_ms is not None:
            fields["rtt_ms"] = max(0.0, ev.rtt_ms)
        if ev.jitter_ms is not None:
            fields["jitter_ms"] = max(0.0, ev.jitter_ms)
        if ev.loss_pct is not None:
            fields["loss_pct"] = max(0.0, min(100.0, ev.loss_pct))
        if not fields:
            fields = {"loss_pct": 12.0, "rtt_ms": 35.0}

        for a in group_a:
            for b in group_b:
                self.store.link_apply(a, b, dict(fields))

        # Also expose federation level link for dashboards/DT state
        self.store.link_apply(str(value_a), str(value_b), dict(fields))
        self.log(
            f"FEDERATION PARTITION {label}:{value_a}<->{value_b} ({len(group_a)*len(group_b)} pairs)"
        )

    def _revert_federation_partition(self, ev: ChaosEvent):
        label = ev.label or "federation"
        value_a = ev.value
        value_b = ev.value_b
        if value_a is None or value_b is None:
            return
        group_a = self._nodes_for(label, value_a)
        group_b = self._nodes_for(label, value_b)
        for a in group_a:
            for b in group_b:
                self.store.link_revert(
                    a, b, ["speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "down"]
                )
        self.store.link_revert(
            str(value_a), str(value_b), ["speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "down"]
        )
        self.log(f"FEDERATION HEAL {label}:{value_a}<->{value_b}")

    def apply_event(self, ev: ChaosEvent):
        k = ev.kind
        # Handle reverts
        if k.startswith("__revert__::"):
            orig = k.split("::", 1)[1]
            return self._revert_for(orig, ev)

        if k in LINK_KINDS:
            if not ev.a or not ev.b:
                self.log(f"SKIP {k}: missing link endpoints")
                return
            if k == "link_down":
                self.store.link_apply(ev.a, ev.b, {"down": True})
                self.log(f"LINK DOWN {ev.a}<->{ev.b}")
            elif k == "link_up":
                # Remove 'down' flag
                self.store.link_revert(ev.a, ev.b, ["down"])
                self.log(f"LINK UP {ev.a}<->{ev.b}")
            elif k == "link_loss_spike":
                fields = {}
                if ev.loss_pct is not None: fields["loss_pct"] = max(0.0, min(100.0, ev.loss_pct))
                self.store.link_apply(ev.a, ev.b, fields)
                self.log(f"LOSS SPIKE {ev.a}<->{ev.b} {fields}")
            elif k == "link_degrade":
                fields = {}
                if ev.speed_gbps is not None: fields["speed_gbps"] = max(0.0, ev.speed_gbps)
                if ev.rtt_ms is not None: fields["rtt_ms"] = max(0.0, ev.rtt_ms)
                if ev.jitter_ms is not None: fields["jitter_ms"] = max(0.0, ev.jitter_ms)
                if ev.loss_pct is not None: fields["loss_pct"] = max(0.0, min(100.0, ev.loss_pct))
                if ev.ecn is not None: fields["ecn"] = bool(ev.ecn)
                self.store.link_apply(ev.a, ev.b, fields)
                self.log(f"DEGRADE {ev.a}<->{ev.b} {fields}")
            return

        if k in NODE_KINDS:
            if k in ("node_kill", "node_recover") and not ev.node:
                self.log(f"SKIP {k}: missing node")
                return

            if k == "node_kill":
                self.store.node_apply(ev.node, {"down": True})
                self.log(f"NODE DOWN {ev.node}")
            elif k == "node_recover":
                self.store.node_revert(ev.node, ["down"])
                self.log(f"NODE UP {ev.node}")
            elif k == "power_cap":
                # limit power (your DT can derate clocks)
                if not ev.node: return
                cap = ev.power_cap_w if ev.power_cap_w is not None else 0
                self.store.node_apply(ev.node, {"power_cap_w": max(0.0, cap)})
                self.log(f"POWER CAP {ev.node} -> {cap}W")
            elif k == "thermal_derate":
                if not ev.node: return
                td = clamp01(ev.thermal_derate if ev.thermal_derate is not None else 0.2)
                self.store.node_apply(ev.node, {"thermal_derate": td})
                self.log(f"THERMAL DERATE {ev.node} -> {td}")
            elif k == "clock_skew":
                if not ev.node: return
                skew = ev.skew_ms if ev.skew_ms is not None else 50.0
                self.store.node_apply(ev.node, {"clock_skew_ms": skew})
                self.log(f"CLOCK SKEW {ev.node} -> {skew} ms")
            elif k == "packet_dup":
                if not ev.node: return
                pd = clamp01(ev.packet_dup if ev.packet_dup is not None else 0.1)
                self.store.node_apply(ev.node, {"packet_dup": pd})
                self.log(f"PACKET DUP {ev.node} -> p={pd}")
            elif k == "packet_reorder":
                if not ev.node: return
                pr = clamp01(ev.packet_reorder if ev.packet_reorder is not None else 0.1)
                self.store.node_apply(ev.node, {"packet_reorder": pr})
                self.log(f"PACKET REORDER {ev.node} -> p={pr}")
            return

        if k in GROUP_KINDS:
            if k == "zone_blackout":
                self._apply_zone_blackout(ev)
            elif k == "zone_recover":
                self._apply_zone_recover(ev)
            elif k == "federation_partition":
                self._apply_federation_partition(ev)
            return

        self.log(f"UNKNOWN EVENT KIND: {k}")

    def _revert_for(self, original_kind: str, ev: ChaosEvent):
        """Best-effort revert for bounded-duration events."""
        if original_kind in LINK_KINDS and ev.a and ev.b:
            if original_kind == "link_down":
                self.store.link_revert(ev.a, ev.b, ["down"])
                self.log(f"REVERT link_down {ev.a}<->{ev.b}")
            elif original_kind == "link_loss_spike":
                self.store.link_revert(ev.a, ev.b, ["loss_pct"])
                self.log(f"REVERT loss_spike {ev.a}<->{ev.b}")
            elif original_kind == "link_degrade":
                self.store.link_revert(ev.a, ev.b, ["speed_gbps","rtt_ms","jitter_ms","loss_pct","ecn"])
                self.log(f"REVERT link_degrade {ev.a}<->{ev.b}")
            elif original_kind == "link_up":
                # 'link_up' is usually one-shot; nothing to revert
                pass
            return

        if original_kind in NODE_KINDS and ev.node:
            if original_kind == "node_kill":
                self.store.node_revert(ev.node, ["down"])
                self.log(f"REVERT node_kill {ev.node}")
            elif original_kind == "power_cap":
                self.store.node_revert(ev.node, ["power_cap_w"])
                self.log(f"REVERT power_cap {ev.node}")
            elif original_kind == "thermal_derate":
                self.store.node_revert(ev.node, ["thermal_derate"])
                self.log(f"REVERT thermal_derate {ev.node}")
            elif original_kind == "clock_skew":
                self.store.node_revert(ev.node, ["clock_skew_ms"])
                self.log(f"REVERT clock_skew {ev.node}")
            elif original_kind == "packet_dup":
                self.store.node_revert(ev.node, ["packet_dup"])
                self.log(f"REVERT packet_dup {ev.node}")
            elif original_kind == "packet_reorder":
                self.store.node_revert(ev.node, ["packet_reorder"])
                self.log(f"REVERT packet_reorder {ev.node}")
            elif original_kind == "node_recover":
                # one-shot; nothing to revert
                pass
            return

        if original_kind in GROUP_KINDS:
            if original_kind == "zone_blackout":
                self._apply_zone_recover(ev)
            elif original_kind == "federation_partition":
                self._revert_federation_partition(ev)
            elif original_kind == "zone_recover":
                # recover is typically one-shot
                pass
            return

    def run(self, schedule: List[ChaosEvent], start_time_s: float = 0.0):
        if not schedule:
            self.log("No chaos events to run.")
            return

        t0 = time.time() - (start_time_s / max(0.001, self.speed))
        idx = 0
        n = len(schedule)

        self.log(f"Starting chaos with {n} events; speed x{self.speed}")
        while not self._stop.is_set() and idx < n:
            # Real time elapsed
            elapsed_real = time.time() - t0
            # Virtual time with acceleration
            vt_s = elapsed_real * self.speed
            ev = schedule[idx]

            if vt_s + 1e-6 >= ev.at_s:
                self.apply_event(ev)
                idx += 1
                continue

            # Sleep a bit before next check
            time.sleep(0.02)

        self.log("Chaos finished (or stopped).")

# ----------------------------- CLI -----------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Fabric Chaos Engine")
    ap.add_argument("--topology", type=str, default=str(Path(__file__).parent / "topology.yaml"),
                    help="Path to topology.yaml")
    ap.add_argument("--scenario", type=str, default=None, help="Scenario name to apply")
    ap.add_argument("--speed", type=float, default=1.0, help="Time acceleration factor (e.g., 20 for 20x)")
    ap.add_argument("--dt", type=str, default=None, help="DT observe endpoint (e.g., http://127.0.0.1:5055/observe)")
    ap.add_argument("--overrides", type=str, default=str(DEFAULT_OVERRIDES_PATH), help="Overrides JSON path")
    ap.add_argument("--nodes", type=str, default=str(DEFAULT_NODES_DIR), help="Directory containing node descriptors")
    ap.add_argument("--dry-run", action="store_true", help="Only print schedule, do not run")
    ap.add_argument("--run", action="store_true", help="Execute schedule")
    return ap

def pretty_event(ev: ChaosEvent) -> str:
    base = f"t={ev.at_s:7.2f}s kind={ev.kind}"
    if ev.end_time() is not None:
        base += f" dur={ev.duration_s:.2f}s"
    if ev.a and ev.b:
        base += f" link={ev.a}<->{ev.b}"
    if ev.node:
        base += f" node={ev.node}"
    if ev.label and ev.value:
        base += f" label={ev.label}:{ev.value}"
    return base

def main():
    ap = build_argparser()
    args = ap.parse_args()

    topo = load_topology(Path(args.topology))
    schedule = collect_chaos_events(topo, args.scenario)

    if args.dry_run or not args.run:
        print(f"[chaos] Loaded {len(schedule)} events")
        for ev in schedule:
            print("  " + pretty_event(ev))
        if not args.run:
            return

    store = OverridesStore(Path(args.overrides), dt_endpoint=args.dt)
    nodes_idx = load_nodes_index(Path(args.nodes))
    engine = ChaosEngine(store, speed=args.speed, verbose=True, nodes_index=nodes_idx)

    def handle_sig(sig, frame):
        engine.stop()
        print("\n[chaos] Stopping...")
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    engine.run(schedule)

if __name__ == "__main__":
    main()

