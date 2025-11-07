#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fabric_docker/launch_fabric.py — spin up lightweight containers that mirror nodes/*.yaml

Usage
-----
python3 -m fabric_docker.launch_fabric \
  --nodes nodes \
  --network fabric-net \
  --image alpine:3.20 \
  --prefix fab- \
  --tc none \
  --force-arch   # (optional) force Docker platform to node.arch even if host supports it

Notes
-----
- Automatically attempts cross-architecture launches when binfmt handlers are present, and
  warns when emulation support is missing.
- Only approximates CPU/mem limits. GPUs are *not* plumbed here (future work).
- Traffic shaping:
    --tc none         : no shaping (default)
    --tc container    : best-effort: install iproute2 (if Alpine) and shape egress on eth0
  Rate selection:
    • If topology.yaml has links touching the node, we set rate = min(all peer link speeds) Gbps
    • Else if topology.defaults.egress_gbps exists, we use it
    • Else unlimited (no shaping)
"""

from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    import docker  # pip install docker
except Exception as e:
    print("FATAL: python 'docker' SDK not installed. pip install docker", file=sys.stderr)
    raise

HOST_ARCH = platform.machine().lower()  # e.g. x86_64, aarch64, riscv64
ARCH_MAP = {
    "x86_64": ["x86_64", "amd64"],
    "amd64": ["x86_64", "amd64"],
    "aarch64": ["arm64", "aarch64"],
    "arm64": ["arm64", "aarch64"],
    "riscv64": ["riscv64"],
}
PLATFORM_MAP = {
    "x86_64": "linux/amd64",
    "amd64": "linux/amd64",
    "aarch64": "linux/arm64",
    "arm64": "linux/arm64",
    "riscv64": "linux/riscv64",
}
BINFMT_ENTRY = {
    "linux/arm64": "qemu-aarch64",
    "linux/riscv64": "qemu-riscv64",
}

def load_yaml(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_float(x, d=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d

def host_supports(arch: str) -> bool:
    a = arch.lower()
    host_list = ARCH_MAP.get(HOST_ARCH, [HOST_ARCH])
    return a in host_list


def binfmt_ready_for(platform_tag: str) -> bool:
    entry = BINFMT_ENTRY.get(platform_tag)
    if not entry:
        return True  # nothing special needed (either native or not tracked)
    root = Path("/proc/sys/fs/binfmt_misc")
    if not root.exists():
        return False
    return (root / entry).exists()

@dataclass
class NodeSpec:
    name: str
    arch: str
    klass: str
    cpu_cores: float
    mem_gb: float
    formats: List[str]
    labels: Dict[str, str]

def node_from_yaml(obj: Dict[str, Any]) -> NodeSpec:
    res = obj.get("resources") or {}
    return NodeSpec(
        name=str(obj.get("name")),
        arch=str(obj.get("arch") or "x86_64"),
        klass=str(obj.get("class") or ""),
        cpu_cores=safe_float(res.get("cpu_cores"), 1.0),
        mem_gb=safe_float(res.get("mem_gb"), 1.0),
        formats=list(obj.get("formats_supported") or []),
        labels=dict(obj.get("labels") or {}),
    )

def find_rate_gbps(topology: Optional[Dict[str, Any]], node_name: str) -> Optional[float]:
    if not topology:
        return None
    # explicit links first
    links = topology.get("links") or []
    speeds = []
    for l in links:
        if l.get("a") == node_name or l.get("b") == node_name:
            eff = l.get("effective") or {}
            sp = eff.get("speed_gbps", l.get("speed_gbps"))
            if sp is not None:
                try:
                    speeds.append(float(sp))
                except Exception:
                    pass
    if speeds:
        return max(0.05, min(speeds))  # pick the tightest link (min), clamp >=50mbps
    # defaults
    defaults = topology.get("defaults") or {}
    d = defaults.get("egress_gbps")
    try:
        return float(d) if d is not None else None
    except Exception:
        return None

def ensure_network(client, name: str):
    for n in client.networks.list(names=[name]):
        return n
    return client.networks.create(name, driver="bridge", check_duplicate=True)

def build_container_spec(
    ns: NodeSpec, image: str, prefix: str, platform: Optional[str] = None
) -> Dict[str, Any]:
    # CPU limits: use cpus -> docker param (requires daemon v20+), memory in bytes
    mem_bytes = int(max(1, ns.mem_gb) * (1024**3))
    labels = {
        "fabric.name": ns.name,
        "fabric.arch": ns.arch,
        "fabric.class": ns.klass,
        "fabric.formats": ",".join(ns.formats),
    }
    for k, v in ns.labels.items():
        labels[f"fabric.label.{k}"] = str(v)

    spec: Dict[str, Any] = {
        "name": f"{prefix}{ns.name}",
        "image": image,
        "detach": True,
        "stdin_open": False,
        "tty": False,
        "labels": labels,
        "hostname": ns.name,
        "environment": {
            "FABRIC_NODE_NAME": ns.name,
            "FABRIC_NODE_ARCH": ns.arch,
            "FABRIC_NODE_CLASS": ns.klass,
        },
        "cpu_period": 100000,  # docker uses period/quota
        "cpu_quota": int(max(1.0, ns.cpu_cores) * 100000),
        "mem_limit": mem_bytes,
        "cap_add": ["NET_ADMIN"],  # needed if we do tc inside the container
        "command": ["sh", "-c", "sleep infinity"],
    }

    if platform:
        spec["platform"] = platform

    return spec

def ensure_image(client, image: str):
    try:
        client.images.get(image)
        return
    except docker.errors.ImageNotFound:
        pass
    print(f"[pull] {image} ...")
    client.images.pull(image)

def install_tc_if_needed(client, container):
    # attempt Alpine detection, then install iproute2
    try:
        rc, out = container.exec_run("sh -lc 'tc -V >/dev/null 2>&1 || echo NO_TC'")
        if rc == 0 and (out or b"") != b"NO_TC\n":
            return True
        # Try Alpine
        container.exec_run("sh -lc 'cat /etc/alpine-release >/dev/null 2>&1 || echo NO_APK'")
        # install iproute2
        print(f"[tc] installing iproute2 within {container.name} ...")
        container.exec_run("sh -lc 'apk update && apk add --no-cache iproute2'", user="root")
        # verify
        rc2, _ = container.exec_run("sh -lc 'tc -V >/dev/null 2>&1'")
        return rc2 == 0
    except Exception as e:
        print(f"[tc] WARN: could not install tc in {container.name}: {e}")
        return False

def apply_tc_rate(container, rate_gbps: float):
    """Apply outbound rate limit (egress) on eth0 using tbf; add basic netem knobs if needed."""
    mbit = max(10.0, rate_gbps * 1000.0)  # convert Gbps → Mbit/s (min 10mbit)
    burst = int(32 * 1024)  # bytes (small but ok)
    latency_ms = 50
    cmds = [
        "tc qdisc del dev eth0 root || true",
        f"tc qdisc add dev eth0 root tbf rate {mbit:.0f}mbit burst {burst} latency {latency_ms}ms",
    ]
    for c in cmds:
        container.exec_run(f"sh -lc '{c}'", user="root")

def main():
    ap = argparse.ArgumentParser(description="Launch Fabric containers from nodes/*.yaml")
    ap.add_argument("--nodes", default="nodes", help="Directory with node YAMLs")
    ap.add_argument("--topology", default="sim/topology.yaml", help="Topology YAML (for default egress shaping)")
    ap.add_argument("--network", default="fabric-net", help="Docker bridge network name")
    ap.add_argument("--image", default="alpine:3.20", help="Container image to run")
    ap.add_argument("--prefix", default="fab-", help="Name prefix for containers")
    ap.add_argument("--tc", choices=["none", "container"], default="none", help="Traffic shaping mode")
    ap.add_argument("--force-arch", action="store_true", help="Ignore host vs node.arch mismatch")
    ap.add_argument("--out", default="fabric_docker/containers.json", help="Write container mapping JSON here")
    args = ap.parse_args()

    nodes_dir = Path(args.nodes)
    if not nodes_dir.exists():
        print(f"error: nodes dir not found: {nodes_dir}", file=sys.stderr)
        sys.exit(2)

    topology = None
    topo_path = Path(args.topology)
    if topo_path.exists():
        try:
            topology = load_yaml(topo_path)
        except Exception as e:
            print(f"[topology] WARN: failed to load {topo_path}: {e}")

    client = docker.from_env()

    # ensure network & image
    net = ensure_network(client, args.network)
    ensure_image(client, args.image)

    # collect node specs
    node_specs: List[NodeSpec] = []
    for f in sorted(nodes_dir.glob("*.yaml")):
        try:
            obj = load_yaml(f)
            ns = node_from_yaml(obj)
            node_specs.append(ns)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")

    created = []
    skipped = []
    missing_binfmt: Set[str] = set()
    host_platform = PLATFORM_MAP.get(HOST_ARCH, f"linux/{HOST_ARCH}")

    for ns in node_specs:
        target_platform = PLATFORM_MAP.get(ns.arch.lower())
        host_can_run = host_supports(ns.arch)
        needs_emulation = bool(target_platform) and target_platform != host_platform

        platform_arg: Optional[str] = None

        if host_can_run and not args.force_arch:
            # Native execution – nothing special to do.
            pass
        elif target_platform:
            platform_arg = target_platform
            if needs_emulation and not binfmt_ready_for(platform_arg):
                missing_binfmt.add(platform_arg)
                skipped.append((
                    ns.name,
                    (
                        "binfmt handler missing for "
                        f"{platform_arg} (install via 'docker run --privileged --rm "
                        "tonistiigi/binfmt --install all')"
                    ),
                ))
                continue
        else:
            skipped.append((
                ns.name,
                f"arch mismatch host={HOST_ARCH} node={ns.arch} (no platform mapping)",
            ))
            continue

        if not host_can_run and platform_arg is None:
            skipped.append((
                ns.name,
                f"arch mismatch host={HOST_ARCH} node={ns.arch} (use --force-arch to ignore)",
            ))
            continue

        spec = build_container_spec(ns, args.image, args.prefix, platform=platform_arg)

        # create or reuse
        cname = spec["name"]
        cont = None
        for c in client.containers.list(all=True, filters={"name": f"^{cname}$"}):
            cont = c
            break
        if cont is None:
            try:
                print(f"[create] {cname}")
                cont = client.containers.create(**spec)
            except docker.errors.APIError as e:
                err = e.explanation or str(e)
                skipped.append((ns.name, f"failed to create on {platform_arg or host_platform}: {err}"))
                if needs_emulation and platform_arg:
                    missing_binfmt.add(platform_arg)
                continue
        # connect to network if not connected
        try:
            net.reload()
            containers = (net.attrs.get("Containers") or {})
            attached_ids = [c["Name"].lstrip("/") for c in containers.values()]
            if cname not in attached_ids:
                net.connect(cont)
        except Exception as e:
            print(f"[net] WARN: could not attach {cname} to {args.network}: {e}")

        # start
        if cont.status != "running":
            print(f"[start] {cname}")
            try:
                cont.start()
            except docker.errors.APIError as e:
                err = e.explanation or str(e)
                skipped.append((ns.name, f"failed to start on {platform_arg or host_platform}: {err}"))
                try:
                    cont.remove(force=True)
                except Exception:
                    pass
                # don't treat as created entry
                continue
            # give it a moment to get eth0
            time.sleep(0.3)

        # traffic shaping?
        if args.tc == "container":
            rate = find_rate_gbps(topology, ns.name)
            if rate:
                if install_tc_if_needed(client, cont):
                    print(f"[tc] {cname}: set egress ≈ {rate:.2f} Gbps")
                    try:
                        apply_tc_rate(cont, rate)
                    except Exception as e:
                        print(f"[tc] WARN: failed tc on {cname}: {e}")
                else:
                    print(f"[tc] WARN: no 'tc' in {cname}; skipping shaping")

        created.append({
            "name": ns.name,
            "container": cname,
            "arch": ns.arch,
            "class": ns.klass,
            "labels": ns.labels,
            "formats": ns.formats,
            "network": args.network,
            "platform": platform_arg or host_platform,
        })

    # write mapping
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"created": created, "skipped": skipped}, indent=2), encoding="utf-8")

    print(f"\nDone. Containers: {len(created)}  Skipped: {len(skipped)}")
    if skipped:
        for n, r in skipped:
            print(f" - {n}: {r}")
        arch_skips = [1 for _, reason in skipped if "arch mismatch" in reason]
        if arch_skips and not args.force_arch:
            print("[hint] To run these anyway, re-launch with --force-arch (requires binfmt/qemu emulation).")
            print("       Install handlers once with: docker run --privileged --rm tonistiigi/binfmt --install all")

    if missing_binfmt:
        plats = ", ".join(sorted(missing_binfmt))
        print(f"[hint] binfmt entries for {plats} were not detected on this host.")
        print("       Install them with: docker run --privileged --rm tonistiigi/binfmt --install all")
    print(f"Map written → {outp}")

if __name__ == "__main__":
    main()

