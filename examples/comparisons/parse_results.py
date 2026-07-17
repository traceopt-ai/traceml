"""Parse raw study_out/ artifacts into a canonical results.json + results.md.

Deterministic. Reads whatever the driver produced and computes the honest
comparison metrics. Missing files degrade to null rather than crashing, so a
partial run still yields a partial table.

Usage:
  python parse_results.py --in <study_out_dir> --out-json results.json \
      --out-md results.md
"""

import argparse
import csv
import glob
import json
import os
import re


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _gpu_util_stats(csv_path):
    """Mean/median GPU-util% from an nvidia-smi util CSV (drop idle ramp)."""
    if not os.path.exists(csv_path):
        return {"mean": None, "p50": None, "n": 0}
    vals = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            m = re.search(r"(\d+)", row[1])
            if m:
                vals.append(int(m.group(1)))
    # Drop leading/trailing zero-util samples (process spin-up / teardown).
    while vals and vals[0] == 0:
        vals.pop(0)
    while vals and vals[-1] == 0:
        vals.pop()
    if not vals:
        return {"mean": None, "p50": None, "n": 0}
    s = sorted(vals)
    return {
        "mean": round(sum(vals) / len(vals), 1),
        "p50": s[len(s) // 2],
        "n": len(vals),
    }


def _wall_from_csv(indir, label):
    path = os.path.join(indir, "wall.csv")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0] == label:
                try:
                    return float(parts[2]), int(parts[4])
                except (IndexError, ValueError):
                    return None, None
    return None, None


def _sizes(indir):
    out = {}
    path = os.path.join(indir, "artifact_sizes.csv")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "config":
                try:
                    out[row[0]] = int(row[1])
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out-json", default="results.json")
    ap.add_argument("--out-md", default="results.md")
    args = ap.parse_args()
    d = args.indir

    sizes = _sizes(d)
    bare_m = _load_json(os.path.join(d, "bare_metrics.json")) or {}
    tp_m = _load_json(os.path.join(d, "torch_profiler_metrics.json")) or {}
    combo_m = _load_json(os.path.join(d, "combo_tp_metrics.json")) or {}
    cp_run = _load_json(os.path.join(d, "cprofile_run_metrics.json")) or {}
    fs = (
        _load_json(
            os.path.join(
                d, "traceml_logs", "traceml_baseline", "final_summary.json"
            )
        )
        or {}
    )

    bare_wall, _ = _wall_from_csv(d, "bare")
    traceml_wall, _ = _wall_from_csv(d, "traceml")
    cprofile_wall, _ = _wall_from_csv(d, "cprofile")

    bare_wall = bare_m.get("wall_s", bare_wall)
    bare_per_step = bare_m.get("per_step_ms")
    steps = bare_m.get("steps")

    # TraceML training duration (authoritative) from final_summary.
    tml_train = fs.get("duration_s")
    tml_verdict = (fs.get("primary_diagnosis") or {}).get("status")
    tml_gpu = (
        ((fs.get("system") or {}).get("global") or {}).get("average") or {}
    ).get("gpu_util_percent")

    # cProfile in-process wall (includes cProfile overhead).
    cp_wall = cp_run.get("wall_s", cprofile_wall)
    cp_per_step = cp_run.get("per_step_ms")

    def pct(a, b):
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / b * 100.0, 1)

    configs = {
        "bare": {
            "role": "reference (no tool)",
            "wall_s": round(bare_wall, 1) if bare_wall else None,
            "per_step_ms": round(bare_per_step, 1) if bare_per_step else None,
            "artifact_bytes": 0,
            "gpu_util": _gpu_util_stats(os.path.join(d, "util_bare.csv")),
        },
        "traceml": {
            "role": "always-on monitor",
            "wall_s": round(traceml_wall, 1) if traceml_wall else None,
            "training_duration_s": round(tml_train, 1) if tml_train else None,
            "verdict": tml_verdict,
            "gpu_util_reported": tml_gpu,
            "artifact_bytes": sizes.get("traceml"),
            "gpu_util": _gpu_util_stats(os.path.join(d, "util_traceml.csv")),
            "overhead_pct_vs_bare": pct(tml_train, bare_wall),
        },
        "torch_profiler": {
            "role": "one-shot windowed deep profile",
            "active_steps": tp_m.get("active_steps"),
            "collect_wall_s": (
                round(tp_m["collect_wall_s"], 1)
                if tp_m.get("collect_wall_s")
                else None
            ),
            "active_per_step_ms": (
                round(tp_m["active_per_step_ms"], 1)
                if tp_m.get("active_per_step_ms")
                else None
            ),
            "export_s": tp_m.get("export_s"),
            "trace_bytes": tp_m.get("trace_bytes"),
            "artifact_bytes": sizes.get("torch_profiler"),
            "gpu_util": _gpu_util_stats(
                os.path.join(d, "util_torch_profiler.csv")
            ),
            "perstep_overhead_pct_vs_bare": pct(
                tp_m.get("active_per_step_ms"), bare_per_step
            ),
        },
        "cprofile": {
            "role": "whole-run Python function profile (CPU only)",
            "wall_s": round(cp_wall, 1) if cp_wall else None,
            "per_step_ms": round(cp_per_step, 1) if cp_per_step else None,
            "artifact_bytes": sizes.get("cprofile"),
            "gpu_util": _gpu_util_stats(os.path.join(d, "util_cprofile.csv")),
            "overhead_pct_vs_bare": pct(cp_wall, bare_wall),
            "sees_gpu_idle": False,
        },
        "combo": {
            "role": "torch.profiler under cProfile (profiler+cProfile)",
            "active_steps": combo_m.get("active_steps"),
            "collect_wall_s": (
                round(combo_m["collect_wall_s"], 1)
                if combo_m.get("collect_wall_s")
                else None
            ),
            "trace_bytes": combo_m.get("trace_bytes"),
            "artifact_bytes": sizes.get("combo"),
            "gpu_util": _gpu_util_stats(os.path.join(d, "util_combo.csv")),
        },
    }

    result = {
        "meta": {
            "steps": steps,
            "workload": (
                "resnet18 imagenette baseline num_workers=0 (input-bound)"
            ),
            "env_manifest": "env_manifest.txt",
        },
        "configs": configs,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Markdown table
    def mb(b):
        return f"{b/1e6:.1f} MB" if isinstance(b, (int, float)) else "-"

    lines = [
        "| Config | Role | Wall / collect (s) | Overhead vs bare | "
        "Data written | GPU util (nvidia-smi) | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    b = configs["bare"]
    lines.append(
        f"| bare | reference | {b['wall_s']} | 0% (ref) | 0 | "
        f"{b['gpu_util']['mean']}% | - |"
    )
    t = configs["traceml"]
    lines.append(
        f"| TraceML | always-on | {t['training_duration_s']} | "
        f"{t['overhead_pct_vs_bare']}% | {mb(t['artifact_bytes'])} | "
        f"{t['gpu_util']['mean']}% | {t['verdict']} |"
    )
    p = configs["torch_profiler"]
    lines.append(
        f"| torch.profiler | one-shot ({p['active_steps']} steps) | "
        f"{p['collect_wall_s']} | {p['perstep_overhead_pct_vs_bare']}%/step | "
        f"{mb(p['artifact_bytes'])} | {p['gpu_util']['mean']}% | "
        "raw trace (must read) |"
    )
    c = configs["cprofile"]
    lines.append(
        f"| cProfile | whole-run CPU | {c['wall_s']} | "
        f"{c['overhead_pct_vs_bare']}% | {mb(c['artifact_bytes'])} | "
        f"{c['gpu_util']['mean']}% | CPU fns (GPU-blind) |"
    )
    cm = configs["combo"]
    lines.append(
        f"| combo | profiler+cProfile | {cm['collect_wall_s']} | - | "
        f"{mb(cm['artifact_bytes'])} | {cm['gpu_util']['mean']}% | - |"
    )
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(result, indent=2))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
