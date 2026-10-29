#!/usr/bin/env python3
"""Read-only inventory for a Gemini Scene Studio scene.

The report separates the dependency-valid active chain from historical runs.  It
never removes, renames, or rewrites scene data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STAGES = ("depth", "poses", "cloud", "tsdf", "train")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def load_runs(scene: Path) -> list[dict]:
    result = []
    for status_path in (scene / "runs").glob("*/status.json"):
        try:
            run = read_json(status_path)
        except (OSError, json.JSONDecodeError):
            continue
        run["_directory"] = status_path.parent
        run["_bytes"] = directory_size(status_path.parent)
        result.append(run)
    return sorted(result, key=lambda item: item.get("created_at", 0))


def active_runs(manifest: dict, runs: list[dict]) -> dict[str, dict]:
    """Mirror studio.core.active_runs so reports match the web application."""
    current: dict[str, dict] = {}
    for run in runs:
        if run.get("state") != "succeeded" or run.get("revision") != manifest.get("revision"):
            continue
        dependencies = run.get("dependencies", {})
        if all(current.get(stage, {}).get("id") == run_id for stage, run_id in dependencies.items()):
            current[run["stage"]] = run
            for stage in list(current):
                stage_dependencies = current[stage].get("dependencies", {})
                if any(current.get(key, {}).get("id") != value for key, value in stage_dependencies.items()):
                    del current[stage]
    return current


def build_report(scene: Path) -> dict:
    manifest = read_json(scene / "manifest.json")
    runs = load_runs(scene)
    active = active_runs(manifest, runs)
    active_ids = {run["id"] for run in active.values()}
    historical = [run for run in runs if run.get("id") not in active_ids]
    by_stage: dict[str, Counter] = defaultdict(Counter)
    bytes_by_stage = Counter()
    for run in runs:
        by_stage[run.get("stage", "unknown")][run.get("state", "unknown")] += 1
        bytes_by_stage[run.get("stage", "unknown")] += run["_bytes"]

    processing = manifest.get("processing", {})
    frame_rois = processing.get("frame_rois", {})
    active_summary = {}
    for stage in STAGES:
        run = active.get(stage)
        if not run:
            continue
        result_path = run["_directory"] / "result.json"
        active_summary[stage] = {
            "id": run["id"],
            "state": run["state"],
            "size_bytes": run["_bytes"],
            "dependencies": run.get("dependencies", {}),
            "result": read_json(result_path) if result_path.is_file() else None,
        }

    return {
        "scene": {
            "id": manifest.get("id", scene.name),
            "name": manifest.get("name"),
            "revision": manifest.get("revision"),
            "frames": len(manifest.get("frames", [])),
            "roi_frames": sum(1 for polygon in frame_rois.values() if len(polygon) >= 3),
            "depth_range_m": [processing.get("crop_min_depth"), processing.get("crop_max_depth")],
        },
        "storage": {
            "frames_bytes": directory_size(scene / "frames"),
            "runs_bytes": sum(run["_bytes"] for run in runs),
            "active_runs_bytes": sum(run["_bytes"] for run in active.values()),
            "historical_runs_bytes": sum(run["_bytes"] for run in historical),
        },
        "run_counts": {stage: dict(states) for stage, states in sorted(by_stage.items())},
        "run_bytes_by_stage": dict(bytes_by_stage),
        "active": active_summary,
        "historical_candidates": [
            {
                "id": run.get("id"),
                "stage": run.get("stage"),
                "state": run.get("state"),
                "revision": run.get("revision"),
                "size_bytes": run["_bytes"],
            }
            for run in sorted(historical, key=lambda item: item["_bytes"], reverse=True)
        ],
    }


def markdown(report: dict) -> str:
    scene = report["scene"]
    storage = report["storage"]
    lines = [
        f"# 场景清单：{scene['name']}",
        "",
        f"- 场景 ID：`{scene['id']}`；revision：`{scene['revision']}`",
        f"- 帧数：{scene['frames']}；已保存多边形 ROI：{scene['roi_frames']}/{scene['frames']}",
        f"- 深度范围：{scene['depth_range_m'][0]}–{scene['depth_range_m'][1]} m",
        f"- 原始帧：{human_size(storage['frames_bytes'])}；全部运行：{human_size(storage['runs_bytes'])}",
        f"- 当前有效链：{human_size(storage['active_runs_bytes'])}；历史运行候选：{human_size(storage['historical_runs_bytes'])}",
        "",
        "## 当前有效依赖链",
        "",
        "| 阶段 | run ID | 大小 | 依赖 |",
        "|---|---|---:|---|",
    ]
    for stage in STAGES:
        run = report["active"].get(stage)
        if run:
            deps = ", ".join(f"{key}:{value}" for key, value in run["dependencies"].items()) or "无"
            lines.append(f"| {stage} | `{run['id']}` | {human_size(run['size_bytes'])} | {deps} |")
        else:
            lines.append(f"| {stage} | — | — | — |")

    lines += ["", "## 运行历史统计", "", "| 阶段 | 状态与数量 | 占用 |", "|---|---|---:|"]
    for stage, states in report["run_counts"].items():
        state_text = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
        lines.append(f"| {stage} | {state_text} | {human_size(report['run_bytes_by_stage'].get(stage, 0))} |")

    lines += [
        "",
        "## 说明",
        "",
        "“历史运行候选”只表示它们不属于当前 revision 的依赖有效链，不等于应当立刻删除。",
        "归档或清理前应先确认不再需要算法对比、失败日志和旧产物。本脚本只读，不执行清理。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="scene directory containing manifest.json")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--output", type=Path, help="write report to a file instead of stdout")
    args = parser.parse_args()
    scene = args.scene.resolve()
    if not (scene / "manifest.json").is_file():
        parser.error(f"not a scene directory: {scene}")
    report = build_report(scene)
    output = json.dumps(report, ensure_ascii=False, indent=2) if args.json else markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
