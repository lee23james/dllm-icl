import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get_rollout_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "rollout_list" in data:
        rollout_list = data.get("rollout_list") or []
        return rollout_list if isinstance(rollout_list, list) else []
    if isinstance(data, list):
        return data
    return []


def _extract_layer_analysis(rollout_list: List[Dict[str, Any]]) -> List[List[Dict[str, float]]]:
    out: List[List[Dict[str, float]]] = []
    for step_item in rollout_list:
        la = step_item.get("layer_anaylse", None)
        if isinstance(la, list) and la:
            out.append(la)
    return out


def _normalize_boundary_index(nshot: int, query_position: int, position_mode: str) -> int:
    if not (0 <= query_position <= nshot):
        query_position = max(0, min(nshot, query_position))

    if position_mode == "boundary":
        return query_position
    if position_mode == "reverse":
        return nshot - query_position
    raise ValueError(f"Unknown position_mode={position_mode}. Use boundary or reverse.")


def _distance_to_boundary(example_idx: int, boundary_idx: int) -> int:
    # boundary_idx=b means boundary between example b and b+1 (1-based example index)
    return min(abs(example_idx - boundary_idx), abs(example_idx - (boundary_idx + 1)))


def collect_attention_records(
    task: str,
    nshot: int,
    max_samples: int,
    max_steps: Optional[int],
    position_mode: str,
    params_dir: str,
) -> Tuple[List[float], List[float], Dict[int, List[float]], Dict[str, int]]:
    neighbor_values: List[float] = []
    non_neighbor_values: List[float] = []
    dist_to_values: Dict[int, List[float]] = {d: [] for d in range(nshot)}

    stats = {
        "files_found": 0,
        "files_used": 0,
        "steps_used": 0,
        "layers_used": 0,
        "values_used": 0,
    }

    for pos in range(nshot + 1):
        boundary_idx = _normalize_boundary_index(
            nshot=nshot,
            query_position=pos,
            position_mode=position_mode,
        )
        for sample_idx in range(max_samples):
            json_path = os.path.join(params_dir, f"{task}_{pos}_{sample_idx}.json")
            if not os.path.exists(json_path):
                continue
            stats["files_found"] += 1

            data = _load_json(json_path)
            rollout_list = _safe_get_rollout_list(data)
            la_steps = _extract_layer_analysis(rollout_list)
            if not la_steps:
                continue

            if max_steps is not None:
                la_steps = la_steps[: max_steps]
            if not la_steps:
                continue
            stats["files_used"] += 1

            for step_layers in la_steps:
                if not isinstance(step_layers, list):
                    continue
                stats["steps_used"] += 1
                for layer_dict in step_layers:
                    if not isinstance(layer_dict, dict):
                        continue
                    stats["layers_used"] += 1
                    for ex_i in range(1, nshot + 1):
                        field = f"example{ex_i}_token"
                        value = layer_dict.get(field, None)
                        if value is None:
                            continue
                        try:
                            fv = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(fv):
                            continue

                        dist = _distance_to_boundary(example_idx=ex_i, boundary_idx=boundary_idx)
                        if dist < 0 or dist >= nshot:
                            continue
                        dist_to_values[dist].append(fv)
                        stats["values_used"] += 1
                        if dist == 0:
                            neighbor_values.append(fv)
                        else:
                            non_neighbor_values.append(fv)

    return neighbor_values, non_neighbor_values, dist_to_values, stats


def _mean_and_ci95(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, 1.96 * sem


def plot_neighbor_vs_nonneighbor(
    neighbor_values: List[float],
    non_neighbor_values: List[float],
    out_path: str,
):
    labels = ["Neighbor (dist = 0)", "Non-neighbor (dist >= 1)"]
    means = []
    cis = []
    counts = []

    for vals in [neighbor_values, non_neighbor_values]:
        mean, ci = _mean_and_ci95(vals)
        means.append(mean)
        cis.append(ci)
        counts.append(len(vals))

    plt.figure(figsize=(7.2, 5.6))
    sns.set_theme(style="whitegrid")
    colors = ["#2A9D8F", "#E76F51"]
    bars = plt.bar(
        range(2),
        means,
        yerr=cis,
        capsize=8,
        color=colors,
        edgecolor="#333333",
        linewidth=1.2,
        alpha=0.95,
    )

    plt.xticks(range(2), labels, fontsize=11)
    plt.ylabel("Attention Flow", fontsize=12)
    plt.title("Neighbor vs Non-neighbor Attention Flow", fontsize=15, pad=10)

    for idx, (bar, n) in enumerate(zip(bars, counts)):
        top = bar.get_height() + cis[idx]
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            top + 0.002,
            f"n={n:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distance_decay(
    dist_to_values: Dict[int, List[float]],
    out_path: str,
):
    dists = sorted(dist_to_values.keys())
    means = []
    cis = []
    counts = []
    for d in dists:
        mean, ci = _mean_and_ci95(dist_to_values[d])
        means.append(mean)
        cis.append(ci)
        counts.append(len(dist_to_values[d]))

    x = np.asarray(dists, dtype=np.float64)
    y = np.asarray(means, dtype=np.float64)
    e = np.asarray(cis, dtype=np.float64)

    plt.figure(figsize=(7.8, 5.8))
    sns.set_theme(style="whitegrid")
    plt.plot(
        x,
        y,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color="#264653",
        label="Mean attention flow",
    )
    plt.fill_between(x, y - e, y + e, color="#2A9D8F", alpha=0.2, label="95% CI")
    plt.xticks(dists, [str(d) for d in dists], fontsize=11)
    plt.xlabel("Relative distance to query boundary", fontsize=12)
    plt.ylabel("Attention Flow", fontsize=12)
    plt.title("Attention Flow Decay vs Relative Distance", fontsize=15, pad=10)
    plt.legend(fontsize=10, loc="upper right")

    for d, yi, n in zip(dists, y, counts):
        plt.text(d, yi + 0.002, f"n={n:,}", ha="center", va="bottom", fontsize=9, color="#444444")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_stats_json(
    out_path: str,
    neighbor_values: List[float],
    non_neighbor_values: List[float],
    dist_to_values: Dict[int, List[float]],
    stats: Dict[str, int],
):
    n_mean, n_ci = _mean_and_ci95(neighbor_values)
    f_mean, f_ci = _mean_and_ci95(non_neighbor_values)
    dist_summary = {}
    for d, vals in sorted(dist_to_values.items()):
        mean, ci = _mean_and_ci95(vals)
        dist_summary[str(d)] = {
            "count": len(vals),
            "mean": mean,
            "ci95": ci,
        }

    payload = {
        "global": {
            "neighbor": {"count": len(neighbor_values), "mean": n_mean, "ci95": n_ci},
            "non_neighbor": {"count": len(non_neighbor_values), "mean": f_mean, "ci95": f_ci},
        },
        "distance": dist_summary,
        "processing_stats": stats,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Plot quantitative attention figures from rollout_params JSONs.")
    parser.add_argument("--task", type=str, default="sudoku")
    parser.add_argument("--nshot", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--position_mode",
        type=str,
        default="reverse",
        choices=["reverse", "boundary"],
        help="How to interpret query_position in filenames.",
    )
    parser.add_argument(
        "--params_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "params" / "rollout_params"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "rollout_results" / "sudoku"),
    )
    args = parser.parse_args()

    neighbor_values, non_neighbor_values, dist_to_values, stats = collect_attention_records(
        task=args.task,
        nshot=args.nshot,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        position_mode=args.position_mode,
        params_dir=args.params_dir,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_path = out_dir / "attention_neighbor_vs_nonneighbor.png"
    decay_path = out_dir / "attention_distance_decay.png"
    stats_path = out_dir / "attention_quant_stats.json"

    plot_neighbor_vs_nonneighbor(
        neighbor_values=neighbor_values,
        non_neighbor_values=non_neighbor_values,
        out_path=str(bar_path),
    )
    plot_distance_decay(
        dist_to_values=dist_to_values,
        out_path=str(decay_path),
    )
    save_stats_json(
        out_path=str(stats_path),
        neighbor_values=neighbor_values,
        non_neighbor_values=non_neighbor_values,
        dist_to_values=dist_to_values,
        stats=stats,
    )

    print(f"Saved: {bar_path}")
    print(f"Saved: {decay_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
