import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get_rollout_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "rollout_list" in data:
        rl = data.get("rollout_list") or []
        return rl if isinstance(rl, list) else []
    if isinstance(data, list):
        return data
    return []


def _extract_layer_analysis(rollout_list: List[Dict[str, Any]]) -> List[List[Dict[str, float]]]:
    """
    Returns: list over steps; each step is list over layers; each layer is dict[field -> value]
    Expected keys (from existing pipeline): answer_token, question_token, example{1..N}_token, other_token
    """
    out: List[List[Dict[str, float]]] = []
    for step_item in rollout_list:
        la = step_item.get("layer_anaylse", None)
        if not isinstance(la, list):
            continue
        # each element is layer_dict
        if not la:
            continue
        out.append(la)
    return out


def _example_fields(nshot: int) -> List[str]:
    return [f"example{i}_token" for i in range(1, nshot + 1)]


def _distance_order_for_position(nshot: int, query_position: int) -> List[int]:
    """
    Map distance rank -> example index (1..nshot).

    Assumption (matches how people talk about query insertion):
    - We have examples 1..nshot in order.
    - query_position is an insertion boundary between examples:
        0 means query is at the end (after example nshot),
        nshot means query is at the beginning (before example 1),
        otherwise query is between example (nshot - query_position) and the next one in some codebases.

    Because different repos invert semantics, we make this configurable via --position_mode.
    This helper is used after we normalize query_position into "boundary index b in [0..nshot]"
    where b means "insert before example b+1", and b==nshot means "after example nshot".

    Then nearest examples are b and b+1 (if exist), then expand outward.
    """
    b = query_position  # already normalized boundary index in [0..nshot]
    # left example index = b (1-based), right example index = b+1 (1-based)
    left = b
    right = b + 1
    chosen: List[int] = []
    step = 0
    while len(chosen) < nshot:
        li = left - step
        ri = right + step
        if li >= 1 and li <= nshot and li not in chosen:
            chosen.append(li)
        if len(chosen) >= nshot:
            break
        if ri >= 1 and ri <= nshot and ri not in chosen:
            chosen.append(ri)
        step += 1
    return chosen


def _normalize_boundary_index(nshot: int, query_position: int, position_mode: str) -> int:
    """
    Normalize repo-specific query_position semantics to boundary index b in [0..nshot]:
      b == 0   => before example 1 (query at very beginning)
      b == nshot => after example nshot (query at very end)
    """
    if not (0 <= query_position <= nshot):
        # clamp just in case
        query_position = max(0, min(nshot, query_position))

    if position_mode == "boundary":
        # user asserts query_position already means boundary index
        return query_position

    if position_mode == "reverse":
        # some pipelines define 0 as end, nshot as start (reversed)
        return nshot - query_position

    raise ValueError(f"Unknown position_mode={position_mode}. Use boundary or reverse.")


def build_recency_heatmap(
    task: str,
    nshot: int,
    max_samples: int,
    max_steps: Optional[int],
    position_mode: str,
    params_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns heatmap of shape [n_layers, nshot]:
      rows: layer index (0..n_layers-1)
      cols: distance rank d (0..nshot-1), where d=0 is nearest example to query, d increases outward
      values: average contribution over (positions, samples, steps)
    """
    n_layers = 32  # matches current plotting code assumption
    accum = np.zeros((n_layers, nshot), dtype=np.float64)
    count = np.zeros((n_layers, nshot), dtype=np.float64)

    for pos in range(nshot + 1):
        b = _normalize_boundary_index(nshot=nshot, query_position=pos, position_mode=position_mode)
        dist_order = _distance_order_for_position(nshot=nshot, query_position=b)  # list of example indices 1..nshot
        dist_fields = [f"example{i}_token" for i in dist_order]

        for sample_idx in range(max_samples):
            json_path = os.path.join(params_dir, f"{task}_{pos}_{sample_idx}.json")
            if not os.path.exists(json_path):
                continue

            data = _load_json(json_path)
            rollout_list = _safe_get_rollout_list(data)
            la_steps = _extract_layer_analysis(rollout_list)
            if not la_steps:
                continue

            if max_steps is not None:
                la_steps = la_steps[: max_steps]

            for step_layers in la_steps:
                if not isinstance(step_layers, list) or len(step_layers) != n_layers:
                    continue
                for layer_idx, layer_dict in enumerate(step_layers):
                    if not isinstance(layer_dict, dict):
                        continue
                    for d, field in enumerate(dist_fields):
                        v = layer_dict.get(field, None)
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(fv):
                            accum[layer_idx, d] += fv
                            count[layer_idx, d] += 1.0

    heat = np.divide(accum, np.maximum(count, 1.0))
    return heat, count


def save_heatmap_png(heat: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10.5, 6.0))
    ax = plt.gca()
    try:
        import seaborn as sns

        ax = sns.heatmap(
            heat,
            cmap="YlGnBu",
            cbar=True,
            square=False,
        )
    except ModuleNotFoundError:
        im = ax.imshow(heat, cmap="YlGnBu", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Example distance rank (0=nearest to query)")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_pos_vs_example_heatmap(
    task: str,
    nshot: int,
    max_samples: int,
    max_steps: Optional[int],
    position_mode: str,
    params_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns heatmap of shape [(nshot+1), nshot]:
      rows: query insertion position pos (0..nshot) as used in filenames
      cols: example index i (1..nshot)
      values: average contribution over layers + steps + samples

    Note:
      We normalize the raw query_position index from filenames into a
      boundary index b in [0..nshot] via _normalize_boundary_index and
      aggregate by this b, so that:
        b == 0   => before example 1 (query at very beginning)
        b == nshot => after example nshot (query at very end)
    """
    n_layers = 32
    accum = np.zeros((nshot + 1, nshot), dtype=np.float64)
    count = np.zeros((nshot + 1, nshot), dtype=np.float64)

    _ = position_mode

    for pos in range(nshot + 1):
        b = _normalize_boundary_index(
            nshot=nshot,
            query_position=pos,
            position_mode=position_mode,
        )

        for sample_idx in range(max_samples):
            json_path = os.path.join(params_dir, f"{task}_{pos}_{sample_idx}.json")
            if not os.path.exists(json_path):
                continue

            data = _load_json(json_path)
            rollout_list = _safe_get_rollout_list(data)
            la_steps = _extract_layer_analysis(rollout_list)
            if not la_steps:
                continue
            if max_steps is not None:
                la_steps = la_steps[: max_steps]

            for step_layers in la_steps:
                if not isinstance(step_layers, list) or len(step_layers) != n_layers:
                    continue
                for layer_dict in step_layers:
                    if not isinstance(layer_dict, dict):
                        continue
                    for ex_i in range(1, nshot + 1):
                        field = f"example{ex_i}_token"
                        v = layer_dict.get(field, None)
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(fv):
                            accum[b, ex_i - 1] += fv
                            count[b, ex_i - 1] += 1.0

    heat = np.divide(accum, np.maximum(count, 1.0))
    return heat, count


def _default_table_prefix(out_path: str) -> Path:
    out = Path(out_path)
    return out.with_suffix("")


def save_matrix_csv(
    heat: np.ndarray,
    out_path: str,
    mode: str,
    nshot: int,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if mode == "pos_vs_example":
            writer.writerow(["query_boundary"] + [f"example{i}" for i in range(1, nshot + 1)])
            for boundary_idx in range(nshot + 1):
                writer.writerow([boundary_idx] + [float(v) for v in heat[boundary_idx]])
            return

        writer.writerow(["layer"] + [f"distance_rank_{d}" for d in range(nshot)])
        for layer_idx in range(heat.shape[0]):
            writer.writerow([layer_idx] + [float(v) for v in heat[layer_idx]])


def save_long_csv(
    heat: np.ndarray,
    count: np.ndarray,
    out_path: str,
    mode: str,
    task: str,
    position_mode: str,
    nshot: int,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if mode == "pos_vs_example":
            writer.writerow(
                [
                    "task",
                    "mode",
                    "position_mode",
                    "query_boundary",
                    "example_index",
                    "attention_flow",
                    "count",
                ]
            )
            for boundary_idx in range(nshot + 1):
                for example_idx in range(1, nshot + 1):
                    writer.writerow(
                        [
                            task,
                            mode,
                            position_mode,
                            boundary_idx,
                            example_idx,
                            float(heat[boundary_idx, example_idx - 1]),
                            int(count[boundary_idx, example_idx - 1]),
                        ]
                    )
            return

        writer.writerow(
            [
                "task",
                "mode",
                "position_mode",
                "layer",
                "distance_rank",
                "attention_flow",
                "count",
            ]
        )
        for layer_idx in range(heat.shape[0]):
            for distance_rank in range(nshot):
                writer.writerow(
                    [
                        task,
                        mode,
                        position_mode,
                        layer_idx,
                        distance_rank,
                        float(heat[layer_idx, distance_rank]),
                        int(count[layer_idx, distance_rank]),
                    ]
                )


def save_pos_vs_example_png(
    heat: np.ndarray,
    out_path: str,
    title: str,
    nshot: int,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.5, 5.8))
    ax = plt.gca()
    try:
        import seaborn as sns

        ax = sns.heatmap(
            heat,
            cmap="YlGnBu_r",
            cbar=True,
            square=False,
            xticklabels=[rf"$e_{i}$" for i in range(1, nshot + 1)],
            yticklabels=[rf"$p={p}$" for p in range(0, nshot + 1)],
        )
        cbar = ax.collections[0].colorbar
    except ModuleNotFoundError:
        im = ax.imshow(heat, cmap="YlGnBu_r", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(nshot))
        ax.set_xticklabels([rf"$e_{i}$" for i in range(1, nshot + 1)])
        ax.set_yticks(range(nshot + 1))
        ax.set_yticklabels([rf"$p={p}$" for p in range(0, nshot + 1)])
        cbar = plt.colorbar(im, ax=ax)
    ax.set_title("")
    ax.set_xlabel("In-Context Examples")
    ax.set_ylabel("Query insertion boundary")
    cbar.set_label("Attention Flow")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Plot recency effect heatmap from rollout_params JSONs")
    p.add_argument("--task", type=str, required=True, help="task name used in rollout_params filenames, e.g. sudoku")
    p.add_argument("--nshot", type=int, required=True, help="number of examples (shots)")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=None, help="optional cap on steps per sample (for speed)")
    p.add_argument(
        "--mode",
        type=str,
        default="pos_vs_example",
        choices=["pos_vs_example", "layer_vs_dist"],
        help="pos_vs_example: Y=pos, X=example; layer_vs_dist: Y=layer, X=distance-rank",
    )
    p.add_argument(
        "--position_mode",
        type=str,
        default="boundary",
        choices=["reverse", "boundary"],
        help="how to interpret query_position index in filenames: reverse means 0=end,nshot=start; boundary means 0=start,nshot=end",
    )
    p.add_argument(
        "--params_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "params" / "rollout_params"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "rollout_results" / "recency_heatmap.png"),
    )
    p.add_argument(
        "--save_tables",
        action="store_true",
        help="also export the aggregated heatmap as CSV tables",
    )
    p.add_argument(
        "--table_prefix",
        type=str,
        default=None,
        help="output prefix for CSV tables; defaults to --out without file suffix",
    )
    args = p.parse_args()

    if args.mode == "layer_vs_dist":
        heat, count = build_recency_heatmap(
            task=args.task,
            nshot=args.nshot,
            max_samples=args.max_samples,
            max_steps=args.max_steps,
            position_mode=args.position_mode,
            params_dir=args.params_dir,
        )
        save_heatmap_png(
            heat=heat,
            out_path=args.out,
            title="Recency heatmap",
        )
    else:
        heat, count = build_pos_vs_example_heatmap(
            task=args.task,
            nshot=args.nshot,
            max_samples=args.max_samples,
            max_steps=args.max_steps,
            position_mode=args.position_mode,
            params_dir=args.params_dir,
        )
        save_pos_vs_example_png(
            heat=heat,
            out_path=args.out,
            title="Recency heatmap",
            nshot=args.nshot,
        )

    print(f"Saved: {args.out}")
    if args.save_tables:
        table_prefix = Path(args.table_prefix) if args.table_prefix else _default_table_prefix(args.out)
        matrix_csv_path = f"{table_prefix}.matrix.csv"
        long_csv_path = f"{table_prefix}.long.csv"
        save_matrix_csv(
            heat=heat,
            out_path=matrix_csv_path,
            mode=args.mode,
            nshot=args.nshot,
        )
        save_long_csv(
            heat=heat,
            count=count,
            out_path=long_csv_path,
            mode=args.mode,
            task=args.task,
            position_mode=args.position_mode,
            nshot=args.nshot,
        )
        print(f"Saved: {matrix_csv_path}")
        print(f"Saved: {long_csv_path}")


if __name__ == "__main__":
    main()
