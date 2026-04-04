import argparse
import json
import re
from pathlib import Path


TASKS = ["gsm8k", "sudoku", "countdown", "math500", "mbpp"]
PLACEMENTS = ["first", "random"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_latest_file(directory: Path, pattern: str):
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def parse_log_timing(log_path: Path):
    text = read_text(log_path)
    total_match = re.search(r"Total Inference Time\s*:\s*([0-9.]+)\s*s", text)
    count_match = re.search(r"Total Generation Count\s*:\s*([0-9]+)", text)
    avg_match = re.search(r"Avg Latency per Query\s*:\s*([0-9.]+)\s*s/query", text)
    return {
        "log_path": str(log_path),
        "total_inference_time_sec": float(total_match.group(1)) if total_match else None,
        "generation_count": int(count_match.group(1)) if count_match else None,
        "avg_latency_sec": float(avg_match.group(1)) if avg_match else None,
    }


def parse_accuracy_file(path: Path):
    text = read_text(path)
    match = re.search(r"Accuracy:\s*([0-9.]+)%?\s*\((\d+)/(\d+)\)", text)
    if not match:
        match = re.search(r"Final Accuracy:?\s*([0-9.]+)\s*\((\d+)/(\d+)\)", text)
    if not match:
        match = re.search(r"Final Accuracy:?\s*([0-9.]+)\((\d+)/(\d+)\)", text)
    if not match:
        raise ValueError(f"Unable to parse accuracy from {path}")
    value = float(match.group(1))
    correct = int(match.group(2))
    total = int(match.group(3))
    if "%" in match.group(0):
        accuracy = value / 100.0
    else:
        accuracy = value
    return {
        "accuracy_file": str(path),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def find_part_accuracy_file(part_root: Path, placement: str, task: str):
    shot_dir = next(part_root.glob("shot_*"), None)
    if shot_dir is None:
        raise FileNotFoundError(f"No shot_* dir under {part_root}")

    if task == "mbpp":
        position_dir = shot_dir / "accuracy" / f"position_{'random' if placement == 'random' else '0'}"
        file_path = find_latest_file(position_dir, "accuracy_*.txt")
    else:
        position_dir = shot_dir / f"position_{'random' if placement == 'random' else '0'}"
        file_path = find_latest_file(position_dir, "*.txt")

    if file_path is None:
        raise FileNotFoundError(f"No accuracy file found under {position_dir}")
    return file_path


def summarize_run(result_root: Path, task: str, placement: str):
    base = result_root / placement / task
    meta_path = base / "run_meta.json"
    meta = json.loads(read_text(meta_path))

    log_a = parse_log_timing((PROJECT_ROOT / meta["log_part_a"]).resolve())
    log_b = parse_log_timing((PROJECT_ROOT / meta["log_part_b"]).resolve())

    acc_a_path = find_part_accuracy_file((PROJECT_ROOT / meta["result_part_a"]).resolve(), placement, task)
    acc_b_path = find_part_accuracy_file((PROJECT_ROOT / meta["result_part_b"]).resolve(), placement, task)
    acc_a = parse_accuracy_file(acc_a_path)
    acc_b = parse_accuracy_file(acc_b_path)

    total_correct = acc_a["correct"] + acc_b["correct"]
    total_samples = acc_a["total"] + acc_b["total"]

    total_inference = None
    total_count = None
    avg_latency = None
    if log_a["total_inference_time_sec"] is not None and log_b["total_inference_time_sec"] is not None:
        total_inference = log_a["total_inference_time_sec"] + log_b["total_inference_time_sec"]
    if log_a["generation_count"] is not None and log_b["generation_count"] is not None:
        total_count = log_a["generation_count"] + log_b["generation_count"]
    if total_inference is not None and total_count:
        avg_latency = total_inference / total_count

    return {
        "task": task,
        "placement": placement,
        "correct": total_correct,
        "total": total_samples,
        "accuracy": total_correct / total_samples if total_samples else None,
        "total_inference_time_sec": total_inference,
        "total_generation_count": total_count,
        "avg_latency_sec": avg_latency,
        "wall_seconds": meta.get("wall_seconds"),
        "return_code_part_a": meta.get("return_code_part_a"),
        "return_code_part_b": meta.get("return_code_part_b"),
        "part_a": {
            "accuracy": acc_a,
            "timing": log_a,
        },
        "part_b": {
            "accuracy": acc_b,
            "timing": log_b,
        },
    }


def format_md_table(rows):
    header = "| task | placement | correct/total | accuracy | total_inference_s | avg_latency_s | wall_s |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        acc = "-" if row["accuracy"] is None else f"{row['accuracy'] * 100:.2f}%"
        total_inf = "-" if row["total_inference_time_sec"] is None else f"{row['total_inference_time_sec']:.2f}"
        avg_lat = "-" if row["avg_latency_sec"] is None else f"{row['avg_latency_sec']:.4f}"
        wall = "-" if row["wall_seconds"] is None else f"{row['wall_seconds']}"
        lines.append(
            f"| {row['task']} | {row['placement']} | {row['correct']}/{row['total']} | {acc} | {total_inf} | {avg_lat} | {wall} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="/hy-tmp/dllm-icl/results/front_random_split_2gpu")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    rows = []
    for task in TASKS:
        for placement in PLACEMENTS:
            rows.append(summarize_run(result_root, task, placement))

    md = format_md_table(rows)
    print(md, end="")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
