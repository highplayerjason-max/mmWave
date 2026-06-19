import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def to_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def bin_index(value: float, low: float, high: float, bins: int) -> int:
    if value <= low:
        return 0
    if value >= high:
        return bins - 1
    return int((value - low) / (high - low) * bins)


def summarize_points(points: list[dict], range_bins: int, velocity_bins: int) -> dict:
    if not points:
        base = {
            "num_points": 0,
            "mean_x": 0,
            "mean_y": 0,
            "mean_z": 0,
            "mean_velocity": 0,
            "max_snr": 0,
            "mean_snr": 0,
            "mean_noise": 0,
            "centroid_range": 0,
        }
    else:
        xs = [to_float(p["x"]) for p in points]
        ys = [to_float(p["y"]) for p in points]
        zs = [to_float(p["z"]) for p in points]
        vs = [to_float(p["velocity"]) for p in points]
        snrs = [to_float(p.get("snr", "")) for p in points]
        noises = [to_float(p.get("noise", "")) for p in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        mean_z = sum(zs) / len(zs)
        base = {
            "num_points": len(points),
            "mean_x": mean_x,
            "mean_y": mean_y,
            "mean_z": mean_z,
            "mean_velocity": sum(vs) / len(vs),
            "max_snr": max(snrs),
            "mean_snr": sum(snrs) / len(snrs),
            "mean_noise": sum(noises) / len(noises),
            "centroid_range": math.sqrt(mean_x * mean_x + mean_y * mean_y + mean_z * mean_z),
        }

    range_hist = [0] * range_bins
    velocity_hist = [0] * velocity_bins
    for point in points:
        x = to_float(point["x"])
        y = to_float(point["y"])
        z = to_float(point["z"])
        velocity = to_float(point["velocity"])
        rng = math.sqrt(x * x + y * y + z * z)
        range_hist[bin_index(rng, 0.0, 9.0, range_bins)] += 1
        velocity_hist[bin_index(velocity, -20.0, 20.0, velocity_bins)] += 1

    for i, value in enumerate(range_hist):
        base[f"range_bin_{i}"] = value
    for i, value in enumerate(velocity_hist):
        base[f"velocity_bin_{i}"] = value
    return base


def load_points(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_features(
    points_path: Path,
    out_path: Path,
    window_s: float,
    range_bins: int,
    velocity_bins: int,
) -> None:
    rows = load_points(points_path)
    windows: dict[int, list[dict]] = defaultdict(list)

    for row in rows:
        elapsed = to_float(row.get("elapsed_s", ""))
        if elapsed == 0 and "timestamp" in row:
            elapsed = to_float(row["timestamp"]) - to_float(rows[0]["timestamp"])
        windows[int(elapsed / window_s)].append(row)

    feature_rows = []
    for window_id in sorted(windows):
        start_s = window_id * window_s
        summary = summarize_points(windows[window_id], range_bins, velocity_bins)
        summary["window_id"] = window_id
        summary["start_s"] = start_s
        summary["end_s"] = start_s + window_s
        feature_rows.append(summary)

    if not feature_rows:
        raise RuntimeError(f"No points found in {points_path}")

    fieldnames = ["window_id", "start_s", "end_s"] + [
        key for key in feature_rows[0].keys() if key not in {"window_id", "start_s", "end_s"}
    ]
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"Wrote {len(feature_rows)} feature rows to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate radar point clouds into fixed-width features.")
    parser.add_argument("points_csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--range-bins", type=int, default=16)
    parser.add_argument("--velocity-bins", type=int, default=16)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out = args.out or args.points_csv.with_name("radar_features.csv")
    write_features(args.points_csv, out, args.window_s, args.range_bins, args.velocity_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
