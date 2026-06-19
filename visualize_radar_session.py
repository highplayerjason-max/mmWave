import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def plot_xy(points: list[dict], out_path: Path) -> None:
    xs = [to_float(row["x"]) for row in points]
    ys = [to_float(row["y"]) for row in points]
    snr = [to_float(row.get("snr", "")) for row in points]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=snr, s=10, cmap="viridis", alpha=0.75)
    plt.colorbar(scatter, label="SNR")
    plt.xlabel("x lateral (m)")
    plt.ylabel("y forward (m)")
    plt.title("Radar Point Cloud Top-Down View")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_frame_counts(frames: list[dict], out_path: Path) -> None:
    if not frames:
        return
    elapsed = [to_float(row.get("elapsed_s", "")) for row in frames]
    counts = [to_float(row["num_points"]) for row in frames]

    plt.figure(figsize=(9, 4))
    plt.plot(elapsed, counts, linewidth=1.5)
    plt.xlabel("elapsed time (s)")
    plt.ylabel("detected points")
    plt.title("Detected Points Per Frame")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_velocity(points: list[dict], out_path: Path) -> None:
    elapsed = [to_float(row.get("elapsed_s", "")) for row in points]
    velocity = [to_float(row["velocity"]) for row in points]
    y = [to_float(row["y"]) for row in points]

    plt.figure(figsize=(9, 5))
    scatter = plt.scatter(elapsed, velocity, c=y, s=8, cmap="plasma", alpha=0.7)
    plt.colorbar(scatter, label="y forward (m)")
    plt.xlabel("elapsed time (s)")
    plt.ylabel("radial velocity (m/s)")
    plt.title("Velocity Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def visualize(session_dir: Path) -> None:
    points_path = session_dir / "points.csv"
    frames_path = session_dir / "frames.csv"
    out_dir = session_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    points = load_csv(points_path)
    frames = load_csv(frames_path) if frames_path.exists() else []
    if not points:
        raise RuntimeError(f"No points found in {points_path}")

    plot_xy(points, out_dir / "xy_point_cloud.png")
    plot_frame_counts(frames, out_dir / "points_per_frame.png")
    plot_velocity(points, out_dir / "velocity_over_time.png")
    print(f"Wrote plots to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create radar-only plots from a recorded session.")
    parser.add_argument("session_dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    visualize(args.session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
