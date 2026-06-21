import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_points(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def point_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    x = np.array([as_float(row, "x") for row in rows], dtype=np.float32)
    y = np.array([as_float(row, "y") for row in rows], dtype=np.float32)
    z = np.array([as_float(row, "z") for row in rows], dtype=np.float32)
    velocity = np.array([as_float(row, "velocity") for row in rows], dtype=np.float32)
    snr = np.array([as_float(row, "snr") for row in rows], dtype=np.float32)
    rng = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    return {"x": x, "y": y, "z": z, "velocity": velocity, "snr": snr, "range": rng}


def make_grid(
    a: np.ndarray,
    b: np.ndarray,
    snr: np.ndarray,
    velocity: np.ndarray,
    a_range: tuple[float, float],
    b_range: tuple[float, float],
    bins: tuple[int, int],
) -> np.ndarray:
    count, _, _ = np.histogram2d(b, a, bins=bins, range=[b_range, a_range])
    snr_sum, _, _ = np.histogram2d(b, a, bins=bins, range=[b_range, a_range], weights=snr)
    snr_max = np.full(bins, 0.0, dtype=np.float32)
    vel_sum, _, _ = np.histogram2d(b, a, bins=bins, range=[b_range, a_range], weights=velocity)

    a_edges = np.linspace(a_range[0], a_range[1], bins[1] + 1)
    b_edges = np.linspace(b_range[0], b_range[1], bins[0] + 1)
    a_idx = np.clip(np.searchsorted(a_edges, a, side="right") - 1, 0, bins[1] - 1)
    b_idx = np.clip(np.searchsorted(b_edges, b, side="right") - 1, 0, bins[0] - 1)
    valid = (a >= a_range[0]) & (a <= a_range[1]) & (b >= b_range[0]) & (b <= b_range[1])
    for ai, bi, value in zip(a_idx[valid], b_idx[valid], snr[valid]):
        snr_max[bi, ai] = max(snr_max[bi, ai], value)

    count = count.astype(np.float32)
    mean_snr = np.divide(snr_sum, count, out=np.zeros_like(snr_sum, dtype=np.float32), where=count > 0)
    mean_velocity = np.divide(vel_sum, count, out=np.zeros_like(vel_sum, dtype=np.float32), where=count > 0)
    return np.stack([count, snr_max, mean_snr.astype(np.float32), mean_velocity.astype(np.float32)], axis=0)


def save_heatmap(grid: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str, extent: tuple[float, float, float, float]) -> None:
    # Channel 1 is max SNR, usually the most useful image-like representation.
    image = grid[1]
    plt.figure(figsize=(7, 5))
    plt.imshow(image, origin="lower", aspect="auto", extent=extent, cmap="magma")
    plt.colorbar(label="max SNR")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_radar_images(points_csv: Path, out_dir: Path, grid_size: int) -> None:
    rows = load_points(points_csv)
    if not rows:
        raise RuntimeError(f"No points found in {points_csv}")

    data = point_arrays(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = (grid_size, grid_size)

    xy = make_grid(data["x"], data["y"], data["snr"], data["velocity"], (-3.0, 3.0), (0.0, 6.0), bins)
    yz = make_grid(data["y"], data["z"], data["snr"], data["velocity"], (0.0, 6.0), (-2.0, 2.0), bins)
    xz = make_grid(data["x"], data["z"], data["snr"], data["velocity"], (-3.0, 3.0), (-2.0, 2.0), bins)
    rv = make_grid(data["range"], data["velocity"], data["snr"], data["velocity"], (0.0, 9.0), (-20.0, 20.0), bins)

    save_heatmap(xy, out_dir / "radar_xy.png", "Radar XY Sensor Image", "x lateral (m)", "y forward (m)", (-3.0, 3.0, 0.0, 6.0))
    save_heatmap(yz, out_dir / "radar_yz.png", "Radar YZ Sensor Image", "y forward (m)", "z vertical (m)", (0.0, 6.0, -2.0, 2.0))
    save_heatmap(xz, out_dir / "radar_xz.png", "Radar XZ Sensor Image", "x lateral (m)", "z vertical (m)", (-3.0, 3.0, -2.0, 2.0))
    save_heatmap(rv, out_dir / "radar_range_velocity.png", "Radar Range-Velocity Sensor Image", "range (m)", "velocity (m/s)", (0.0, 9.0, -20.0, 20.0))

    tensor = np.stack([xy, yz, xz, rv], axis=0)
    np.savez_compressed(
        out_dir / "radar_tensor.npz",
        tensor=tensor,
        views=np.array(["xy", "yz", "xz", "range_velocity"]),
        channels=np.array(["count", "max_snr", "mean_snr", "mean_velocity"]),
    )
    print(f"Wrote radar images and tensor to {out_dir}")
    print("tensor shape:", tensor.shape, "(views, channels, height, width)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert mmWave point clouds into OmniVLA/MuseVLA-style radar sensor images.")
    parser.add_argument("points_csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=128)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir or args.points_csv.with_name("radar_images")
    build_radar_images(args.points_csv, out_dir, args.grid_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
