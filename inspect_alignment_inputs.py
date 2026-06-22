import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


def read_points(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                [
                    float(row.get("x", row.get("x_m", 0.0))),
                    float(row.get("y", row.get("y_m", 0.0))),
                    float(row.get("z", row.get("z_m", 0.0))),
                    float(row.get("snr", row.get("snr_db", 0.0)) or 0.0),
                ]
            )
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def kde(points: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    heat = np.zeros_like(xx, dtype=np.float32)
    for x, y, _z, snr in points:
        if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
            weight = max(0.5, float(snr) - 12.0)
            heat += weight * np.exp(-0.5 * (((xx - x) / 0.075) ** 2 + ((yy - y) / 0.065) ** 2))
    if float(np.max(heat)) > 0:
        heat /= float(np.max(heat))
    return xs, ys, heat


def top_bins(points: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float], bin_m: float, limit: int):
    if len(points) == 0:
        return []
    selected = points[
        (points[:, 0] >= xlim[0])
        & (points[:, 0] <= xlim[1])
        & (points[:, 1] >= ylim[0])
        & (points[:, 1] <= ylim[1])
    ]
    bins = {}
    for x, y, z, snr in selected:
        key = (round(float(x) / bin_m) * bin_m, round(float(y) / bin_m) * bin_m)
        entry = bins.setdefault(key, {"count": 0, "snr": 0.0, "z": []})
        entry["count"] += 1
        entry["snr"] += float(snr)
        entry["z"].append(float(z))
    ranked = []
    for (x, y), entry in bins.items():
        ranked.append((entry["count"], entry["snr"], x, y, float(np.mean(entry["z"]))))
    ranked.sort(reverse=True)
    return ranked[:limit]


def save_rgb_grid(image_path: Path, out_path: Path, title: str) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    ax.imshow(image)
    ax.set_title(title)
    ax.set_xlabel("image u (px)")
    ax.set_ylabel("image v (px)")
    ax.set_xticks(np.arange(0, image.width + 1, 80))
    ax.set_yticks(np.arange(0, image.height + 1, 60))
    ax.grid(color="cyan", alpha=0.35, linewidth=0.7)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect RGB/mmWave alignment inputs before choosing anchors.")
    parser.add_argument("--points-a", type=Path, default=Path("captures/A_points.csv"))
    parser.add_argument("--points-b", type=Path, default=Path("captures/B_points.csv"))
    parser.add_argument("--rgb-a", type=Path, default=Path("captures/A_rgb.jpg"))
    parser.add_argument("--rgb-b", type=Path, default=Path("captures/B_rgb.jpg"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/inspect_alignment"))
    parser.add_argument("--xlim", nargs=2, type=float, default=(-1.4, 1.6))
    parser.add_argument("--ylim", nargs=2, type=float, default=(0.2, 3.0))
    parser.add_argument("--near-y-min", type=float, default=0.45)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xlim = tuple(args.xlim)
    ylim = tuple(args.ylim)
    pts_a = read_points(args.points_a)
    pts_b = read_points(args.points_b)
    far_a = pts_a[pts_a[:, 1] >= args.near_y_min]
    far_b = pts_b[pts_b[:, 1] >= args.near_y_min]

    save_rgb_grid(args.rgb_a, args.out_dir / "A_rgb_with_grid.png", "A RGB with pixel grid")
    save_rgb_grid(args.rgb_b, args.out_dir / "B_rgb_with_grid.png", "B RGB with pixel grid")

    image_a = Image.open(args.rgb_a).convert("RGB")
    image_b = Image.open(args.rgb_b).convert("RGB").resize(image_a.size)
    ImageChops.difference(image_a, image_b).save(args.out_dir / "rgb_abs_difference.png")

    xs, ys, heat_a = kde(far_a, xlim, ylim, 320, 260)
    _xs, _ys, heat_b = kde(far_b, xlim, ylim, 320, 260)
    common = np.minimum(heat_a, heat_b)
    diff = np.clip(heat_a - heat_b, 0.0, 1.0)
    show_a = np.clip(heat_a - 0.35 * common, 0.0, 1.0)
    show_b = np.clip(heat_b - 0.35 * common, 0.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=160, constrained_layout=True)
    panels = [
        (heat_a, "A raw radar KDE"),
        (heat_b, "B raw radar KDE"),
        (common, "Common A/B clutter"),
        (show_a, "A after common suppression"),
        (show_b, "B after common suppression"),
        (diff, "Positive difference A - B"),
    ]
    for ax, (heat, title) in zip(axes.flat, panels):
        im = ax.imshow(heat, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap="magma", vmin=0, vmax=max(1.0, float(np.max(heat))))
        ax.scatter(far_a[:, 0], far_a[:, 1], s=4, c="cyan", alpha=0.15, label="A pts" if "A" in title else None)
        ax.set_title(title)
        ax.set_xlabel("x left/right (m)")
        ax.set_ylabel("y forward (m)")
        ax.grid(alpha=0.2)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    fig.savefig(args.out_dir / "radar_kde_debug.png", bbox_inches="tight")
    plt.close(fig)

    with (args.out_dir / "top_radar_peaks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "rank", "count", "snr_sum", "x_m", "y_m", "mean_z_m"])
        for scene, points in [("A", far_a), ("B", far_b)]:
            for rank, item in enumerate(top_bins(points, xlim, ylim, 0.08, 30), start=1):
                writer.writerow([scene, rank, *item])

    print(args.out_dir)
    print("A points:", len(pts_a), "A after near filter:", len(far_a))
    print("B points:", len(pts_b), "B after near filter:", len(far_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
