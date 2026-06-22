import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_pair(text: str) -> tuple[float, float]:
    left, right = text.split(",", 1)
    return float(left), float(right)


def read_points(path: Path, limits: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = limits
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x = float(row.get("x", row.get("x_m", 0.0)))
            y = float(row.get("y", row.get("y_m", 0.0)))
            snr_raw = row.get("snr", row.get("snr_db", "0")) or "0"
            snr = float(snr_raw)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                rows.append((x, y, snr))
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


def make_grid(limits: tuple[float, float, float, float], nx: int, ny: int):
    xmin, xmax, ymin, ymax = limits
    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, xx, yy


def kde_heat(points: np.ndarray, xx: np.ndarray, yy: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    heat = np.zeros_like(xx, dtype=np.float32)
    for x, y, snr in points:
        weight = max(0.5, float(snr) - 12.0)
        heat += weight * np.exp(-0.5 * (((xx - x) / sigma_x) ** 2 + ((yy - y) / sigma_y) ** 2))
    if float(np.max(heat)) > 0:
        heat /= float(np.max(heat))
    return heat


def enhance(heat: np.ndarray, gamma: float) -> np.ndarray:
    adjusted = np.clip((heat - 0.035) / 0.965, 0.0, 1.0)
    return adjusted ** gamma


def similarity_from_anchors(mmwave: np.ndarray, pixels: np.ndarray):
    src0, src1 = mmwave
    dst0, dst1 = pixels
    v_src = src1 - src0
    v_dst = dst1 - dst0
    scale = np.linalg.norm(v_dst) / np.linalg.norm(v_src)
    theta = math.atan2(v_dst[1], v_dst[0]) - math.atan2(v_src[1], v_src[0])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    trans = dst0 - scale * (rot @ src0)
    return scale, rot, trans, theta


def warp_heat_to_pixels(
    heat: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    width: int,
    height: int,
    threshold: float,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros_like(canvas)
    y_idx, x_idx = np.where(heat > threshold)
    for iy, ix in zip(y_idx, x_idx):
        value = float(heat[iy, ix])
        pixel = scale * (rot @ np.array([xs[ix], ys[iy]])) + trans
        u = int(round(float(pixel[0])))
        v = int(round(float(pixel[1])))
        if -5 <= u < width + 5 and -5 <= v < height + 5:
            radius = 4
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px = u + dx
                    py = v + dy
                    if 0 <= px < width and 0 <= py < height:
                        weight = max(0.0, 1.0 - (dx * dx + dy * dy) / (radius * radius + 1))
                        canvas[py, px] += value * weight
                        weights[py, px] += weight
    mask = weights > 0
    canvas[mask] /= weights[mask]
    if float(np.max(canvas)) > 0:
        canvas /= float(np.max(canvas))
    return canvas


def plot_scene(rgb_path: Optional[Path], radar_pixels: np.ndarray, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.65), dpi=180, constrained_layout=True)
    if rgb_path:
        rgb = plt.imread(rgb_path)
        axes[0].imshow(rgb)
    else:
        axes[0].imshow(np.ones((720, 1280, 3), dtype=np.float32) * 0.92)
        axes[0].text(640, 360, "RGB not provided", ha="center", va="center", fontsize=14)
    axes[0].set_title(f"{title} RGB")
    axes[0].axis("off")

    im = axes[1].imshow(
        radar_pixels,
        cmap="magma",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[0, 1280, 720, 0],
        interpolation="bilinear",
    )
    axes[1].set_title("Aligned mmWave")
    axes[1].set_xlabel("image u (px)")
    axes[1].set_ylabel("image v (px)")
    axes[1].set_xlim(0, 1280)
    axes[1].set_ylim(720, 0)
    fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replicate the RGB-aligned mmWave presentation figure workflow.")
    parser.add_argument("--points-a", type=Path, required=True)
    parser.add_argument("--points-b", type=Path)
    parser.add_argument("--rgb-a", type=Path)
    parser.add_argument("--rgb-b", type=Path)
    parser.add_argument("--title-a", default="Scene A")
    parser.add_argument("--title-b", default="Scene B")
    parser.add_argument("--mm-anchor", action="append", required=True, help="exactly two x,y radar anchors in meters")
    parser.add_argument("--px-anchor", action="append", required=True, help="exactly two u,v pixel anchors")
    parser.add_argument("--limits", default="-1.35,1.35,0.0,1.65", help="xmin,xmax,ymin,ymax in radar meters")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/aligned_mmwave"))
    parser.add_argument("--soft-common-suppression", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.68)
    parser.add_argument("--threshold", type=float, default=0.025)
    args = parser.parse_args()

    if len(args.mm_anchor) != 2 or len(args.px_anchor) != 2:
        raise SystemExit("Provide exactly two --mm-anchor and two --px-anchor values.")

    limits = tuple(float(value) for value in args.limits.split(","))
    xs, ys, xx, yy = make_grid(limits, 360, 260)
    mm_anchors = np.array([parse_pair(item) for item in args.mm_anchor], dtype=np.float64)
    px_anchors = np.array([parse_pair(item) for item in args.px_anchor], dtype=np.float64)
    scale, rot, trans, theta = similarity_from_anchors(mm_anchors, px_anchors)

    heat_a = kde_heat(read_points(args.points_a, limits), xx, yy, 0.075, 0.065)
    heat_b = None
    if args.points_b:
        heat_b = kde_heat(read_points(args.points_b, limits), xx, yy, 0.075, 0.065)

    show_a = heat_a
    show_b = heat_b
    if heat_b is not None:
        common = np.minimum(heat_a, heat_b)
        show_a = np.clip(heat_a - args.soft_common_suppression * common, 0.0, 1.0)
        show_b = np.clip(heat_b - args.soft_common_suppression * common, 0.0, 1.0)

    outputs = []
    for name, heat, rgb, title in (
        ("scene_a_rgb_radar.png", show_a, args.rgb_a, args.title_a),
        ("scene_b_rgb_radar.png", show_b, args.rgb_b, args.title_b),
    ):
        if heat is None:
            continue
        if float(np.max(heat)) > 0:
            heat = heat / float(np.max(heat))
        pixels = warp_heat_to_pixels(
            enhance(heat, args.gamma), xs, ys, scale, rot, trans, 1280, 720, args.threshold
        )
        out = args.out_dir / name
        plot_scene(rgb, pixels, title, out)
        np.save(out.with_suffix(".heat.npy"), pixels)
        outputs.append(str(out))

    if heat_b is not None:
        diff = np.clip(heat_a - heat_b, 0.0, 1.0)
        if float(np.max(diff)) > 0:
            diff /= float(np.max(diff))
        pixels = warp_heat_to_pixels(enhance(diff, 0.60), xs, ys, scale, rot, trans, 1280, 720, args.threshold)
        out = args.out_dir / "positive_difference_rgb_radar.png"
        plot_scene(args.rgb_a, pixels, f"{args.title_a} positive difference", out)
        np.save(out.with_suffix(".heat.npy"), pixels)
        outputs.append(str(out))

    metadata = {
        "transform": "similarity: rotation + scale + translation",
        "scale_px_per_m": scale,
        "rotation_deg": math.degrees(theta),
        "translation_px": trans.tolist(),
        "mmwave_anchors_xy_m": mm_anchors.tolist(),
        "rgb_anchor_pixels": px_anchors.tolist(),
        "limits": limits,
        "outputs": outputs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "alignment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
