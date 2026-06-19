import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import serial

from mmwave_uart_logger import find_next_packet, parse_packet, send_config


SDK_PROFILE_3D = Path(
    r"C:\ti\mmwave_sdk_03_06_02_00-LTS\packages\ti\demo\xwr68xx\mmw\profiles\profile_3d.cfg"
)


def open_csv(path: Optional[Path]):
    if path is None:
        return None, None

    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "timestamp",
            "elapsed_s",
            "frame_number",
            "point_id",
            "x",
            "y",
            "z",
            "velocity",
            "snr",
            "noise",
        ],
    )
    writer.writeheader()
    return file, writer


def write_points(frame: dict, elapsed_s: float, writer) -> None:
    if writer is None:
        return

    timestamp = time.time()
    for point_id, point in enumerate(frame["points"]):
        writer.writerow(
            {
                "timestamp": timestamp,
                "elapsed_s": elapsed_s,
                "frame_number": frame["frame_number"],
                "point_id": point_id,
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "velocity": point.velocity,
                "snr": point.snr,
                "noise": point.noise,
            }
        )


class LiveRadarPlot:
    def __init__(self, trail_frames: int, xlim: tuple[float, float], ylim: tuple[float, float]):
        self.trail_frames = trail_frames
        self.points_history = deque(maxlen=trail_frames)
        self.frame_times = deque(maxlen=200)
        self.frame_counts = deque(maxlen=200)
        self.velocity_times = deque(maxlen=2000)
        self.velocities = deque(maxlen=2000)
        self.velocity_ranges = deque(maxlen=2000)

        self.fig, ((self.ax_xy, self.ax_counts), (self.ax_vel, self.ax_text)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Live mmWave Radar Viewer")

        self.xy_scatter = self.ax_xy.scatter([], [], s=18, c=[], cmap="viridis", vmin=0, vmax=400)
        self.xy_colorbar = self.fig.colorbar(self.xy_scatter, ax=self.ax_xy)
        self.xy_colorbar.set_label("SNR")
        self.ax_xy.set_title(f"Top-Down XY Point Cloud ({trail_frames}-frame trail)")
        self.ax_xy.set_xlabel("x lateral (m)")
        self.ax_xy.set_ylabel("y forward (m)")
        self.ax_xy.set_xlim(*xlim)
        self.ax_xy.set_ylim(*ylim)
        self.ax_xy.grid(True, alpha=0.3)

        (self.count_line,) = self.ax_counts.plot([], [], linewidth=1.5)
        self.ax_counts.set_title("Detected Points Per Frame")
        self.ax_counts.set_xlabel("elapsed time (s)")
        self.ax_counts.set_ylabel("points")
        self.ax_counts.grid(True, alpha=0.3)

        self.vel_scatter = self.ax_vel.scatter([], [], s=8, c=[], cmap="plasma", vmin=0, vmax=9)
        self.vel_colorbar = self.fig.colorbar(self.vel_scatter, ax=self.ax_vel)
        self.vel_colorbar.set_label("range (m)")
        self.ax_vel.set_title("Velocity Over Time")
        self.ax_vel.set_xlabel("elapsed time (s)")
        self.ax_vel.set_ylabel("velocity (m/s)")
        self.ax_vel.set_ylim(-5, 5)
        self.ax_vel.grid(True, alpha=0.3)

        self.ax_text.axis("off")
        self.status_text = self.ax_text.text(0.02, 0.95, "", va="top", family="monospace")
        self.fig.tight_layout()

    def update(self, frame: dict, elapsed_s: float) -> None:
        points = frame["points"]
        self.points_history.append(points)
        self.frame_times.append(elapsed_s)
        self.frame_counts.append(len(points))

        xs = []
        ys = []
        snrs = []
        for history_points in self.points_history:
            for point in history_points:
                xs.append(point.x)
                ys.append(point.y)
                snrs.append(point.snr or 0)
        self.xy_scatter.set_offsets(list(zip(xs, ys)) if xs else [])
        self.xy_scatter.set_array(snrs)

        self.count_line.set_data(list(self.frame_times), list(self.frame_counts))
        if self.frame_times:
            self.ax_counts.set_xlim(max(0, self.frame_times[0]), max(5, self.frame_times[-1]))
        max_count = max(self.frame_counts) if self.frame_counts else 1
        self.ax_counts.set_ylim(0, max(10, max_count * 1.2))

        for point in points:
            rng = (point.x * point.x + point.y * point.y + point.z * point.z) ** 0.5
            self.velocity_times.append(elapsed_s)
            self.velocities.append(point.velocity)
            self.velocity_ranges.append(rng)
        self.vel_scatter.set_offsets(
            list(zip(self.velocity_times, self.velocities)) if self.velocity_times else []
        )
        self.vel_scatter.set_array(list(self.velocity_ranges))
        if self.velocity_times:
            self.ax_vel.set_xlim(max(0, self.velocity_times[0]), max(5, self.velocity_times[-1]))

        self.status_text.set_text(
            "\n".join(
                [
                    f"frame_number : {frame['frame_number']}",
                    f"elapsed_s    : {elapsed_s:0.2f}",
                    f"points       : {len(points)}",
                    f"num_tlvs     : {frame['num_tlvs']}",
                    f"packet_len   : {frame['total_packet_len']}",
                    "",
                    "Close this window or press Ctrl+C to stop.",
                ]
            )
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def run_viewer(args: argparse.Namespace) -> int:
    if not args.skip_config:
        send_config(args.cfg_port, args.cfg_baud, args.cfg_file, args.line_delay)
        time.sleep(1)

    csv_file, csv_writer = open_csv(args.out)
    buffer = bytearray()
    start = time.time()
    last_rx = start
    frames_read = 0
    plot = LiveRadarPlot(
        trail_frames=args.trail_frames,
        xlim=(args.x_min, args.x_max),
        ylim=(args.y_min, args.y_max),
    )

    try:
        try:
            data_serial = serial.Serial(args.data_port, args.data_baud, timeout=0.1)
        except serial.SerialException as exc:
            print(f"Could not open data port {args.data_port}: {exc}", file=sys.stderr)
            return 1

        with data_serial:
            while plt.fignum_exists(plot.fig.number):
                if args.frames is not None and frames_read >= args.frames:
                    break
                if args.seconds is not None and time.time() - start >= args.seconds:
                    break

                chunk = data_serial.read(4096)
                if chunk:
                    last_rx = time.time()
                    buffer.extend(chunk)
                elif time.time() - last_rx > args.idle_timeout:
                    print(f"No data received from {args.data_port} for {args.idle_timeout:.1f}s.", file=sys.stderr)
                    return 1

                packet = find_next_packet(buffer)
                if packet is None:
                    plt.pause(0.001)
                    continue

                try:
                    frame = parse_packet(packet)
                except ValueError as exc:
                    print(f"Skipping malformed packet: {exc}", file=sys.stderr)
                    continue

                elapsed_s = time.time() - start
                write_points(frame, elapsed_s, csv_writer)
                if csv_file is not None:
                    csv_file.flush()
                plot.update(frame, elapsed_s)
                frames_read += 1
                if frames_read % 10 == 0:
                    print(
                        f"frames={frames_read} frame_number={frame['frame_number']} "
                        f"points={len(frame['points'])} elapsed={elapsed_s:.1f}s",
                        flush=True,
                    )

    finally:
        if csv_file is not None:
            csv_file.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live radar point-cloud viewer for TI mmWave OOB Demo.")
    parser.add_argument("--cfg-file", type=Path, default=SDK_PROFILE_3D)
    parser.add_argument("--cfg-port", default="COM13")
    parser.add_argument("--data-port", default="COM14")
    parser.add_argument("--cfg-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--line-delay", type=float, default=0.05)
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--out", type=Path, help="Optional CSV log path.")
    parser.add_argument("--seconds", type=float, help="Optional runtime limit.")
    parser.add_argument("--frames", type=int, help="Optional frame limit.")
    parser.add_argument("--idle-timeout", type=float, default=15.0)
    parser.add_argument("--trail-frames", type=int, default=5)
    parser.add_argument("--x-min", type=float, default=-3.0)
    parser.add_argument("--x-max", type=float, default=3.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=6.0)
    return parser


def main() -> int:
    return run_viewer(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
