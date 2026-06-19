import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import serial

from mmwave_uart_logger import (
    find_next_packet,
    parse_packet,
    send_config,
)


DEFAULT_CFG = Path(
    r"C:\ti\mmwave_sdk_03_06_02_00-LTS\packages\ti\demo\xwr68xx\mmw\profiles\profile_3d.cfg"
)


def session_dir(base_dir: Path) -> Path:
    name = time.strftime("session_%Y%m%d_%H%M%S")
    path = base_dir / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def open_writers(out_dir: Path):
    points_file = (out_dir / "points.csv").open("w", newline="", encoding="utf-8")
    frames_file = (out_dir / "frames.csv").open("w", newline="", encoding="utf-8")

    points_writer = csv.DictWriter(
        points_file,
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
    frames_writer = csv.DictWriter(
        frames_file,
        fieldnames=[
            "timestamp",
            "elapsed_s",
            "frame_number",
            "num_points",
            "num_tlvs",
            "subframe_number",
            "total_packet_len",
        ],
    )
    points_writer.writeheader()
    frames_writer.writeheader()
    return points_file, frames_file, points_writer, frames_writer


def write_frame(frame: dict, start_time: float, points_writer, frames_writer) -> int:
    timestamp = time.time()
    elapsed = timestamp - start_time
    points = frame["points"]

    frames_writer.writerow(
        {
            "timestamp": timestamp,
            "elapsed_s": elapsed,
            "frame_number": frame["frame_number"],
            "num_points": len(points),
            "num_tlvs": frame["num_tlvs"],
            "subframe_number": frame["subframe_number"],
            "total_packet_len": frame["total_packet_len"],
        }
    )

    for point_id, point in enumerate(points):
        points_writer.writerow(
            {
                "timestamp": timestamp,
                "elapsed_s": elapsed,
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

    return len(points)


def record_session(
    out_dir: Path,
    data_port: str,
    data_baud: int,
    seconds: Optional[float],
    frames: Optional[int],
    idle_timeout: float,
) -> int:
    buffer = bytearray()
    frames_read = 0
    points_read = 0
    start_time = time.time()
    last_rx_time = start_time
    points_file, frames_file, points_writer, frames_writer = open_writers(out_dir)

    try:
        try:
            data_serial = serial.Serial(data_port, data_baud, timeout=0.1)
        except serial.SerialException as exc:
            print(f"Could not open data port {data_port}: {exc}", file=sys.stderr)
            return 1

        with data_serial:
            print(f"Recording radar session into {out_dir}", flush=True)

            while True:
                if frames is not None and frames_read >= frames:
                    break
                if seconds is not None and time.time() - start_time >= seconds:
                    break

                chunk = data_serial.read(4096)
                if chunk:
                    last_rx_time = time.time()
                    buffer.extend(chunk)
                elif time.time() - last_rx_time > idle_timeout:
                    print(f"No data received from {data_port} for {idle_timeout:.1f}s.", file=sys.stderr)
                    return 1

                while True:
                    packet = find_next_packet(buffer)
                    if packet is None:
                        break

                    try:
                        frame = parse_packet(packet)
                    except ValueError as exc:
                        print(f"Skipping malformed packet: {exc}", file=sys.stderr)
                        continue

                    num_points = write_frame(frame, start_time, points_writer, frames_writer)
                    frames_read += 1
                    points_read += num_points

                    if frames_read % 10 == 0:
                        print(f"frames={frames_read} points={points_read}", flush=True)

    finally:
        points_file.close()
        frames_file.close()

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": time.time() - start_time,
        "frames": frames_read,
        "points": points_read,
        "data_port": data_port,
        "data_baud": data_baud,
        "files": {
            "points": "points.csv",
            "frames": "frames.csv",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: frames={frames_read} points={points_read}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a radar-only mmWave session.")
    parser.add_argument("--out-root", type=Path, default=Path("sessions"))
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--cfg-port", default="COM13")
    parser.add_argument("--data-port", default="COM14")
    parser.add_argument("--cfg-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--line-delay", type=float, default=0.05)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--idle-timeout", type=float, default=15.0)
    parser.add_argument("--skip-config", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = session_dir(args.out_root)

    if not args.skip_config:
        send_config(args.cfg_port, args.cfg_baud, args.cfg_file, args.line_delay)
        time.sleep(1)

    return record_session(
        out_dir=out_dir,
        data_port=args.data_port,
        data_baud=args.data_baud,
        seconds=args.seconds,
        frames=args.frames,
        idle_timeout=args.idle_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
