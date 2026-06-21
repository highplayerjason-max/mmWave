import argparse
import csv
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import serial


MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 40
TLV_HEADER_LEN = 8

TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_RANGE_DOPPLER_HEATMAP = 5
TLV_STATS = 6
TLV_SIDE_INFO = 7


@dataclass
class RadarPoint:
    x: float
    y: float
    z: float
    velocity: float
    snr: Optional[int] = None
    noise: Optional[int] = None


def send_config(port: str, baud: int, cfg_path: Path, line_delay: float) -> None:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with serial.Serial(port, baud, timeout=1) as cfg_serial:
        for raw_line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            cfg_serial.write((line + "\n").encode("ascii"))
            time.sleep(line_delay)

            response = cfg_serial.read_all().decode(errors="ignore").strip()
            if response:
                print(response)


def probe_cli(port: str, baud: int) -> int:
    try:
        cli_serial = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open CLI port {port}: {exc}", file=sys.stderr)
        return 1

    with cli_serial:
        cli_serial.reset_input_buffer()
        cli_serial.write(b"\n")
        time.sleep(0.3)
        response = cli_serial.read(512).decode(errors="ignore")

    if response.strip():
        print(response.strip())
        return 0

    print(f"No CLI response from {port}. Reset or power-cycle the board, then try again.", file=sys.stderr)
    return 1


def parse_tlv_payload(packet: bytes, offset: int, tlv_length: int) -> tuple[bytes, int]:
    remaining = len(packet) - offset

    # mmWave SDK OOB Demo stores TLV length as payload bytes only. The 8-byte
    # TLV header has already been consumed before this function is called.
    if tlv_length <= remaining:
        payload_len = tlv_length
    else:
        payload_len = max(0, remaining)

    return packet[offset : offset + payload_len], offset + payload_len


def parse_packet(packet: bytes) -> dict:
    if len(packet) < HEADER_LEN:
        raise ValueError("Packet is shorter than the frame header")

    (
        magic,
        version,
        total_packet_len,
        platform,
        frame_number,
        time_cpu_cycles,
        num_detected_obj,
        num_tlvs,
        subframe_number,
    ) = struct.unpack_from("<8s8I", packet, 0)

    if magic != MAGIC_WORD:
        raise ValueError("Packet does not start with the TI magic word")

    points: list[RadarPoint] = []
    side_info: list[tuple[int, int]] = []
    range_profile: list[int] = []
    noise_profile: list[int] = []
    range_doppler_heatmap: list[int] = []
    stats = {}
    offset = HEADER_LEN

    for _ in range(num_tlvs):
        if offset + TLV_HEADER_LEN > len(packet):
            break

        tlv_type, tlv_length = struct.unpack_from("<2I", packet, offset)
        offset += TLV_HEADER_LEN
        payload, offset = parse_tlv_payload(packet, offset, tlv_length)

        if tlv_type == TLV_DETECTED_POINTS:
            for i in range(len(payload) // 16):
                x, y, z, velocity = struct.unpack_from("<4f", payload, i * 16)
                points.append(RadarPoint(x=x, y=y, z=z, velocity=velocity))

        elif tlv_type == TLV_SIDE_INFO:
            for i in range(len(payload) // 4):
                snr, noise = struct.unpack_from("<2H", payload, i * 4)
                side_info.append((snr, noise))

        elif tlv_type == TLV_RANGE_PROFILE:
            range_profile = list(struct.unpack_from(f"<{len(payload) // 2}H", payload, 0)) if payload else []

        elif tlv_type == TLV_NOISE_PROFILE:
            noise_profile = list(struct.unpack_from(f"<{len(payload) // 2}H", payload, 0)) if payload else []

        elif tlv_type == TLV_RANGE_DOPPLER_HEATMAP:
            range_doppler_heatmap = list(struct.unpack_from(f"<{len(payload) // 2}H", payload, 0)) if payload else []

        elif tlv_type == TLV_STATS and len(payload) >= 24:
            (
                inter_frame_processing_time,
                transmit_output_time,
                inter_frame_processing_margin,
                inter_chirp_processing_margin,
                active_frame_cpu_load,
                inter_frame_cpu_load,
            ) = struct.unpack_from("<6I", payload, 0)
            stats = {
                "inter_frame_processing_time": inter_frame_processing_time,
                "transmit_output_time": transmit_output_time,
                "inter_frame_processing_margin": inter_frame_processing_margin,
                "inter_chirp_processing_margin": inter_chirp_processing_margin,
                "active_frame_cpu_load": active_frame_cpu_load,
                "inter_frame_cpu_load": inter_frame_cpu_load,
            }

    for point, (snr, noise) in zip(points, side_info):
        point.snr = snr
        point.noise = noise

    return {
        "version": version,
        "total_packet_len": total_packet_len,
        "platform": platform,
        "frame_number": frame_number,
        "time_cpu_cycles": time_cpu_cycles,
        "num_detected_obj": num_detected_obj,
        "num_tlvs": num_tlvs,
        "subframe_number": subframe_number,
        "points": points,
        "range_profile": range_profile,
        "noise_profile": noise_profile,
        "range_doppler_heatmap": range_doppler_heatmap,
        "stats": stats,
    }


def find_next_packet(buffer: bytearray) -> Optional[bytes]:
    magic_index = buffer.find(MAGIC_WORD)
    if magic_index < 0:
        if len(buffer) > 4096:
            del buffer[:-len(MAGIC_WORD)]
        return None

    if magic_index > 0:
        del buffer[:magic_index]

    if len(buffer) < HEADER_LEN:
        return None

    total_packet_len = struct.unpack_from("<I", buffer, 12)[0]
    if total_packet_len < HEADER_LEN:
        del buffer[: len(MAGIC_WORD)]
        return None

    if len(buffer) < total_packet_len:
        return None

    packet = bytes(buffer[:total_packet_len])
    del buffer[:total_packet_len]
    return packet


def open_csv(path: Optional[Path]):
    if path is None:
        return None, None

    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp",
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
    return csv_file, writer


def log_frame(frame: dict, writer: Optional[csv.DictWriter]) -> None:
    points: list[RadarPoint] = frame["points"]
    print(f"Frame {frame['frame_number']} | points: {len(points)}")

    for point in points[:5]:
        print(
            "  "
            f"x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f}, "
            f"v={point.velocity:.2f}, snr={point.snr}, noise={point.noise}"
        )

    if writer is None:
        return

    timestamp = time.time()
    for point_id, point in enumerate(points):
        writer.writerow(
            {
                "timestamp": timestamp,
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


def read_data(
    port: str,
    baud: int,
    out_csv: Optional[Path],
    max_frames: Optional[int],
    idle_timeout: Optional[float],
) -> int:
    buffer = bytearray()
    frames_read = 0
    last_rx_time = time.time()
    csv_file, writer = open_csv(out_csv)

    try:
        try:
            data_serial = serial.Serial(port, baud, timeout=0.1)
        except serial.SerialException as exc:
            print(f"Could not open data port {port}: {exc}", file=sys.stderr)
            print("Close mmWave Demo Visualizer or any other serial terminal and try again.", file=sys.stderr)
            return 1

        with data_serial:
            print(f"Reading radar data from {port} at {baud} baud. Press Ctrl+C to stop.", flush=True)

            while max_frames is None or frames_read < max_frames:
                chunk = data_serial.read(4096)
                if chunk:
                    last_rx_time = time.time()
                    buffer.extend(chunk)
                elif idle_timeout is not None and time.time() - last_rx_time > idle_timeout:
                    print(f"No data received from {port} for {idle_timeout:.1f}s.", file=sys.stderr)
                    return 1

                packet = find_next_packet(buffer)
                if packet is None:
                    continue

                try:
                    frame = parse_packet(packet)
                except ValueError as exc:
                    print(f"Skipping malformed packet: {exc}", file=sys.stderr)
                    continue

                log_frame(frame, writer)
                frames_read += 1

                if csv_file is not None:
                    csv_file.flush()

    finally:
        if csv_file is not None:
            csv_file.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read TI mmWave SDK OOB Demo UART TLV point-cloud data."
    )
    parser.add_argument("--cfg-file", type=Path, help="Path to the .cfg profile to send.")
    parser.add_argument("--cfg-port", default="COM13", help="CLI/config UART port.")
    parser.add_argument("--data-port", default="COM14", help="Data UART port.")
    parser.add_argument("--cfg-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--line-delay", type=float, default=0.05)
    parser.add_argument("--out", type=Path, help="Optional CSV output path.")
    parser.add_argument("--frames", type=int, help="Stop after this many frames.")
    parser.add_argument("--idle-timeout", type=float, default=15.0)
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Do not send a .cfg file before reading data.",
    )
    parser.add_argument(
        "--probe-cli",
        action="store_true",
        help="Only check whether the CLI port responds, then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.probe_cli:
        return probe_cli(args.cfg_port, args.cfg_baud)

    if not args.skip_config:
        if args.cfg_file is None:
            print("Either pass --cfg-file or use --skip-config.", file=sys.stderr)
            return 2
        send_config(args.cfg_port, args.cfg_baud, args.cfg_file, args.line_delay)
        time.sleep(1)

    return read_data(args.data_port, args.data_baud, args.out, args.frames, args.idle_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
