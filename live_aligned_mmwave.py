import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import serial

from mmwave_uart_logger import find_next_packet, parse_packet, send_config
from replicate_aligned_mmwave import (
    enhance,
    kde_heat,
    make_grid,
    parse_pair,
    similarity_from_anchors,
    warp_heat_to_pixels,
)


DEFAULT_CFG = Path(
    r"C:\ti\mmwave_sdk_03_06_02_00-LTS\packages\ti\demo\xwr68xx\mmw\profiles\profile_3d.cfg"
)
DEFAULT_MM_ANCHORS = [(-0.64, 2.56), (0.72, 2.56)]
DEFAULT_PX_ANCHORS = [(205.0, 300.0), (465.0, 245.0)]


class RadarThread(threading.Thread):
    def __init__(
        self,
        events: queue.Queue,
        stop_event: threading.Event,
        cfg_port: str,
        data_port: str,
        cfg_baud: int,
        data_baud: int,
        cfg_file: Path,
        skip_config: bool,
    ):
        super().__init__(daemon=True)
        self.events = events
        self.stop_event = stop_event
        self.cfg_port = cfg_port
        self.data_port = data_port
        self.cfg_baud = cfg_baud
        self.data_baud = data_baud
        self.cfg_file = cfg_file
        self.skip_config = skip_config

    def run(self) -> None:
        try:
            if not self.skip_config:
                self.events.put(("status", "sending radar config"))
                send_config(self.cfg_port, self.cfg_baud, self.cfg_file, 0.05)
                time.sleep(1.0)

            self.events.put(("status", f"opening {self.data_port}"))
            with serial.Serial(self.data_port, self.data_baud, timeout=0.1) as data_serial:
                buffer = bytearray()
                last_rx = time.time()
                self.events.put(("status", "radar streaming"))

                while not self.stop_event.is_set():
                    chunk = data_serial.read(4096)
                    if chunk:
                        last_rx = time.time()
                        buffer.extend(chunk)
                    elif time.time() - last_rx > 5:
                        self.events.put(("warning", "no radar data for 5s"))
                        last_rx = time.time()

                    while not self.stop_event.is_set():
                        packet = find_next_packet(buffer)
                        if packet is None:
                            break
                        try:
                            self.events.put(("frame", time.time(), parse_packet(packet)))
                        except ValueError as exc:
                            self.events.put(("warning", f"bad packet: {exc}"))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.events.put(("stopped", None))


def points_to_array(frames: deque, limits: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = limits
    rows = []
    for frame in frames:
        for point in frame.get("points", []):
            if xmin <= point.x <= xmax and ymin <= point.y <= ymax:
                rows.append((point.x, point.y, float(point.snr or 80)))
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


def heat_to_bgr(heat: np.ndarray) -> np.ndarray:
    image = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    color_map = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_INFERNO)
    return cv2.applyColorMap(image, color_map)


def overlay_heat(frame_bgr: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    color = heat_to_bgr(heat)
    mask = np.clip(heat[..., None] * 1.8, 0.0, 1.0)
    return np.clip(frame_bgr * (1.0 - alpha * mask) + color * (alpha * mask), 0, 255).astype(np.uint8)


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def radar_topdown_panel(heat: np.ndarray, width: int, height: int) -> np.ndarray:
    if float(np.max(heat)) > 0:
        heat = heat / float(np.max(heat))
    image = heat_to_bgr(enhance(heat, 0.68))
    image = cv2.flip(image, 0)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def parse_limits(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("limits must be xmin,xmax,ymin,ymax")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Live RGB + aligned mmWave foreground viewer.")
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--cfg-port", default="COM13")
    parser.add_argument("--data-port", default="COM14")
    parser.add_argument("--cfg-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--skip-config", action="store_true", help="Do not send cfg; use this if the sensor is already streaming.")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--limits", type=parse_limits, default=(-1.4, 1.6, 0.20, 3.0))
    parser.add_argument("--mm-anchor", action="append", help="two radar anchors as x,y meters")
    parser.add_argument("--px-anchor", action="append", help="two RGB anchors as u,v pixels")
    parser.add_argument("--trail-frames", type=int, default=8)
    parser.add_argument("--background-seconds", type=float, default=3.0)
    parser.add_argument("--background-scale", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--gamma", type=float, default=0.68)
    parser.add_argument("--no-background", action="store_true", help="Show live returns without subtracting a background model.")
    args = parser.parse_args()

    mm_anchors = np.array(
        [parse_pair(item) for item in args.mm_anchor] if args.mm_anchor else DEFAULT_MM_ANCHORS,
        dtype=np.float64,
    )
    px_anchors = np.array(
        [parse_pair(item) for item in args.px_anchor] if args.px_anchor else DEFAULT_PX_ANCHORS,
        dtype=np.float64,
    )
    if mm_anchors.shape != (2, 2) or px_anchors.shape != (2, 2):
        raise SystemExit("Provide exactly two --mm-anchor and exactly two --px-anchor values.")

    xs, ys, xx, yy = make_grid(args.limits, 360, 260)
    scale, rot, trans, _theta = similarity_from_anchors(mm_anchors, px_anchors)

    events: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    radar = RadarThread(
        events,
        stop_event,
        args.cfg_port,
        args.data_port,
        args.cfg_baud,
        args.data_baud,
        args.cfg_file,
        args.skip_config,
    )
    radar.start()

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        stop_event.set()
        raise SystemExit(f"Could not open camera index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    recent_frames = deque(maxlen=max(1, args.trail_frames))
    bg_sum = np.zeros_like(xx, dtype=np.float32)
    bg_samples = 0
    background = np.zeros_like(xx, dtype=np.float32)
    started = time.time()
    status = "starting"
    frame_count = 0
    last_heat = np.zeros((args.height, args.width), dtype=np.float32)
    last_fg = np.zeros_like(last_heat)

    print("Controls: q/ESC quit, b reset background, c clear radar trail")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise SystemExit("Camera opened, but no frame was received.")
            frame_bgr = cv2.resize(frame_bgr, (args.width, args.height))

            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                kind = event[0]
                if kind == "frame":
                    recent_frames.append(event[2])
                    frame_count += 1
                elif kind in {"status", "warning", "error"}:
                    status = event[1]
                elif kind == "stopped":
                    if not stop_event.is_set():
                        status = "radar stopped"

            raw_points = sum(len(frame.get("points", [])) for frame in recent_frames)
            points = points_to_array(recent_frames, args.limits)
            heat = kde_heat(points, xx, yy, 0.075, 0.065)

            elapsed = time.time() - started
            calibrating = elapsed < args.background_seconds
            if calibrating:
                bg_sum += heat
                bg_samples += 1
                background = bg_sum / max(1, bg_samples)

            scene_max = max(float(np.max(heat)), float(np.max(background)), 1.0)
            live_heat = enhance(heat / scene_max, args.gamma)
            if args.no_background:
                fg_heat = heat.copy()
            else:
                fg_heat = np.maximum(heat - args.background_scale * background, 0.0)
            if float(np.max(fg_heat)) > 0:
                fg_heat = fg_heat / float(np.max(fg_heat))
            fg_heat = enhance(fg_heat, 0.60)

            last_heat = warp_heat_to_pixels(
                live_heat, xs, ys, scale, rot, trans, args.width, args.height, args.threshold, normalize=False
            )
            last_fg = warp_heat_to_pixels(
                fg_heat, xs, ys, scale, rot, trans, args.width, args.height, args.threshold, normalize=True
            )

            topdown_panel = radar_topdown_panel(heat, args.width, args.height)
            live_panel = heat_to_bgr(last_heat)
            fg_overlay = overlay_heat(frame_bgr, last_fg, 0.72)
            status_text = (
                f"Calibrating background {elapsed:.1f}/{args.background_seconds:.1f}s"
                if calibrating
                else f"{status} | radar frames {frame_count} | raw pts {raw_points} | filtered pts {len(points)}"
            )
            camera_panel = label_panel(frame_bgr, "RGB camera")
            topdown_panel = label_panel(topdown_panel, "Radar top-down before alignment")
            live_panel = label_panel(live_panel, "Live aligned mmWave")
            fg_overlay = label_panel(fg_overlay, "Foreground mmWave overlay")

            combined = np.vstack([np.hstack([camera_panel, topdown_panel]), np.hstack([live_panel, fg_overlay])])
            cv2.putText(
                combined,
                status_text,
                (10, combined.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Live RGB aligned mmWave", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                bg_sum.fill(0)
                bg_samples = 0
                background.fill(0)
                started = time.time()
                status = "background reset"
            if key == ord("c"):
                recent_frames.clear()
                last_heat.fill(0)
                last_fg.fill(0)
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
