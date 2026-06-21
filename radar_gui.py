import csv
import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import serial
from serial.tools import list_ports

from mmwave_uart_logger import find_next_packet, parse_packet, send_config


DEFAULT_CFG = Path(
    r"C:\ti\mmwave_sdk_03_06_02_00-LTS\packages\ti\demo\xwr68xx\mmw\profiles\profile_3d.cfg"
)


def try_open_port(port: str, baud: int) -> tuple[bool, str]:
    try:
        serial_port = serial.Serial(port, baud, timeout=0.2, write_timeout=0.2)
    except serial.SerialException as exc:
        return False, str(exc)
    except OSError as exc:
        return False, str(exc)

    serial_port.close()
    return True, "OK"


def probe_cli_prompt(port: str, baud: int) -> tuple[bool, str]:
    try:
        with serial.Serial(port, baud, timeout=0.5, write_timeout=0.5) as cli_serial:
            cli_serial.reset_input_buffer()
            cli_serial.write(b"\n")
            time.sleep(0.2)
            response = cli_serial.read(512).decode(errors="ignore")
    except Exception as exc:
        return False, str(exc)

    if "mmwDemo:/>" in response:
        return True, "mmwDemo prompt OK"
    return False, "CLI opened, but no mmwDemo prompt. Reset the board or flash OOB demo."


def send_cli_command(port: str, baud: int, command: str, delay: float = 0.5) -> str:
    with serial.Serial(port, baud, timeout=1, write_timeout=1) as cli_serial:
        cli_serial.write((command + "\n").encode("ascii"))
        time.sleep(delay)
        return cli_serial.read(4096).decode(errors="ignore")


def live_heatmap(points, view: str, grid_size: int = 80) -> np.ndarray:
    if view == "xy":
        a = np.array([point.x for point in points], dtype=np.float32)
        b = np.array([point.y for point in points], dtype=np.float32)
        a_range = (-3.0, 3.0)
        b_range = (0.0, 6.0)
    elif view == "yz":
        a = np.array([point.y for point in points], dtype=np.float32)
        b = np.array([point.z for point in points], dtype=np.float32)
        a_range = (0.0, 6.0)
        b_range = (-2.0, 2.0)
    elif view == "xz":
        a = np.array([point.x for point in points], dtype=np.float32)
        b = np.array([point.z for point in points], dtype=np.float32)
        a_range = (-3.0, 3.0)
        b_range = (-2.0, 2.0)
    elif view == "rv":
        a = np.array([(point.x * point.x + point.y * point.y + point.z * point.z) ** 0.5 for point in points], dtype=np.float32)
        b = np.array([point.velocity for point in points], dtype=np.float32)
        a_range = (0.0, 9.0)
        b_range = (-20.0, 20.0)
    else:
        raise ValueError(f"Unknown heatmap view: {view}")

    if len(points) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    snr = np.array([point.snr or 0 for point in points], dtype=np.float32)
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)
    a_edges = np.linspace(a_range[0], a_range[1], grid_size + 1)
    b_edges = np.linspace(b_range[0], b_range[1], grid_size + 1)
    a_idx = np.clip(np.searchsorted(a_edges, a, side="right") - 1, 0, grid_size - 1)
    b_idx = np.clip(np.searchsorted(b_edges, b, side="right") - 1, 0, grid_size - 1)
    valid = (a >= a_range[0]) & (a <= a_range[1]) & (b >= b_range[0]) & (b <= b_range[1])
    for ai, bi, value in zip(a_idx[valid], b_idx[valid], snr[valid]):
        heat[bi, ai] = max(heat[bi, ai], value)
    return heat


def strongest_point(points):
    if not points:
        return None
    return max(points, key=lambda point: point.snr or 0)


def closest_point(points):
    if not points:
        return None
    return min(points, key=lambda point: (point.x * point.x + point.y * point.y + point.z * point.z) ** 0.5)


def point_range(point) -> float:
    return (point.x * point.x + point.y * point.y + point.z * point.z) ** 0.5


def range_axis(profile_len: int, max_range_m: float = 9.04) -> np.ndarray:
    if profile_len <= 0:
        return np.array([], dtype=np.float32)
    return np.linspace(0.0, max_range_m, profile_len, endpoint=False, dtype=np.float32)


def range_doppler_image(values: list[int], range_bins: int = 256) -> np.ndarray:
    if not values:
        return np.zeros((32, range_bins), dtype=np.float32)
    arr = np.array(values, dtype=np.float32)
    if len(arr) % range_bins != 0:
        side = int(np.sqrt(len(arr)))
        if side > 0 and side * side == len(arr):
            return arr.reshape(side, side)
        return arr.reshape(1, -1)
    doppler_bins = len(arr) // range_bins
    return arr.reshape(doppler_bins, range_bins)


class RadarWorker(threading.Thread):
    def __init__(
        self,
        event_queue: queue.Queue,
        stop_event: threading.Event,
        cfg_port: str,
        data_port: str,
        cfg_baud: int,
        data_baud: int,
        cfg_file: Path,
        skip_config: bool,
        csv_path: Optional[Path],
    ):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.cfg_port = cfg_port
        self.data_port = data_port
        self.cfg_baud = cfg_baud
        self.data_baud = data_baud
        self.cfg_file = cfg_file
        self.skip_config = skip_config
        self.csv_path = csv_path
        self.rx_started = False

    def run(self) -> None:
        csv_file = None
        writer = None
        try:
            if not self.skip_config:
                self.event_queue.put(("status", "Sending config..."))
                send_config(self.cfg_port, self.cfg_baud, self.cfg_file, 0.05)
                time.sleep(1)

            if self.csv_path:
                csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
                writer = csv.DictWriter(
                    csv_file,
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

            self.event_queue.put(("status", f"Opening {self.data_port}..."))
            with serial.Serial(self.data_port, self.data_baud, timeout=0.1) as data_serial:
                buffer = bytearray()
                start_time = time.time()
                last_rx = start_time
                frames = 0
                points_total = 0
                self.event_queue.put(("status", "Streaming"))

                while not self.stop_event.is_set():
                    chunk = data_serial.read(4096)
                    if chunk:
                        last_rx = time.time()
                        self.rx_started = True
                        buffer.extend(chunk)
                    elif time.time() - last_rx > 5:
                        if self.rx_started:
                            self.event_queue.put(("warning", "No data for 5 seconds"))
                        else:
                            self.event_queue.put(
                                (
                                    "no_data",
                                    "Data port is open, but no radar stream arrived. "
                                    "Use Restart Sensor or uncheck Skip config and Start again.",
                                )
                            )
                        last_rx = time.time()

                    while not self.stop_event.is_set():
                        packet = find_next_packet(buffer)
                        if packet is None:
                            break

                        try:
                            frame = parse_packet(packet)
                        except ValueError as exc:
                            self.event_queue.put(("warning", f"Malformed packet: {exc}"))
                            continue

                        elapsed_s = time.time() - start_time
                        frames += 1
                        points_total += len(frame["points"])

                        if writer is not None:
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
                            csv_file.flush()

                        self.event_queue.put(("frame", elapsed_s, frame, frames, points_total))

        except Exception as exc:
            self.event_queue.put(("error", str(exc)))
        finally:
            if csv_file is not None:
                csv_file.close()
            self.event_queue.put(("stopped", None))


class RadarGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("mmWave Radar GUI")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[RadarWorker] = None

        self.frame_times = deque(maxlen=250)
        self.frame_counts = deque(maxlen=250)
        self.velocity_times = deque(maxlen=2500)
        self.velocities = deque(maxlen=2500)
        self.velocity_ranges = deque(maxlen=2500)
        self.point_history = deque(maxlen=5)

        self.cfg_port = tk.StringVar(value="COM13")
        self.data_port = tk.StringVar(value="COM14")
        self.cfg_baud = tk.StringVar(value="115200")
        self.data_baud = tk.StringVar(value="921600")
        self.cfg_file = tk.StringVar(value=str(DEFAULT_CFG))
        self.skip_config = tk.BooleanVar(value=True)
        self.csv_path = tk.StringVar(value="gui_live_points.csv")
        self.status = tk.StringVar(value="Idle")
        self.stats = tk.StringVar(value="frames=0 points=0")
        self.last_error = ""

        self._build_layout()
        self._refresh_ports()
        self.after(50, self._process_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="CLI").grid(row=0, column=0, sticky="w")
        self.cfg_port_combo = ttk.Combobox(controls, textvariable=self.cfg_port, width=9)
        self.cfg_port_combo.grid(row=0, column=1, padx=(4, 12))

        ttk.Label(controls, text="Data").grid(row=0, column=2, sticky="w")
        self.data_port_combo = ttk.Combobox(controls, textvariable=self.data_port, width=9)
        self.data_port_combo.grid(row=0, column=3, padx=(4, 12))

        ttk.Label(controls, text="CFG").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.cfg_file, width=62).grid(row=0, column=5, sticky="ew", padx=(4, 4))
        ttk.Button(controls, text="Browse", command=self._choose_cfg).grid(row=0, column=6, padx=(0, 8))
        ttk.Checkbutton(controls, text="Skip config", variable=self.skip_config).grid(row=0, column=7)

        ttk.Label(controls, text="CSV").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.csv_path, width=40).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(4, 12), pady=(6, 0))
        ttk.Button(controls, text="Save As", command=self._choose_csv).grid(row=1, column=4, sticky="w", pady=(6, 0))

        self.start_button = ttk.Button(controls, text="Start", command=self._start)
        self.start_button.grid(row=1, column=5, sticky="e", padx=(4, 4), pady=(6, 0))
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=6, sticky="w", padx=(4, 4), pady=(6, 0))
        self.restart_button = ttk.Button(controls, text="Restart Sensor", command=self._restart_sensor)
        self.restart_button.grid(row=1, column=7, sticky="w", padx=(4, 4), pady=(6, 0))
        ttk.Button(controls, text="Refresh Ports", command=self._refresh_ports).grid(row=1, column=8, sticky="w", pady=(6, 0))

        controls.columnconfigure(5, weight=1)

        status_bar = ttk.Frame(root)
        status_bar.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(status_bar, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Label(status_bar, textvariable=self.stats).pack(side=tk.RIGHT)

        self.figure = Figure(figsize=(11, 7.2), dpi=100)
        self.ax_xy = self.figure.add_subplot(231)
        self.ax_yz = self.figure.add_subplot(232)
        self.ax_range = self.figure.add_subplot(233)
        self.ax_rd = self.figure.add_subplot(234)
        self.ax_counts = self.figure.add_subplot(235)
        self.ax_info = self.figure.add_subplot(236)

        blank = np.zeros((80, 80), dtype=np.float32)
        self.xy_image = self.ax_xy.imshow(blank, origin="lower", aspect="auto", extent=(-3, 3, 0, 6), cmap="magma", vmin=0, vmax=400)
        self.ax_xy.set_title("XY sensor image")
        self.ax_xy.set_xlabel("x lateral (m)")
        self.ax_xy.set_ylabel("y forward (m)")

        self.yz_image = self.ax_yz.imshow(blank, origin="lower", aspect="auto", extent=(0, 6, -2, 2), cmap="magma", vmin=0, vmax=400)
        self.ax_yz.set_title("YZ sensor image")
        self.ax_yz.set_xlabel("y forward (m)")
        self.ax_yz.set_ylabel("z vertical (m)")

        (self.range_line,) = self.ax_range.plot([], [], linewidth=1.2, label="range")
        (self.noise_line,) = self.ax_range.plot([], [], linewidth=1.0, alpha=0.7, label="noise")
        self.ax_range.set_title("Range Profile")
        self.ax_range.set_xlabel("range (m)")
        self.ax_range.set_ylabel("magnitude")
        self.ax_range.grid(True, alpha=0.3)
        self.ax_range.legend(loc="upper right", fontsize=8)
        for ref_range in (0.5, 1.0, 1.5, 2.0):
            self.ax_range.axvline(ref_range, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        self.rd_image = self.ax_rd.imshow(blank[:32], origin="lower", aspect="auto", extent=(0, 9.04, -20.16, 20.16), cmap="magma")
        self.ax_rd.set_title("Range-Doppler TLV")
        self.ax_rd.set_xlabel("range (m)")
        self.ax_rd.set_ylabel("velocity (m/s)")

        (self.count_line,) = self.ax_counts.plot([], [], linewidth=1.5)
        self.ax_counts.set_title("Detected Points Per Frame")
        self.ax_counts.set_xlabel("elapsed time (s)")
        self.ax_counts.set_ylabel("points")
        self.ax_counts.grid(True, alpha=0.3)

        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(0.02, 0.95, "No frame yet", va="top", family="monospace")

        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _refresh_ports(self) -> None:
        ports = [port.device for port in list_ports.comports()]
        self.cfg_port_combo["values"] = ports
        self.data_port_combo["values"] = ports

    def _choose_cfg(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Config files", "*.cfg"), ("All files", "*.*")])
        if path:
            self.cfg_file.set(path)

    def _choose_csv(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path.set(path)

    def _start(self) -> None:
        if self.worker is not None:
            return

        ok, message = self._preflight()
        if not ok:
            messagebox.showerror("Preflight failed", message)
            self.status.set(message)
            return

        self._clear_data()
        self.stop_event.clear()
        csv_path = Path(self.csv_path.get()) if self.csv_path.get().strip() else None
        self.worker = RadarWorker(
            event_queue=self.event_queue,
            stop_event=self.stop_event,
            cfg_port=self.cfg_port.get(),
            data_port=self.data_port.get(),
            cfg_baud=int(self.cfg_baud.get()),
            data_baud=int(self.data_baud.get()),
            cfg_file=Path(self.cfg_file.get()),
            skip_config=self.skip_config.get(),
            csv_path=csv_path,
        )
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status.set("Starting")

    def _stop(self) -> None:
        if self.worker is None:
            return
        self.stop_event.set()
        self.status.set("Stopping")
        self.after(100, self._finish_stop)

    def _finish_stop(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.after(100, self._finish_stop)
            return
        self.worker = None
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status.set("Stopped")

    def _preflight(self) -> tuple[bool, str]:
        try:
            cfg_baud = int(self.cfg_baud.get())
            data_baud = int(self.data_baud.get())
        except ValueError:
            return False, "Baud rates must be integers."

        data_ok, data_message = try_open_port(self.data_port.get(), data_baud)
        if not data_ok:
            return (
                False,
                f"Cannot open data port {self.data_port.get()}: {data_message}\n\n"
                "Close Radar GUI, live viewer, record scripts, TI Visualizer, or any serial terminal using COM14.",
            )

        cli_ok, cli_message = probe_cli_prompt(self.cfg_port.get(), cfg_baud)
        if not cli_ok:
            return False, f"CLI check failed on {self.cfg_port.get()}: {cli_message}"

        if not self.skip_config.get() and not Path(self.cfg_file.get()).exists():
            return False, f"Config file does not exist: {self.cfg_file.get()}"

        self.status.set("Preflight OK")
        return True, "OK"

    def _restart_sensor(self) -> None:
        if self.worker is not None:
            messagebox.showinfo("Restart Sensor", "Stop streaming before restarting the sensor.")
            return

        ok, message = try_open_port(self.data_port.get(), int(self.data_baud.get()))
        if not ok:
            messagebox.showerror(
                "Data port busy",
                f"Cannot open {self.data_port.get()}: {message}\n\nClose any program using the data port first.",
            )
            return

        if not Path(self.cfg_file.get()).exists():
            messagebox.showerror("Missing config", f"Config file does not exist: {self.cfg_file.get()}")
            return

        self.start_button.configure(state=tk.DISABLED)
        self.restart_button.configure(state=tk.DISABLED)
        self.status.set("Restarting sensor...")
        thread = threading.Thread(target=self._restart_sensor_worker, daemon=True)
        thread.start()

    def _restart_sensor_worker(self) -> None:
        try:
            try:
                send_cli_command(self.cfg_port.get(), int(self.cfg_baud.get()), "sensorStop", delay=0.3)
            except Exception:
                pass
            send_config(self.cfg_port.get(), int(self.cfg_baud.get()), Path(self.cfg_file.get()), 0.05)
            self.event_queue.put(("restart_done", None))
        except Exception as exc:
            self.event_queue.put(("error", f"Restart Sensor failed: {exc}"))

    def _clear_data(self) -> None:
        self.frame_times.clear()
        self.frame_counts.clear()
        self.velocity_times.clear()
        self.velocities.clear()
        self.velocity_ranges.clear()
        self.point_history.clear()

    def _process_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.after(50, self._process_events)

    def _handle_event(self, event) -> None:
        kind = event[0]
        if kind == "status":
            self.status.set(event[1])
        elif kind == "warning":
            self.status.set(event[1])
        elif kind == "no_data":
            self.status.set("No data")
            messagebox.showwarning("No radar stream", event[1])
        elif kind == "error":
            self.status.set("Error")
            self.start_button.configure(state=tk.NORMAL)
            self.restart_button.configure(state=tk.NORMAL)
            messagebox.showerror("Radar GUI Error", event[1])
        elif kind == "stopped":
            self.worker = None
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            if self.status.get() != "Error":
                self.status.set("Stopped")
        elif kind == "restart_done":
            self.start_button.configure(state=tk.NORMAL)
            self.restart_button.configure(state=tk.NORMAL)
            self.skip_config.set(True)
            self.status.set("Sensor restarted. Click Start with Skip config checked.")
        elif kind == "frame":
            _, elapsed_s, frame, frames, points_total = event
            self._update_plots(elapsed_s, frame)
            self.stats.set(f"frames={frames} points={points_total}")

    def _update_plots(self, elapsed_s: float, frame: dict) -> None:
        points = frame["points"]
        self.point_history.append(points)
        self.frame_times.append(elapsed_s)
        self.frame_counts.append(len(points))

        recent_points = []
        for history_points in self.point_history:
            for point in history_points:
                recent_points.append(point)

        self.xy_image.set_data(live_heatmap(recent_points, "xy"))
        self.yz_image.set_data(live_heatmap(recent_points, "yz"))
        self._update_range_profile(frame)
        self._update_range_doppler(frame)

        self.count_line.set_data(list(self.frame_times), list(self.frame_counts))
        if self.frame_times:
            self.ax_counts.set_xlim(max(0, self.frame_times[0]), max(5, self.frame_times[-1]))
        max_count = max(self.frame_counts) if self.frame_counts else 1
        self.ax_counts.set_ylim(0, max(10, max_count * 1.2))

        self.info_text.set_text(
            "\n".join(
                self._info_lines(elapsed_s, frame)
            )
        )
        self.canvas.draw_idle()

    def _update_range_profile(self, frame: dict) -> None:
        profile = frame.get("range_profile", [])
        noise = frame.get("noise_profile", [])
        x = range_axis(len(profile))
        self.range_line.set_data(x, profile)
        if noise:
            self.noise_line.set_data(range_axis(len(noise)), noise)
        else:
            self.noise_line.set_data([], [])

        if len(profile) > 0:
            max_y = max(max(profile), max(noise) if noise else 0, 1)
            self.ax_range.set_xlim(0, 9.04)
            self.ax_range.set_ylim(0, max_y * 1.1)

    def _update_range_doppler(self, frame: dict) -> None:
        values = frame.get("range_doppler_heatmap", [])
        if values:
            image = range_doppler_image(values)
            self.rd_image.set_data(image)
            self.rd_image.set_extent((0, 9.04, -20.16, 20.16))
            self.rd_image.set_clim(vmin=float(np.min(image)), vmax=float(np.max(image) or 1.0))
        else:
            # Fallback view from detected points. This is not the raw TI heatmap,
            # but still gives a human-readable range/velocity summary.
            self.rd_image.set_data(live_heatmap(frame["points"], "rv")[:32])
            self.rd_image.set_extent((0, 9.0, -20.0, 20.0))

    def _info_lines(self, elapsed_s: float, frame: dict) -> list[str]:
        points = frame["points"]
        strong = strongest_point(points)
        close = closest_point(points)
        profile = frame.get("range_profile", [])
        stats = frame.get("stats", {})
        lines = [
            f"frame_number : {frame['frame_number']}",
            f"elapsed_s    : {elapsed_s:.2f}",
            f"points       : {len(points)}",
            f"num_tlvs     : {frame['num_tlvs']}",
        ]

        if strong is not None:
            lines.extend(
                [
                    "",
                    "strongest point",
                    f"  range : {point_range(strong):.2f} m",
                    f"  x/y/z : {strong.x:.2f}, {strong.y:.2f}, {strong.z:.2f}",
                    f"  v/snr : {strong.velocity:.2f}, {strong.snr}",
                ]
            )

        if close is not None:
            lines.extend(
                [
                    "",
                    "closest point",
                    f"  range : {point_range(close):.2f} m",
                    f"  x/y/z : {close.x:.2f}, {close.y:.2f}, {close.z:.2f}",
                ]
            )

        if profile:
            peak_idx = int(np.argmax(np.array(profile)))
            axis = range_axis(len(profile))
            lines.extend(
                [
                    "",
                    "range profile peak",
                    f"  range : {axis[peak_idx]:.2f} m",
                    f"  value : {profile[peak_idx]}",
                ]
            )

        if stats:
            lines.extend(
                [
                    "",
                    "stats",
                    f"  tx output us : {stats.get('transmit_output_time')}",
                    f"  active CPU   : {stats.get('active_frame_cpu_load')}",
                    f"  inter CPU    : {stats.get('inter_frame_cpu_load')}",
                ]
            )

        lines.extend(["", "Tip: put a metal object at 1m and look for the range-profile peak."])
        return lines

    def _on_close(self) -> None:
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        self.destroy()


def main() -> int:
    app = RadarGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
