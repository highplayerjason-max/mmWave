# mmWave UART Logger

This workspace contains a small Python logger for the TI mmWave SDK Out-of-Box
Demo UART stream.

## Setup

Close mmWave Demo Visualizer first. COM ports cannot be opened by Visualizer and
Python at the same time.

Install the only runtime dependency:

```powershell
pip install pyserial
```

## Run

Send a saved Visualizer or SDK `.cfg` file to `COM13`, read point-cloud TLVs from
`COM14`, print frames, and save CSV:

```powershell
python .\mmwave_uart_logger.py --cfg-file C:\path\to\profile.cfg --out radar_points.csv
```

If the radar is already configured and running, skip the config step:

```powershell
python .\mmwave_uart_logger.py --skip-config --out radar_points.csv
```

Stop after a fixed number of frames:

```powershell
python .\mmwave_uart_logger.py --cfg-file C:\path\to\profile.cfg --out radar_points.csv --frames 100
```

The CSV columns are:

```text
timestamp, frame_number, point_id, x, y, z, velocity, snr, noise
```

## Notes

This parser targets the SDK Out-of-Box Demo style stream:

- magic word: `02 01 04 03 06 05 08 07`
- TLV type `1`: detected points as `x, y, z, velocity` float32 values
- TLV type `7`: side info as `snr, noise` uint16 values

If you flashed a different lab, such as people counting or vital signs, its TLV
format may differ and the parser may need changes.

## Radar-Only Proposal Workflow

Record a structured radar session. If the sensor is already running, use
`--skip-config`; otherwise omit it and the default xwr68xx `profile_3d.cfg` will
be sent first.

```powershell
python .\record_radar_session.py --skip-config --seconds 30
```

Each session is written under `sessions\session_YYYYMMDD_HHMMSS`:

```text
metadata.json
frames.csv
points.csv
```

Generate radar-only features for model input:

```powershell
python .\radar_features.py .\sessions\session_YYYYMMDD_HHMMSS\points.csv --window-s 0.5
```

This produces `radar_features.csv`, with per-window summary features and simple
range/velocity histograms. It is the first low-cost approximation of a temporal
radar token.

Generate OmniVLA/MuseVLA-style radar sensor images:

```powershell
python .\radar_image.py .\sessions\session_YYYYMMDD_HHMMSS\points.csv
```

This produces:

```text
radar_images\radar_xy.png
radar_images\radar_yz.png
radar_images\radar_xz.png
radar_images\radar_range_velocity.png
radar_images\radar_tensor.npz
```

`radar_tensor.npz` has shape:

```text
(views, channels, height, width)
```

with views `xy`, `yz`, `xz`, `range_velocity` and channels `count`, `max_snr`,
`mean_snr`, `mean_velocity`.

Generate diagnostic plots:

```powershell
python .\visualize_radar_session.py .\sessions\session_YYYYMMDD_HHMMSS
```

Plots are written under the session's `plots` directory:

```text
xy_point_cloud.png
points_per_frame.png
velocity_over_time.png
```

## Live Viewer

Show a live demo-style radar view while reading `COM14`:

```powershell
python .\live_radar_viewer.py --skip-config --out live_points.csv
```

If the board has just been reset and needs configuration:

```powershell
python .\live_radar_viewer.py --out live_points.csv
```

The live viewer shows:

- top-down XY point cloud, colored by SNR
- detected point count over time
- velocity over time, colored by range
- current frame status

The window can be closed to stop. Add `--seconds 30` or `--frames 300` for a
fixed run.

## Desktop GUI

Launch the desktop GUI:

```powershell
python .\radar_gui.py
```

The GUI provides:

- CLI/data port selection
- config file selection
- start/stop controls
- `Restart Sensor` for `sensorStop + full profile_3d.cfg + sensorStart`
- preflight checks for busy data port and missing CLI prompt
- optional CSV logging
- live XY, YZ, XZ, and range-velocity radar sensor images
- point-count plot and frame status

Use `Skip config` when the radar is already running. Uncheck it after a board
reset if you want the GUI to send `profile_3d.cfg` before streaming.

If the GUI says the data port is open but no stream arrived, use
`Restart Sensor`, then click `Start` again with `Skip config` checked.
