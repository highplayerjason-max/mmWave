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
- live range profile, with 0.5m/1.0m/1.5m/2.0m reference lines
- range-Doppler TLV view when enabled, with a point-derived fallback view
- aligned-style mmWave pseudo-camera view with black background and Gaussian blobs
- clustered object candidates overlaid on the XY view with distance/side labels
- strongest/closest target summaries, object summaries, point-count plot, and frame status

Use `Skip config` when the radar is already running. Uncheck it after a board
reset if you want the GUI to send `profile_3d.cfg` before streaming.

If the GUI says the data port is open but no stream arrived, use
`Restart Sensor`, then click `Start` again with `Skip config` checked.

To try the real TI range-Doppler TLV instead of the point-derived fallback,
select this config in the GUI and use `Restart Sensor`:

```powershell
profiles\profile_3d_range_doppler.cfg
```

This enables:

```text
guiMonitor -1 1 1 1 0 1 1
```

It is more interpretable, but it uses more UART bandwidth.

To get a denser no-camera mmWave image, select this config instead:

```powershell
profiles\profile_3d_azimuth_heatmap.cfg
```

This enables:

```text
guiMonitor -1 1 1 1 1 0 1
```

The fifth GUI panel will then use TLV type `4`
(`AZIMUT_STATIC_HEAT_MAP`) and render a dense range-azimuth heatmap from the
virtual antenna complex data. This is closer to paper-style no-camera mmWave
images than the point-cloud fallback.

### Making Sense Of Where Objects Are

The raw TI OOB point cloud is not an RGB-like picture. Each point is a strong
peak in range-Doppler-angle space, so many real objects will not appear as a
solid shape. To make the output easier to understand, the GUI now groups recent
detected points into coarse object candidates:

- `#1`, `#2`, ... cyan circles in the XY view are clustered candidates.
- `x` is left/right in meters, `y` is forward distance in meters, and `z` is
  vertical height in meters.
- The info panel lists range, left/right offset, azimuth angle, height, mean
  velocity, point count, and peak SNR for each candidate.

This follows the practical lesson from OmniVLA/MuseVLA-style systems: do not
feed or inspect sparse raw radar points directly. First convert the sensor
measurement into a spatially grounded image or mask. In this project, the
current no-RGB version is:

```text
TI point cloud / range profile / range-Doppler TLV
    -> XY/YZ/range-velocity sensor images
    -> aligned-style pseudo-camera heatmap
    -> clustered object candidates
    -> optional saved radar images/tensors for later model training
```

The pseudo-camera view treats the antenna like a low-resolution radar camera:

- when TLV 4 is available, horizontal position comes from an antenna FFT over
  azimuth angle and vertical position is range
- when TLV 4 is not available, the GUI falls back to detected points expanded
  into Gaussian blobs
- brightness comes from radar return strength
- the point fallback is only a visualization; TLV 4 is the better data source

Without RGB calibration, this view is not a real camera overlay. It is a
camera-like rendering of radar coordinates.

## Replicate RGB-Aligned Demo Figure

The script `replicate_aligned_mmwave.py` reproduces the lightweight alignment
workflow used by the RGB/mmWave demo repository:

```powershell
python .\replicate_aligned_mmwave.py `
  --points-a .\radar_points.csv `
  --title-a "Current radar sample" `
  --mm-anchor 1.03,0.66 `
  --mm-anchor 0.02,0.52 `
  --px-anchor 505,455 `
  --px-anchor 890,385 `
  --out-dir .\outputs\replicate_test
```

The anchors are manually chosen correspondences. The script computes the
rotation, scale, and translation, then warps a Gaussian KDE mmWave heatmap into
`1280 x 720` image coordinates. Add `--rgb-a image.jpg` to put an RGB frame on
the left side. Add `--points-b`, `--rgb-b`, and `--title-b` to generate the
two-scene comparison and positive-difference panel.
