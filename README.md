# Driver Drowsiness Detection

Real-time driver drowsiness monitoring with:
- facial landmarks via MediaPipe
- eye closure / PERCLOS analysis
- yawn detection
- head pose analysis
- weighted drowsiness scoring
- state machine with alerting and session logging

## Features

- Production-style `main.py` entrypoint
- Config-driven runtime via `configs/system.yaml`
- Camera reconnect support
- On-screen HUD for score, state, and analyzer outputs
- Alert handling with sound support
- JSONL session telemetry under `logs/sessions/`
- Rotating application logs

## Project Structure

```text
alerts/       Alert backends
analyzers/    EAR, MAR, head pose, score aggregation
capture/      Camera access
configs/      System configuration
core/         State machine
detectors/    Face landmark detection
storage/      Session telemetry writer
utils/        Config and logging helpers
main.py       Main application entrypoint
```

## Requirements

- Windows PowerShell recommended
- Python 3.12+ recommended
- Webcam

This repository already includes the MediaPipe model asset at:

```text
assets/face_landmarker_v2.task
```

## Setup

### 1. Create virtual environment

```powershell
python -m venv env
```

```cmd
python -m venv env
```

### 2. Activate virtual environment

```powershell
.\env\Scripts\Activate.ps1
```

```cmd
env\Scripts\activate.bat
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Run the full application:

```powershell
python .\main.py
```

```cmd
python main.py
```

Or without activating first:

```powershell
.\env\Scripts\python.exe .\main.py
```

```cmd
env\Scripts\python.exe main.py
```

## Runtime Controls

- `Q`: quit application
- `R`: reset analyzers and state machine

## Output Files

- App log: `logs/driver_monitor.log`
- Session metrics: `logs/sessions/session_YYYYMMDD_HHMMSS.jsonl`
- Session summary: `logs/sessions/session_YYYYMMDD_HHMMSS_summary.json`

Each JSONL line is a structured record such as:
- `frame`
- `state_change`
- `manual_reset`

Notes:
- `.jsonl` session files are machine-readable telemetry logs, not human-friendly reports.
- `_summary.json` is the human-readable session file.
- By default, the app now logs only important events.
- To also log periodic frame snapshots, set `app.frame_log_enabled: true`.
- Control frame log frequency with `app.frame_log_interval_frames`.

## Configuration

Main runtime config lives in:

```text
configs/system.yaml
```

Important sections:

- `camera`: index, resolution, FPS, reconnect behavior
- `detector`: MediaPipe confidence and model path
- `analyzer`: thresholds, calibration windows, smoothing
- `scoring`: weighted score behavior
- `state_machine`: score thresholds, hysteresis, cooldowns
- `logging`: log level and rotating file settings
- `alerting`: sound/overlay controls
- `app`: window name, FPS overlay, session telemetry

## Production Notes

- `main.py` is the supported entrypoint for running the integrated system.
- The analyzer demo files in `analyzers/`, `core/`, and `detectors/` are still useful for isolated debugging.
- Logger settings are read from `configs/system.yaml`.
- Alert sound uses `winsound` on Windows and terminal bell fallback elsewhere.
- Session telemetry is written continuously, so long runs can be audited after execution.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

You are likely not using the project virtual environment.

Use:

```powershell
.\env\Scripts\python.exe .\main.py
```

### Camera opens but no landmarks are detected

- Check camera framing and lighting
- Ensure the face is visible during calibration
- Verify `assets/face_landmarker_v2.task` exists

### App starts but PowerShell cannot activate the environment

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

## Recommended Development Flow

```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
python .\main.py
```

```cmd
env\Scripts\activate.bat
pip install -r requirements.txt
python main.py
```
