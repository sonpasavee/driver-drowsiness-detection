# Driver Drowsiness Detection System

ระบบตรวจจับอาการง่วงขณะขับขี่แบบ Real-time ด้วย Computer Vision และ MediaPipe สำหรับวิเคราะห์พฤติกรรมผู้ขับขี่ผ่าน Facial Landmarks, Eye Closure, Yawn Detection และ Head Pose Analysis พร้อมระบบ Scoring, State Machine, Alerting และ Session Telemetry Logging

---

## Features

### Real-time Face Analysis
- ตรวจจับใบหน้าและ Facial Landmarks ด้วย MediaPipe
- วิเคราะห์การปิดตา (EAR / PERCLOS)
- ตรวจจับการหาว (MAR)
- วิเคราะห์ทิศทางศีรษะ (Head Pose Estimation)

### Drowsiness Intelligence
- Weighted Drowsiness Scoring
- State Machine สำหรับจัดการสถานะผู้ขับขี่
- Threshold + Hysteresis Handling
- Alert Cooldown Mechanism

### Production Runtime
- Config-Driven Runtime (`configs/system.yaml`)
- Camera Auto-Reconnect
- Runtime Reset Support
- Structured Logging
- Session Telemetry Recording

### Monitoring & Alerting
- HUD Overlay แบบ Real-time
- Drowsiness Score Display
- State Visualization
- Sound Alert Support

---

## System Architecture

```text
Camera Input
     │
     ▼
Face Landmark Detection
(MediaPipe)
     │
     ▼
Analyzer Layer
 ├── Eye Closure (EAR / PERCLOS)
 ├── Yawn Detection (MAR)
 ├── Head Pose
 └── Score Aggregation
     │
     ▼
State Machine
(Alert / Warning / Safe)
     │
     ▼
HUD + Alerting + Telemetry Logging
```

---

## Project Structure

```text
driver-drowsiness-detection/

├── alerts/                # Alert backends
│
├── analyzers/             # Feature analyzers
│   ├── ear.py
│   ├── mar.py
│   ├── head_pose.py
│   └── score.py
│
├── capture/               # Camera abstraction layer
│
├── configs/               # Runtime configuration
│   └── system.yaml
│
├── core/                  # State machine logic
│
├── detectors/             # Face landmark detection
│
├── storage/               # Session telemetry writer
│
├── utils/                 # Config / logger helpers
│
├── assets/
│   └── face_landmarker_v2.task
│
├── logs/
│
└── main.py                # Application entrypoint
```

---

## Requirements

### Software

- Python 3.12+
- Windows / Linux / macOS
- Webcam

### Recommended Environment

- Windows PowerShell
- Virtual Environment (venv)

---

## Installation

### 1. Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd driver-drowsiness-detection
```

---

### 2. Create Virtual Environment

PowerShell:

```powershell
python -m venv env
```

CMD:

```cmd
python -m venv env
```

---

### 3. Activate Virtual Environment

PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

CMD:

```cmd
env\Scripts\activate.bat
```

หาก PowerShell ไม่อนุญาตให้ Activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

---

### 4. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Application

### Standard Run

```powershell
python .\main.py
```

หรือ

```cmd
python main.py
```

---

### Run Without Activation

```powershell
.\env\Scripts\python.exe .\main.py
```

---

## Runtime Controls

| Key | Action |
|-----|-----|
| `Q` | Exit application |
| `R` | Reset analyzers + state machine |

---

## Configuration

ระบบใช้ Runtime Configuration ผ่านไฟล์:

```text
configs/system.yaml
```

### Important Sections

| Section | Description |
|----------|-------------|
| `camera` | Camera index, FPS, resolution |
| `detector` | MediaPipe model + confidence |
| `analyzer` | Thresholds, smoothing, calibration |
| `scoring` | Weighted score configuration |
| `state_machine` | Threshold / cooldown / hysteresis |
| `logging` | Logger settings |
| `alerting` | Alert overlay + sound |
| `app` | Window / telemetry options |

---

## Logging & Telemetry

### Application Log

```text
logs/driver_monitor.log
```

### Session Telemetry

```text
logs/sessions/
```

Generated files:

```text
session_YYYYMMDD_HHMMSS.jsonl
session_YYYYMMDD_HHMMSS_summary.json
```

---

### JSONL Event Types

Telemetry รองรับ Event หลักดังนี้:

- `frame`
- `state_change`
- `manual_reset`

ตัวอย่าง:

```json
{
  "event":"state_change",
  "from":"SAFE",
  "to":"DROWSY",
  "timestamp":"2026-05-25T14:12:08"
}
```

---

### Frame Logging

เปิด Frame Snapshot Logging ได้ผ่าน config:

```yaml
app:
  frame_log_enabled: true
  frame_log_interval_frames: 30
```

---

## Production Notes

- `main.py` คือ Supported Entrypoint สำหรับระบบเต็ม
- Analyzer modules สามารถรันแยกเพื่อ Debug ได้
- Runtime Settings ถูกควบคุมผ่าน YAML Config
- Alert Sound ใช้ `winsound` บน Windows
- Session Telemetry เขียนต่อเนื่องระหว่าง Runtime

---

## Troubleshooting

### ModuleNotFoundError: No module named 'cv2'

มักเกิดจากการไม่ได้ใช้ Virtual Environment

ใช้คำสั่ง:

```powershell
.\env\Scripts\python.exe .\main.py
```

---

### Camera Works But No Face Detection

ตรวจสอบ:

- Camera framing
- แสงสว่าง
- ใบหน้าอยู่ในมุมกล้อง
- Asset file มีอยู่จริง

```text
assets/face_landmarker_v2.task
```

---

### PowerShell Activation Error

ใช้:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## Recommended Development Workflow

```powershell
.\env\Scripts\Activate.ps1

pip install -r requirements.txt

python .\main.py
```

---

## Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- YAML Configuration
- Structured Logging

---

## Future Improvements

- Multi-Face Support
- Driver Identity Tracking
- REST API / Dashboard Integration
- Cloud Telemetry Upload
- Model-Based Drowsiness Classification
- Docker Deployment

---

## License

This project is intended for educational, research, and development purposes.
