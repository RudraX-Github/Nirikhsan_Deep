🛡️ Niraakshan (निराक्षण) | Intelligent Guard Monitoring System

"निराक्षण करे रक्षण" (Observe to Protect)

Niraakshan is a state-of-the-art AI surveillance suite designed for real-time security guard tracking and action monitoring. It combines computer vision, pose estimation, and face recognition to ensure guards are active, alert, and performing their duties.

📹 Demo https://github.com/RudraX-Github/Nirikhsan_Deep/blob/main/Sunday%20Laggon/Testing.mp4

Watch the Demo Video

🚀 Key Features

🧠 AI Core

Multi-Target Tracking: Tracks multiple guards simultaneously using identifying bounding boxes.

Pose Estimation (MediaPipe): Detects skeletal landmarks to classify actions (Standing, Sitting, Hands Up, T-Pose).

Face Recognition: Robust identification using face_recognition (dlib) with multi-angle onboarding (Front, Left, Right, Back).

Fugitive Detection: "Blacklist" mode to detect specific unauthorized individuals and trigger immediate alarms.

Anti-Ghosting Logic: Advanced filtering to prevent tracking drift or tracking inanimate objects.

🛠️ Operational Capabilities

Night Mode: Adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light environments.

Multi-Language UI: Real-time switching between Hindi, English, Marathi, and Gujarati.

RTSP/IP Camera Support: dedicated ThreadedIPCamera class for smooth, non-blocking streaming from security cameras.

Stillness Detection (Pro Mode): Alerts if a guard remains motionless for too long (e.g., sleeping).

📊 Reporting & Logging

Automated Snapshots: Captures evidence photos upon Alert Triggers or Fugitive Detection.

CSV Event Logging: detailed logs in logs/events.csv containing timestamps, confidence scores, and actions.

Performance Monitor: Real-time FPS and Memory usage display.

🏗️ System Architecture

The system operates on a hybrid pipeline optimized for CPU inference:

graph TD
    A[Input Source] -->|USB/RTSP| B(Threaded Frame Grabber)
    B --> C{Lighting Condition?}
    C -- Low Light --> D[CLAHE Enhancement]
    C -- Normal --> E[Standard RGB]
    D --> F[Face Detection]
    E --> F
    F --> G[Re-ID & Face Recognition]
    G --> H[MediaPipe Pose Estimation]
    H --> I[Action Classifier Logic]
    I --> J{Alert Rules}
    J -- Violation --> K[Audio Alarm & Snapshot]
    J -- Normal --> L[UI Overlay]


⚙️ Installation

Prerequisites

Python 3.10+

Visual C++ Redistributable (for Windows)

CUDA Toolkit (Optional, for GPU acceleration)

1. Clone the Repository

git clone [https://github.com/yourusername/niraakshan.git](https://github.com/yourusername/niraakshan.git)
cd niraakshan


2. Set up Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


Note: If dlib fails to install, ensure you have CMake installed or download a pre-compiled .whl file.

4. File Structure Setup

The application requires specific folders to function. The script creates them automatically, but ensure your structure looks like this:

Niraakshan/
├── Niraskhan_Done.py       # Main Executable
├── config.json             # Configuration
├── audio_files/            # Place 'siren.mp3' here
├── guard_profiles/         # Database of registered faces
├── alert_snapshots/        # Auto-saved violation images
├── capture_snapshots/      # Manual snapshots
└── logs/                   # CSV Logs


🔧 Configuration

Configure the system behavior by editing config.json.

{
  "detection": {
    "min_detection_confidence": 0.5,
    "face_recognition_tolerance": 0.52,
    "re_detect_interval": 5
  },
  "alert": {
    "default_interval_seconds": 10,
    "alert_cooldown_seconds": 2.5
  },
  "performance": {
    "pose_buffer_size": 12,
    "enable_frame_skipping": true
  },
  "storage": {
    "snapshot_retention_days": 30
  }
}


🎯 Usage Guide

Starting the System

Run the main script:

python Niraskhan_Done.py


Onboarding a Guard

Click ➕ Add Guard in the sidebar.

Enter the Guard's Name.

The system will prompt for 4 angles. Follow the on-screen green box:

Front: Look at camera.

Left: Turn 90° Left.

Right: Turn 90° Right.

Back/Top: (Optional for Pro Mode).

The profile is saved to guard_profiles/.

Monitoring

Select Guards: Click "Select Guard" and check the boxes for guards on duty.

Start Tracking: Click "🎯 Track Guard".

Set Alert: Use the dropdown to select the required action (e.g., "Hands Up"). If the guard does not perform this action within the timeout, an alarm sounds.

⚠️ Known Limitations & Best Practices

RTSP Streams: Using multiple high-resolution RTSP streams may cause high CPU usage. It is recommended to use the sub-stream (lower resolution) from IP cameras.

Lighting: While Night Mode helps, ensure the environment has at least minimal lighting for accurate Pose Estimation.

Windows Console: If you see weird characters in the terminal, it's due to Emoji rendering. The app uses a SafeLogger to handle this, but using Windows Terminal is recommended over CMD.

🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License

Distributed under the MIT License. See LICENSE for more information.

📞 Contact & Support

Project Lead: (Your Name)
Tech Stack: Python, CustomTkinter, OpenCV, MediaPipe
