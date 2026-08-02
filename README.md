# Ward

A lightweight, real-time Python application for detecting human faces directly from your desktop screen. It combines custom screen region capture with the **InsightFace (Buffalo_L)** deep learning model to perform fast, accurate face detection and visual bounding-box tracking on live screen content.

---

## Features

* **Custom Region Capture:** Dynamically retrieves screen dimensions via `screeninfo` and captures specific display regions (default: central 50%).
* **InsightFace Integration:** Powered by InsightFace's `buffalo_l` model for highly accurate face detection.
* **Apple Silicon Optimized:** Pre-configured with `CoreMLExecutionProvider` for accelerated performance on macOS devices (configurable for CUDA/CPU).
* **Live Visual Feedback:** Draws real-time bounding boxes around detected faces using OpenCV.

---

## Project Structure

```text
.
├── main.py            # Main entry point and frame loop
├── face_detector.py   # InsightFace model wrapper
└── screen_capture.py  # Screen grabbing helper using PIL
