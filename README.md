<div align="center">
<!-- <img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" /> -->
</div>

# AgroGuard AI - Offline-First Crop Management

This application uses a local AI engine (YOLOv11 + ResNet) to analyze crop health without requiring an internet connection.

## Run Locally

### 1. Start the AI Backend (Python)
Requirements: Python 3.8+, pip

```bash
cd backend
python3 -m pip install -r requirements.txt
python3 main.py
```

### 2. Start the Frontend (React)
Requirements: Node.js

```bash
npm install
npm run dev
```

## Features
- **YOLOv11 Detection**: Precise plant localizing.
- **Neural Classification**: ResNet50-based disease identification.
- **Offline First**: All data and processing stay on your device.
