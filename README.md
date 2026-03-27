# GSoC-2026-Uramaki-Project
This is the repository for all the scripts files related to the Uramaki lab GSoC project " Voice/Speech analysis for emotion and cognitive load detection" . 

# 🎤 Voice & Speech Analysis for Emotion and Cognitive State Detection

## 📌 Overview

This project extends a sentiment-analysis pipeline by adding **voice-based analysis** from speech signals.

It extracts:

- Emotion (ML-based)
- Cognitive load
- Stress indicators
- Speech features
- Transcription with timestamps

---

## ✨ Features

### 🎶🎙️ Speech Analysis

- Energy (RMS)
- Zero Crossing Rate (ZCR)
- Silence ratio
- Dynamic range

### 😊 Emotion Recognition

- Model: wav2vec2 (HuggingFace)
- Output: label + confidence score

### 🧠 Cognitive Load

- Based on pause patterns and speech behavior

### 🤯 Stress Detection
- stress score, stress level.
- Normalized score (0–1)
- Interpretable levels (low, medium, high)

### 📝 Transcription

- Model: Whisper
- Includes timestamped chunks

---

## 🧩 Architecture

Audio → Whisper → Speech Features → Emotion Model → Cognitive + Stress → Output JSON
<img width="850" height="500" alt="New_Architecture" src="https://github.com/user-attachments/assets/8d3b09ae-725f-4490-ba75-b282999d9f52" />


---

## 📂 Project Structure

- `models/` → ML models
- `services/` → core logic
- `tests/` → demo scripts

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python tests/test_voice.py
```
