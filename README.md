# SIGN_LANGUAGE_DETECTION
# 🤟 Sign Language Detection with Offline Voice Integration

This project is a real-time Sign Language Detection system that uses **MediaPipe**, **machine learning**, and an **offline text-to-speech (TTS) engine** to convert hand gestures into spoken words. Designed to be simple, editable, and powerful, it allows users to train, test, and expand the gesture recognition system without internet connectivity.

## 🔧 Features

- 🖐️ Real-time hand gesture detection using MediaPipe
- 🧠 Keypoint and pointer-based gesture classification
- 🗣️ Offline voice integration for speech output
- 📝 Easy-to-edit CSV files for adding new data
- ⌨️ Simple key controls:
  - `h` – Hand-only detection
  - `k` – Keypoint data recording
  - `n` – Pointer history recording

## 🧪 How It Works

1. **Detection** – Uses your webcam to detect hand landmarks in real-time.
2. **Classification** – Matches gestures to a trained dataset using keypoint history or pointer movement.
3. **Voice Output** – Detected gesture is spoken aloud using an offline TTS engine.
4. **Customization** – You can press keys to record new data and retrain the model easily.

## 🗂️ Project Structure

