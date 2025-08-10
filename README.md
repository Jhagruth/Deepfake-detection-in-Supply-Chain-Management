# Deepfake Detection in Supply Chain Management

**Multi-modal deepfake & forgery detection suite** — face, voice, and receipt/document analysis designed for supply-chain scenarios (transaction verification, secure checkouts, identity verification, etc.).

---

## Contents (quick)
- `app.py` — Streamlit UI (Face / Receipt / Voice modes)
- `face_run.py` — Face deepfake detection via Roboflow ensemble
- `download_dataset_*` — script to download Hugging Face deepfake image dataset
- `receipt.py` — Receipt/document forensic analysis using Google Gemini
- `synthetic_receipts.py` — Generates forged receipts for training/augmentation
- `voice_train.py` — Train audio deepfake detector (ASVspoof-2019)
- `voice_run.py` — Run audio inference with saved model `deepfake_detector.keras`
- `deepfake_detector.keras` — Trained audio-model (binary; not included here)

---

## Project overview
This repository combines signal/image forensics and ML to detect falsified media across three channels:

- **Face**: Uses OpenCV face detection + an ensemble of Roboflow models for image/video deepfake detection.
- **Receipt**: Uses a forensic prompt sent to Google Gemini to analyze receipts / PDFs for signs of manipulation. Also includes synthetic forgery generation to augment training data.
- **Voice**: Trains and runs an audio-based detection model using log-Mel spectrograms (ASVspoof dataset).

---

## Requirements & quick setup

### System prerequisites
- Python 3.8+  
- `pip` / virtualenv  
- **Poppler** (for PDF → image):  
  - macOS: `brew install poppler`  
  - Debian/Ubuntu: `sudo apt-get install poppler-utils`  
  - Windows: download Poppler and add `bin/` to PATH  
- **Tesseract-OCR** (for synthetic receipt editing):  
  - macOS: `brew install tesseract`  
  - Debian/Ubuntu: `sudo apt-get install tesseract-ocr`  
  - Windows: install from UB Mannheim builds

### Python dependencies (suggested)
Create venv and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install streamlit opencv-python numpy Pillow requests pdf2image librosa tensorflow datasets pytesseract tqdm scikit-learn kagglehub inference_sdk
