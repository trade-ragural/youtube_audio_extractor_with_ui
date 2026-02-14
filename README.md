# Audio-to-Text Extractor

Extracts speech from YouTube videos or uploaded video files and transcribes it using Google Gemini.
Supports Kannada + auto language detection. Outputs chunked transcriptions with synchronized video playback.

## Prerequisites

- Python 3.9+
- `ffmpeg` installed and on PATH

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Opens a Gradio UI at `http://localhost:7860`.

## Usage

1. Choose **YouTube Link** or **Upload Video**
2. Set the **chunk duration** (seconds) — each chunk gets independently transcribed
3. Click **Process Video**
4. View each chunk with its video player, timestamp, and transcribed text
