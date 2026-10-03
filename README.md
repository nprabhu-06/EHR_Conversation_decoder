# Doctor-Patient Consultation Transcription App

A Python CLI application that ingests audio of 2-person medical consultations, performs speaker diarization, transcribes speech to text, and outputs clean, role-labeled transcripts identifying each speaker as **Doctor** or **Patient**.

---

## Project Structure

```
ehr project/
├── main.py          # CLI entrypoint
├── transcriber.py   # Core diarization + transcription logic
├── config.py        # .env loading, engine config, startup validation
├── utils.py         # Audio helpers, timestamp formatting, output writers
├── evaluate.py      # DER + WER benchmarking against test_data/
├── requirements.txt # Python dependencies
├── .env.example     # Environment variable template
├── output/          # Generated transcripts (auto-created)
└── test_data/       # Add labeled audio + annotations here (see below)
```

---

## Quick Start

### 1. Install system dependency: FFmpeg

FFmpeg is required for audio format conversion. Install it first:

- **Windows**: Download from https://www.gyan.dev/ffmpeg/builds/ — add `bin/` to PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 2. Install Python dependencies

```bash
# PyTorch first (match your CUDA version at https://pytorch.org/get-started/locally/)
# For CPU-only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then the rest
pip install -r requirements.txt
```

### 3. Configure your API keys

```bash
cp .env.example .env
```

Edit `.env` and add:
- `ASSEMBLYAI_API_KEY` — get a free key at https://www.assemblyai.com/app/account
- `HUGGINGFACE_TOKEN` — get a token at https://huggingface.co/settings/tokens

  You must also accept the pyannote model terms on HuggingFace:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

---

## Usage

### Transcribe a consultation audio file

```bash
# Using AssemblyAI (cloud, fast, requires API key)
python main.py --audio consultation.mp3 --engine assemblyai

# Using pyannote + WhisperX (local, private, requires GPU recommended)
python main.py --audio consultation.mp3 --engine pyannote-whisperx
```

### Override speaker-role assignment

By default, the **first speaker heard** is assigned as "Doctor". To override:

```bash
python main.py --audio consultation.mp3 --engine assemblyai --doctor-speaker SPEAKER_A
```

For the pyannote engine, use `SPEAKER_00` or `SPEAKER_01` as the label.

### Run evaluation against test data

```bash
python main.py --evaluate --engine assemblyai
```

> **Note**: The `test_data/` loader is a placeholder. Add your labeled dataset to `test_data/` and share the folder structure so the loader can be implemented.

---

## Output Format

Each transcription produces two files in `output/`:

**JSON** (`output/<stem>_transcript.json`):
```json
[
  {
    "speaker_label": "Doctor",
    "text": "How are you feeling today?",
    "start_time": 12.34,
    "end_time": 14.56,
    "confidence": 0.97
  },
  {
    "speaker_label": "Patient",
    "text": "I've had a headache for three days.",
    "start_time": 15.10,
    "end_time": 17.88,
    "confidence": 0.94
  }
]
```

**Plain text** (`output/<stem>_transcript.txt`):
```
[00:00:12] Doctor: How are you feeling today?
[00:00:15] Patient: I've had a headache for three days.
```

---

## Engine Comparison

| Feature | AssemblyAI | pyannote + WhisperX |
|---|---|---|
| Diarization | Built-in | pyannote.audio 3.1 |
| ASR | AssemblyAI ASR | Whisper (medium.en default) |
| Word boost | ✅ Medical vocabulary | ✗ (Whisper handles natively) |
| Cloud required | ✅ Yes | ✗ Fully local |
| GPU required | ✗ No | Recommended |
| HIPAA-friendliness | Requires BAA with AssemblyAI | ✅ Self-hosted, data stays local |
| Speed | Fast (cloud) | Slower on CPU |

---

## Architecture

```
AudioFile
    │
    ▼
utils.ensure_wav()          ← pydub + ffmpeg: normalize to 16 kHz mono WAV
    │
    ▼
TranscriptionEngine.transcribe()
    ├── AssemblyAIEngine    ← AssemblyAI SDK: diarization + ASR in one call
    └── PyannoteWhisperXEngine
            ├── pyannote.audio 3.1   → speaker segments (SPEAKER_00/01)
            ├── whisperx             → word-level ASR + alignment
            └── merge step           → words assigned to speaker segments
    │
    ▼
TranscriptionEngine.map_speakers_to_roles()
    └── SPEAKER_A/SPEAKER_00 → "Doctor" | "Patient"
    │
    ▼
utils.write_json_transcript()
utils.write_txt_transcript()
```

---

## Toward HIPAA Compliance

This app is architected with HIPAA-conscious development in mind:

- **Use the pyannote-whisperx engine** for fully offline, self-hosted processing — audio never leaves your infrastructure.
- **AssemblyAI** requires a Business Associate Agreement (BAA) with AssemblyAI before use with PHI. See https://www.assemblyai.com/hipaa.
- **At-rest encryption**: Encrypt the `output/` directory containing transcripts.
- **Audit logging**: Add structured logging (Python `logging` module or similar) to `main.py` for access trails.
- **Minimum necessary**: Output only the fields required for the use case; avoid storing raw audio alongside transcripts unless needed.

---

## Evaluation (DER + WER)

The `evaluate.py` module scores transcriptions against labeled ground truth:

- **DER** (Diarization Error Rate) — pyannote.metrics
- **WER** (Word Error Rate) — jiwer

### Adding your test dataset

1. Place your labeled audio and annotation files in `test_data/`.
2. Share the folder structure (file names, annotation format: RTTM, JSON, CSV, TextGrid).
3. The `load_test_dataset()` function in `evaluate.py` will be implemented to match.

### Expected output

```
evaluation_report.txt:

Doctor-Patient Transcription — Evaluation Report
============================================================
Engine          : AssemblyAIEngine
Cases evaluated : 10
Average DER     : 0.0823 (8.2%)
Average WER     : 0.1140 (11.4%)

Per-Case Results
------------------------------------------------------------
  consult_001                     DER=0.043  WER=0.071
  consult_002                     DER=0.121  WER=0.158
  ...
```
