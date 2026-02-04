# Telegram Export → NotebookLM

Convert a Telegram chat export (folder with `result.json` or `messages.html` + media) into Markdown for AI/NotebookLM. Messages stay in chronological order; media is labeled; voice and video are transcribed to text by default (Whisper) and inserted into the .md.

**Input:** Put the export folder (as from Telegram Desktop) in **src/** or pass its path. It must contain **result.json** (enable “Machine-readable JSON” when exporting) or **messages.html**. Media subfolders: `voice_messages/`, `round_video_messages/`, `video_files/`, `photos/`, etc.

**Output:** **dist/** — one or more `.md` files named after the export folder (e.g. `ChatExport_test_1.md`). Long chats are split by word count (NotebookLM ~500k words per source). While running, `dist/` also contains `export.log` and `.export_state.json` (checkpoint); the state file is removed when the export finishes successfully.

---

## How it works

- **export_to_md.py** reads `result.json` (or falls back to `messages.html`), iterates messages in order, and writes Markdown blocks: `### date | author` + text + media labels (`[Voice message: …]`, `[Video: …]`, etc.). Service messages, replies, forwards, polls are included. Progress is printed; state and log are saved in **dist/** so you can resume after interruption.
- By default, files from `voice_messages/`, `round_video_messages/` and `video_files/` are matched to messages by order and transcribed with Whisper; the text is inserted into the .md. Use `--no-transcribe` to disable.
- **split_json.py** is for plain JSON (no export folder): streams a large JSON array and splits it into .md or .json parts by size/word/object limits; used when you only have a raw .json file.

---

## Commands

### export_to_md.py — full export folder → .md

| Action | Command |
|--------|--------|
| Convert folder (default: `src/`) | `python3 export_to_md.py` |
| Convert specific folder | `python3 export_to_md.py src/MyExport` |
| Resume from checkpoint (default) | (automatic; state and log in `dist/`) |
| Start from scratch | `python3 export_to_md.py --no-resume` |
| Disable transcription | `python3 export_to_md.py --no-transcribe` |
| Progress log every N messages (default: 100) | `python3 export_to_md.py --progress-interval 500` |
| Custom output dir | `python3 export_to_md.py -o /path/to/out` |
| Max words per .md file | `python3 export_to_md.py --max-words 450000` |
| Clean `dist` folder | `python3 export_to_md.py clean` |

**Transcription:** Requires `pip install openai-whisper` and **ffmpeg** in PATH. On by default; if Whisper is missing, a warning is printed and export runs without transcription.

### split_json.py — raw JSON → split .md / .json

Supports resuming after interruption (state and log in `dist/`).

| Action | Command |
|--------|--------|
| All `.json` in `src/` → `dist/` | `python3 split_json.py` |
| One file | `python3 split_json.py src/file.json` |
| Start from scratch | `python3 split_json.py --no-resume` |
| Show config | `python3 split_json.py config --show` |
| Set limits | `python3 split_json.py config --max-words 450000 --array-path messages` |
| Clean `dist` or `src` | `python3 split_json.py clean dist` |
| JSON → plain .txt | `python3 split_json.py to-txt [input]` |

Config: **config.json** (`max_file_size_mb`, `max_objects_per_file`, `max_words_per_file`, `array_path`, `output_format`).

---

## Setup

**Requirements:** Python 3; **ffmpeg** in PATH (for transcription).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# For transcription (default):
pip install openai-whisper   # and install ffmpeg on the system
```

On macOS/Linux use `python3` (often `python` is not available).

---

## Limits

- **NotebookLM:** ~200 MB and ~500k words per source; 50 sources per notebook. Default split is 450k words per .md.
- **Telegram export schema:** [core.telegram.org/import-export](https://core.telegram.org/import-export)

---

## Performance: what takes time

Almost all runtime is **Whisper transcription**: each voice/video message is processed one after another; one file can take tens of seconds (length + CPU/GPU). Reading `result.json` is streamed (ijson), so it is not the bottleneck. Writing `.md` and reading media paths are negligible.

**Ways to speed up (current / future):**

| Option | Description |
|--------|-------------|
| **`--no-transcribe`** | Skip voice/video transcription; export finishes in minutes instead of hours. |
| **Smaller Whisper model** | Code uses `base` by default; switching to `tiny` (in code) is faster but less accurate. |
| **faster-whisper** (future) | Use `faster-whisper` (CTranslate2) instead of `openai-whisper` — same quality, much faster. |
| **Parallel transcription** (future) | Transcribe several files in parallel (e.g. multiprocessing) to use several CPU cores. |
| **Cache transcriptions** (future) | Store results per file; on re-run skip already transcribed files. |
