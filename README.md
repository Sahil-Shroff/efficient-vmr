# Video-MMLU Subtitle-Based VMR Baselines

This repo is organized around **Video Moment Retrieval (VMR)** / **timestamp localization**, not general video QA.

Primary task:

- given a question and its source `video_id`
- retrieve the most relevant subtitle window
- predict a timestamp span `(start_time, end_time)`

The current baseline is deliberately simple:

- download YouTube subtitles for each dataset video
- parse them into timestamped subtitle segments
- build fixed temporal windows
- run BM25 over subtitle-window text
- use the top retrieved window as the predicted span

Legacy QA-oriented transcript answering remains in the repo, but it is isolated under [`src/legacy`](./src/legacy) and is no longer the main path.

## Project Focus

- **Primary**: subtitle-first temporal retrieval for VMR
- **Near-term extensions**: OCR, keyframes, multimodal reranking, routing
- **Secondary / optional**: QA-style answer generation experiments

## Directory Structure

```text
config/
  example.env
data/
  video_ids.txt
  subtitles/
    raw/
    parsed/
    windows/
    subtitle_manifest.jsonl
scripts/
  setup_vm.sh
  run_vmr_baseline.sh
  run_transcript_baseline.sh
src/
  data/
    videommlu.py
    extract_video_ids.py
  subtitles/
    downloader.py
    parsing.py
    windows.py
  retrieval/
    bm25.py
  eval/
    moment_retrieval.py
  legacy/
    eval_transcript_qa.py
  utils/
```

## Dataset Assumptions

The current dataset loader supports the Hugging Face dataset `Enxin/Video-MMLU`.

Raw rows are video-level and include fields such as:

- `video_id`
- `caption`
- `reasoning_qa`
- `captions_qa`

Flattened examples are question-level and include:

- `video_id`
- `question_id`
- `question`
- `answer`
- `qa_type`
- `transcript`
- optional `start_time`, `end_time`

Important current limitation:

- the released flattened data does **not** include gold timestamp spans for VMR
- the VMR evaluation code is structured for IoU / Recall once spans exist
- until then, the baseline still produces timestamp predictions, but gold-span metrics remain `null`

## Setup

On a fresh Ubuntu GCP VM:

```bash
sudo apt update
sudo apt install -y git
git clone <your-repo-url> videommlu_baselines_gcp
cd videommlu_baselines_gcp
bash scripts/setup_vm.sh
cp config/example.env .env
```

The setup script installs:

- Python + virtualenv
- `ffmpeg`
- Node.js 20 via NodeSource, which helps `yt-dlp` handle YouTube JS challenges more reliably

Activate the environment:

```bash
source .venv/bin/activate
```

Set the dataset source in `.env`:

```bash
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU
VIDEOMMLU_SPLIT=Video_MMLU
```

## End-to-End Pipeline

### 1. Extract unique video IDs

```bash
source .venv/bin/activate
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU \
python -m src.data.extract_video_ids \
  --split Video_MMLU \
  --output data/video_ids.txt
```

### 2. Download English subtitles

This downloader:

- prefers human subtitles
- falls back to auto-generated English subtitles
- skips already downloaded videos unless `--force`
- supports resumability via the manifest
- can retry failed/unavailable entries with `--retry-failed`

```bash
source .venv/bin/activate
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU \
python -m src.subtitles.downloader \
  --split Video_MMLU \
  --video-ids-path data/video_ids.txt \
  --raw-dir data/subtitles/raw \
  --manifest-path data/subtitles/subtitle_manifest.jsonl \
  --retries 3 \
  --sleep-seconds 1.0 \
  --request-sleep-seconds 0.5
```

Manifest rows look like:

```json
{
  "video_id": "Y8KMa8tJw-o",
  "status": "success_auto",
  "subtitle_path": "data/subtitles/raw/Y8KMa8tJw-o.auto.en.vtt",
  "subtitle_type": "auto",
  "error_message": null
}
```

Supported statuses:

- `success_human`
- `success_auto`
- `unavailable`
- `failed`

### 3. Parse VTT subtitles into timestamped segments

```bash
source .venv/bin/activate
python -m src.subtitles.parsing \
  --manifest-path data/subtitles/subtitle_manifest.jsonl \
  --output-dir data/subtitles/parsed
```

Parsed segment format:

```json
{
  "video_id": "Y8KMa8tJw-o",
  "segment_id": "Y8KMa8tJw-o:00000",
  "start_time": 12.34,
  "end_time": 16.78,
  "text": "normalized subtitle text"
}
```

Current parsing heuristics:

- strip VTT / HTML markup
- normalize whitespace
- drop empty cues
- merge consecutive cues whose normalized text is identical by extending the previous end time

### 4. Build temporal retrieval windows

```bash
source .venv/bin/activate
python -m src.subtitles.windows \
  --parsed-dir data/subtitles/parsed \
  --output-dir data/subtitles/windows \
  --window-size-sec 12 \
  --stride-sec 6
```

Window format:

```json
{
  "video_id": "Y8KMa8tJw-o",
  "window_id": "Y8KMa8tJw-o:000012000:000024000",
  "start_time": 12.0,
  "end_time": 24.0,
  "text": "concatenated subtitle text for this time window"
}
```

### 5. Run the VMR baseline

Using the shell helper:

```bash
source .venv/bin/activate
bash scripts/run_vmr_baseline.sh
```

Or directly:

```bash
source .venv/bin/activate
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU \
python -m src.eval.moment_retrieval \
  --split Video_MMLU \
  --windows-dir data/subtitles/windows \
  --top-k 3 \
  --output-dir outputs/vmr
```

The VMR baseline:

- uses the known `video_id` for each question
- retrieves top-k subtitle windows within that video
- predicts the top-1 window timestamps as the moment

If gold spans are present, the evaluator reports:

- IoU
- Recall@1 / Recall@3 at IoU `>= 0.3`
- Recall@1 / Recall@3 at IoU `>= 0.5`
- mean start error
- mean end error

If gold spans are absent, the retrieval pipeline still runs and writes timestamp predictions, but those gold-span metrics remain `null`.

## Legacy QA Path

The older transcript-answering baseline is preserved as a legacy experiment:

```bash
source .venv/bin/activate
bash scripts/run_transcript_baseline.sh
```

Implementation:

- [`src/legacy/eval_transcript_qa.py`](./src/legacy/eval_transcript_qa.py)

Compatibility wrapper:

- [`src/eval_transcript_bm25.py`](./src/eval_transcript_bm25.py)

## Useful Inspection Commands

Inspect flattened examples:

```bash
source .venv/bin/activate
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU \
python -m src.inspect_dataset --split Video_MMLU
```

Check raw vs flattened dataset views:

```bash
source .venv/bin/activate
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU \
python -m src.check_dataset_views --split Video_MMLU
```

Characterize the raw video-level dataset:

```bash
source .venv/bin/activate
VIDEOMMLU_HF_DATASET=Enxin/Video-MMLU \
python -m src.characterize_dataset --split Video_MMLU
```

## Outputs

- video IDs: [`data/video_ids.txt`](./data/video_ids.txt)
- subtitle manifest: [`data/subtitles/subtitle_manifest.jsonl`](./data/subtitles/subtitle_manifest.jsonl)
- raw subtitles: [`data/subtitles/raw`](./data/subtitles/raw)
- parsed subtitle segments: [`data/subtitles/parsed`](./data/subtitles/parsed)
- temporal windows: [`data/subtitles/windows`](./data/subtitles/windows)
- VMR predictions / summaries: [`outputs/vmr`](./outputs/vmr)
- legacy QA predictions / summaries: [`outputs/legacy_qa`](./outputs/legacy_qa)

## Notes and Limitations

- YouTube subtitle availability is incomplete; some videos are unavailable, blocked, or missing captions.
- `yt-dlp` may still warn about impersonation support on some systems. The downloader is designed to continue and log failures cleanly.
- The current VMR baseline is CPU-friendly and intentionally simple.
- The code structure is meant to leave clean room for:
  - OCR
  - keyframe extraction
  - multimodal reranking
  - confidence-based routing
