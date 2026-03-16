# Video-MMLU Baselines on GCP

A lightweight starter repo for running **baseline experiments** on a GCP VM for your Independent Study.

This scaffold is intentionally simple and focused on the first experiments your advisor asked for:

1. **Inspect the dataset**
2. **Run a transcript-only retrieval baseline**
3. **Measure accuracy and basic latency**
4. **Prepare a clean path for adding keyframes/OCR later**

## What is included

- `scripts/setup_vm.sh` — apt + Python environment setup for an Ubuntu GCP VM
- `scripts/run_transcript_baseline.sh` — one-command baseline runner
- `src/inspect_dataset.py` — dataset characterization
- `src/eval_transcript_bm25.py` — transcript-only BM25 baseline for MCQ QA
- `src/load_videommlu.py` — flexible local/HuggingFace dataset loading helpers
- `src/utils.py` — shared helpers
- `requirements.txt` — minimal Python dependencies
- `config/example.env` — environment variables template

## Expected dataset format

The code supports either:

### Option A: local JSON / JSONL
A file where each row/object looks roughly like:

```json
{
  "video_id": "Y8KMa8tJw-o",
  "question_id": "...",
  "question": "...",
  "choices": ["A", "B", "C", "D"],
  "answer": 1,
  "subject": "physics",
  "transcript": "...",
  "start_time": 12.5,
  "end_time": 31.0
}
```

### Option B: Hugging Face dataset
If you already know the dataset repo name, set it in `.env`.

## Quick start on GCP

```bash
sudo apt update
sudo apt install -y git
git clone <your-repo-url> videommlu_baselines_gcp
cd videommlu_baselines_gcp
bash scripts/setup_vm.sh
cp config/example.env .env
```

Edit `.env` and set either:

- `VIDEOMMLU_LOCAL_PATH=/path/to/train.jsonl`
- or `VIDEOMMLU_HF_DATASET=your_dataset_name`

### Inspect the dataset

```bash
source .venv/bin/activate
python -m src.inspect_dataset --split train
```

### Check raw vs flattened dataset views

For Hugging Face Video-MMLU, use the dataset split name directly:

```bash
source .venv/bin/activate
python -m src.check_dataset_views --split Video_MMLU
```

This prints:

- the number of raw video rows
- the number of flattened QA rows
- unique `video_id` counts in both views
- the average number of QA rows per video
- a few sample raw and flattened examples

For `Enxin/Video-MMLU`, a healthy result is that the flattened QA row count is much larger than the raw video row count, and the average QA rows per video is roughly in the 20-30 range.

### Characterize the dataset

To compute basic statistics from the original video-level rows:

```bash
source .venv/bin/activate
python -m src.characterize_dataset --split Video_MMLU
```

This prints:

- average transcript length in tokens
- average questions per video
- the `qa_type` distribution across `reasoning_qa` and `captions_qa`

### Run transcript-only baseline

```bash
source .venv/bin/activate
bash scripts/run_transcript_baseline.sh
```

For `Enxin/Video-MMLU`, this runs a transcript-only BM25 baseline over flattened QA rows and reports:

- `retrieval_recall_at_k`: whether the gold answer string appears in the top-`k` retrieved transcript windows
- `answer_contains_accuracy`: whether the predicted answer sentence from the top retrieved window contains the gold answer string
- per-query timing fields in the predictions file:
  `retrieval_time_ms`, `llm_time_ms`, `total_latency_ms`

This dataset is free-form QA rather than multiple choice, so the baseline currently reports retrieval and extractive answer metrics instead of MCQ accuracy.

## Baseline logic

The baseline is deliberately cheap:

- Split transcript into fixed windows
- Build BM25 index over windows per example/video
- Use the question + answer choices as the retrieval query
- Retrieve top-k windows
- Score each answer choice by lexical overlap with retrieved evidence
- Predict the best answer

This is not fancy, but it is exactly the kind of **first baseline** you want before adding OCR, CLIP, reranking, or routing.

## Output

Results go to `outputs/`:

- summary JSON
- per-example predictions JSONL

## Suggested next experiments

After this baseline works:

1. Add transcript segmentation ablations: 10s / 20s / 30s windows
2. Compare BM25 vs dense retrieval
3. Add keyframe candidates from slide-change detection
4. Add OCR only for low-confidence cases
5. Measure latency / cost deltas

## Notes

- This scaffold is CPU-friendly for early experiments.
- It avoids hard-coding a specific Video-MMLU schema beyond common fields.
- If your dataset schema differs, edit `src/load_videommlu.py` in one place instead of patching everything like a gremlin.
