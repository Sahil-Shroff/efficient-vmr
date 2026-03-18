from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation import AnnotationWorkspace, AnnotationWorkspaceConfig, parse_gold_timestamps


def _normalize_spans_for_state(spans: list[dict[str, Any]] | None) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for span in spans or []:
        try:
            normalized.append(
                {
                    "start": float(span.get("start", 0.0)),
                    "end": float(span.get("end", 0.0)),
                }
            )
        except (TypeError, ValueError, AttributeError):
            continue
    return normalized


def _set_spans_state(spans: list[dict[str, Any]] | None) -> None:
    normalized = _normalize_spans_for_state(spans)
    if not normalized:
        normalized = [{"start": 0.0, "end": 0.0}]
    st.session_state["gold_spans"] = normalized
    st.session_state["gold_timestamps_text"] = json.dumps(normalized, indent=2)


def _sync_json_from_spans() -> None:
    spans = _normalize_spans_for_state(st.session_state.get("gold_spans", []))
    st.session_state["gold_timestamps_text"] = json.dumps(spans, indent=2)


def _sync_spans_from_json() -> tuple[bool, str | None]:
    try:
        spans = parse_gold_timestamps(st.session_state.get("gold_timestamps_text", ""))
    except (ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
    _set_spans_state(spans)
    return True, None


def _max_video_time(example: dict[str, Any], workspace: AnnotationWorkspace) -> float:
    video_id = str(example.get("video_id", ""))
    values: list[float] = []
    for segment in workspace.get_segments(video_id):
        try:
            values.append(float(segment["end_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    for window in workspace.get_windows(video_id):
        try:
            values.append(float(window["end_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _load_workspace(
    split: str,
    dataset_path: str,
    windows_dir: str,
    parsed_dir: str,
    annotations_path: str,
    video_ids_path: str,
    top_k: int,
) -> AnnotationWorkspace:
    config = AnnotationWorkspaceConfig(
        split=split,
        dataset_path=dataset_path,
        windows_dir=windows_dir,
        parsed_dir=parsed_dir,
        annotations_path=annotations_path,
        video_ids_path=video_ids_path,
        top_k=top_k,
    )
    return AnnotationWorkspace(config)


@st.cache_resource(show_spinner=False)
def get_workspace(
    split: str,
    dataset_path: str,
    windows_dir: str,
    parsed_dir: str,
    annotations_path: str,
    video_ids_path: str,
    top_k: int,
) -> AnnotationWorkspace:
    return _load_workspace(
        split=split,
        dataset_path=dataset_path,
        windows_dir=windows_dir,
        parsed_dir=parsed_dir,
        annotations_path=annotations_path,
        video_ids_path=video_ids_path,
        top_k=top_k,
    )


def _initialize_form_state(question_id: str, annotation: dict[str, Any] | None) -> None:
    if st.session_state.get("form_question_id") == question_id:
        return

    st.session_state["form_question_id"] = question_id
    st.session_state["is_relevant"] = str((annotation or {}).get("isRelevant", 1))
    st.session_state["confidence"] = str((annotation or {}).get("confidence", "medium"))
    st.session_state["notes"] = str((annotation or {}).get("notes", ""))
    _set_spans_state((annotation or {}).get("gold_timestamps") or [])


def _filtered_examples(
    workspace: AnnotationWorkspace,
    selected_videos: list[str],
    selected_qa_types: list[str],
    annotated_filter: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    selected_video_set = set(selected_videos)
    selected_qa_type_set = set(selected_qa_types)

    for example in workspace.examples:
        if selected_video_set and str(example.get("video_id", "")) not in selected_video_set:
            continue
        if selected_qa_type_set and str(example.get("qa_type", "")) not in selected_qa_type_set:
            continue

        has_annotation = workspace.get_annotation(str(example.get("question_id", ""))) is not None
        if annotated_filter == "Annotated only" and not has_annotation:
            continue
        if annotated_filter == "Unannotated only" and has_annotation:
            continue
        results.append(example)

    return results


def _save_current_annotation(workspace: AnnotationWorkspace, example: dict[str, Any]) -> tuple[bool, str]:
    gold_timestamps = _normalize_spans_for_state(st.session_state.get("gold_spans", []))
    try:
        gold_timestamps = parse_gold_timestamps(json.dumps(gold_timestamps))
    except (ValueError, json.JSONDecodeError) as exc:
        return False, f"Could not save annotation: {exc}"

    row = {
        "question_id": str(example.get("question_id")),
        "video_id": str(example.get("video_id")),
        "qa_type": str(example.get("qa_type")),
        "isRelevant": int(st.session_state.get("is_relevant", "1")),
        "gold_timestamps": gold_timestamps,
        "confidence": str(st.session_state.get("confidence", "medium")),
        "notes": str(st.session_state.get("notes", "")),
    }
    workspace.save_annotation(row)
    return True, f"Saved annotation for {row['question_id']}"


def _render_span_editor(example: dict[str, Any], workspace: AnnotationWorkspace, candidates: list[dict[str, Any]]) -> None:
    max_time = _max_video_time(example, workspace)
    step = 0.5

    st.markdown("**gold_timestamps**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        if st.button("Add span", use_container_width=True):
            spans = _normalize_spans_for_state(st.session_state.get("gold_spans", []))
            spans.append({"start": 0.0, "end": max_time})
            _set_spans_state(spans)
            st.rerun()
    with quick_col2:
        if st.button("Reset span", use_container_width=True):
            _set_spans_state([{"start": 0.0, "end": 0.0}])
            st.rerun()
    with quick_col3:
        if st.button("Entire video", use_container_width=True):
            _set_spans_state([{"start": 0.0, "end": max_time}])
            st.rerun()
    with quick_col4:
        if st.button("Top-1 span", use_container_width=True, disabled=not candidates):
            top1 = candidates[0]
            _set_spans_state([{"start": top1["start_time"], "end": top1["end_time"]}])
            st.rerun()

    spans = _normalize_spans_for_state(st.session_state.get("gold_spans", []))
    if not spans:
        spans = [{"start": 0.0, "end": max_time}]
        _set_spans_state(spans)

    updated_spans: list[dict[str, float]] = []
    for idx, span in enumerate(spans):
        row_col1, row_col2, row_col3, row_col4 = st.columns([0.7, 1.3, 1.3, 0.8])
        with row_col1:
            st.caption(f"Span {idx + 1}")
        with row_col2:
            start_kwargs: dict[str, Any] = {
                "label": f"start_{idx}",
                "min_value": 0.0,
                "value": min(float(span["start"]), max_time) if max_time > 0 else float(span["start"]),
                "step": step,
                "key": f"span_start_{idx}",
                "label_visibility": "collapsed",
            }
            if max_time > 0:
                start_kwargs["max_value"] = max_time
            start_value = st.number_input(**start_kwargs)
            st.caption("start")
        with row_col3:
            end_kwargs: dict[str, Any] = {
                "label": f"end_{idx}",
                "min_value": 0.0,
                "value": min(float(span["end"]), max_time) if max_time > 0 else float(span["end"]),
                "step": step,
                "key": f"span_end_{idx}",
                "label_visibility": "collapsed",
            }
            if max_time > 0:
                end_kwargs["max_value"] = max_time
            end_value = st.number_input(**end_kwargs)
            st.caption("end")
        with row_col4:
            if st.button("Remove", key=f"remove_span_{idx}", use_container_width=True, disabled=len(spans) == 1):
                remaining = [item for span_idx, item in enumerate(spans) if span_idx != idx]
                _set_spans_state(remaining)
                st.rerun()

        if end_value < start_value:
            end_value = start_value
        updated_spans.append({"start": float(start_value), "end": float(end_value)})

    st.session_state["gold_spans"] = updated_spans
    _sync_json_from_spans()

    with st.expander("Advanced JSON editor", expanded=False):
        st.text_area(
            "gold_timestamps_json",
            key="gold_timestamps_text",
            height=140,
            label_visibility="collapsed",
        )
        if st.button("Apply JSON to spans", use_container_width=True):
            success, error = _sync_spans_from_json()
            if success:
                st.success("Updated span controls from JSON.")
                st.rerun()
            else:
                st.error(f"Could not parse JSON: {error}")


def _jump_to_question(filtered_examples: list[dict[str, Any]], question_id: str) -> int | None:
    for idx, example in enumerate(filtered_examples):
        if str(example.get("question_id", "")) == question_id:
            return idx
    return None


def _current_index_label(example: dict[str, Any], index: int) -> str:
    return f"{index + 1:03d} | {example.get('video_id', '')} | {example.get('qa_type', '')} | {example.get('question_id', '')}"


def main() -> None:
    st.set_page_config(page_title="VMR Annotation", layout="wide")
    st.title("Video-MMLU VMR Annotation")
    st.caption("Annotate question-level temporal relevance and gold timestamp spans with subtitle and BM25 context.")

    defaults = AnnotationWorkspaceConfig()

    with st.sidebar:
        st.header("Data")
        split = st.text_input("Split", value=defaults.split)
        dataset_path = st.text_input("Dataset JSON/JSONL", value=defaults.dataset_path)
        windows_dir = st.text_input("Windows dir", value=defaults.windows_dir)
        parsed_dir = st.text_input("Parsed subtitles dir", value=defaults.parsed_dir)
        annotations_path = st.text_input("Annotations JSONL", value=defaults.annotations_path)
        video_ids_path = st.text_input("Optional video_id subset file", value="")
        top_k = st.number_input("BM25 suggestions", min_value=1, max_value=10, value=3, step=1)

    try:
        workspace = get_workspace(
            split=split,
            dataset_path=dataset_path,
            windows_dir=windows_dir,
            parsed_dir=parsed_dir,
            annotations_path=annotations_path,
            video_ids_path=video_ids_path,
            top_k=int(top_k),
        )
    except Exception as exc:
        st.error(f"Could not load workspace: {exc}")
        return

    if not workspace.examples:
        st.error("No examples loaded. Check the dataset path and optional filters.")
        return

    with st.sidebar:
        st.header("Filters")
        available_video_ids = workspace.list_video_ids()
        available_qa_types = sorted({str(example.get("qa_type", "")) for example in workspace.examples})
        selected_videos = st.multiselect("video_id", options=available_video_ids)
        selected_qa_types = st.multiselect("qa_type", options=available_qa_types, default=available_qa_types)
        annotated_filter = st.selectbox("Annotation status", options=["All", "Unannotated only", "Annotated only"])
        jump_question_id = st.text_input("Jump to question_id")

    filtered_examples = _filtered_examples(workspace, selected_videos, selected_qa_types, annotated_filter)
    if not filtered_examples:
        st.warning("No examples match the current filters.")
        return

    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0

    if jump_question_id.strip():
        jump_index = _jump_to_question(filtered_examples, jump_question_id.strip())
        if jump_index is not None:
            st.session_state["current_index"] = jump_index
        else:
            st.sidebar.warning("question_id not found in current filtered view.")

    st.session_state["current_index"] = max(0, min(st.session_state["current_index"], len(filtered_examples) - 1))

    jump_col1, jump_col2 = st.columns([1, 2])
    with jump_col1:
        go_to_number = st.number_input(
            "Go to example #",
            min_value=1,
            max_value=len(filtered_examples),
            value=st.session_state["current_index"] + 1,
            step=1,
        )
        if int(go_to_number) - 1 != st.session_state["current_index"]:
            st.session_state["current_index"] = int(go_to_number) - 1
            st.rerun()
    with jump_col2:
        current_label = _current_index_label(
            filtered_examples[st.session_state["current_index"]],
            st.session_state["current_index"],
        )
        selected_label = st.selectbox(
            "Jump to question",
            options=[_current_index_label(example, idx) for idx, example in enumerate(filtered_examples)],
            index=st.session_state["current_index"],
        )
        if selected_label != current_label:
            st.session_state["current_index"] = int(selected_label.split(" | ", 1)[0]) - 1
            st.rerun()

    slider_index = st.slider(
        "Browse filtered examples",
        min_value=1,
        max_value=len(filtered_examples),
        value=st.session_state["current_index"] + 1,
    )
    if slider_index - 1 != st.session_state["current_index"]:
        st.session_state["current_index"] = slider_index - 1
        st.rerun()

    current_example = filtered_examples[st.session_state["current_index"]]
    current_question_id = str(current_example.get("question_id", ""))
    current_annotation = workspace.get_annotation(current_question_id)
    _initialize_form_state(current_question_id, current_annotation)

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 2])
    with nav_col1:
        if st.button("Prev", use_container_width=True, disabled=st.session_state["current_index"] == 0):
            st.session_state["current_index"] -= 1
            st.rerun()
    with nav_col2:
        if st.button(
            "Next",
            use_container_width=True,
            disabled=st.session_state["current_index"] >= len(filtered_examples) - 1,
        ):
            st.session_state["current_index"] += 1
            st.rerun()
    with nav_col3:
        st.metric("Progress", f"{st.session_state['current_index'] + 1} / {len(filtered_examples)}")
    with nav_col4:
        annotated_count = sum(1 for example in filtered_examples if workspace.get_annotation(str(example["question_id"])))
        st.metric("Annotated", f"{annotated_count} / {len(filtered_examples)}")

    left_col, right_col = st.columns([1.05, 0.95])

    with left_col:
        st.subheader("Example")
        youtube_url = f"https://www.youtube.com/watch?v={current_example.get('video_id', '')}"
        st.markdown(f"**video_id**: `{current_example.get('video_id', '')}`")
        st.markdown(f"**YouTube**: {youtube_url}")
        st.markdown(f"**question_id**: `{current_question_id}`")
        st.markdown(f"**qa_type**: `{current_example.get('qa_type', '')}`")
        st.markdown("**Question**")
        st.write(current_example.get("question", ""))
        st.markdown("**Answer**")
        st.write(current_example.get("answer", ""))

        transcript = str(current_example.get("transcript", "")).strip()
        with st.expander("Video caption / transcript", expanded=False):
            st.write(transcript or "No transcript text found in the flattened example.")

        candidates = workspace.get_retrieval_candidates(current_question_id, top_k=int(top_k))
        st.subheader("BM25 Suggestions")
        if not candidates:
            st.info("No subtitle windows available for this video.")
        else:
            top1 = candidates[0]
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("Mark relevant", use_container_width=True):
                    st.session_state["is_relevant"] = "1"
            with action_col2:
                if st.button("Mark not relevant", use_container_width=True):
                    st.session_state["is_relevant"] = "-1"
            with action_col3:
                if st.button("Use top-1 as gold seed", use_container_width=True):
                    _set_spans_state([{"start": top1["start_time"], "end": top1["end_time"]}])
                    st.rerun()

            for candidate in candidates:
                header = (
                    f"Rank {candidate['rank']} | "
                    f"{candidate['start_time']:.3f}s - {candidate['end_time']:.3f}s | "
                    f"score={candidate['score']:.3f}"
                )
                with st.expander(header, expanded=candidate["rank"] == 1):
                    if st.button(
                        f"Seed with rank {candidate['rank']}",
                        key=f"seed_{current_question_id}_{candidate['rank']}",
                    ):
                        _set_spans_state([{"start": candidate["start_time"], "end": candidate["end_time"]}])
                        st.rerun()
                    st.write(candidate.get("text", ""))

    with right_col:
        st.subheader("Annotation")
        st.radio("isRelevant", options=["1", "-1"], horizontal=True, key="is_relevant", format_func=lambda v: v)
        st.selectbox("confidence", options=["high", "medium", "low"], key="confidence")
        _render_span_editor(current_example, workspace, candidates)
        st.text_area("notes", key="notes", height=120)

        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("Save", type="primary", use_container_width=True):
                success, message = _save_current_annotation(workspace, current_example)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        with save_col2:
            if st.button("Save and next", use_container_width=True):
                success, message = _save_current_annotation(workspace, current_example)
                if success:
                    st.success(message)
                    if st.session_state["current_index"] < len(filtered_examples) - 1:
                        st.session_state["current_index"] += 1
                    st.rerun()
                else:
                    st.error(message)

        st.subheader("Subtitle Context")
        segments = workspace.get_segments(str(current_example.get("video_id", "")))
        if not segments:
            st.info("No parsed subtitle segments found for this video.")
        else:
            subtitle_lines = [
                f"[{segment['start_time']:.3f} - {segment['end_time']:.3f}] {segment['text']}"
                for segment in segments
            ]
            st.text_area(
                "Segments",
                value="\n".join(subtitle_lines),
                height=420,
                disabled=True,
                label_visibility="collapsed",
            )

    st.caption(
        "Annotations are written incrementally to "
        f"`{Path(workspace.annotations_path).as_posix()}` and existing question_id rows are loaded for editing."
    )


if __name__ == "__main__":
    main()
