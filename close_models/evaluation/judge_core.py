"""
Shared evaluation core for LLM-as-a-Judge scripts.

This module provides the common evaluation logic used by all judge variants:
- Counting answer normalization (with MAE/RMSE support)
- Enumeration evaluation via LLM (GPT-5)
- Clip interval parsing and IoU computation
- DataFrame evaluation pipeline (unified superset)
- Ground-truth answer merging
- Answer column detection utilities
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ModuleNotFoundError:
    pass

from evaluation.prompts import PROMPTS

# ────────────────────────────────────────────────────────────
# OpenAI Settings
# ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gpt-5"

_client = None


def _get_client():
    """Lazy-init OpenAI client so importing this module doesn't crash without OPENAI_API_KEY."""
    global _client
    if _client is None:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY が設定されていません。環境変数をセットしてください。"
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ────────────────────────────────────────────────────────────
# Counting Utilities
# ────────────────────────────────────────────────────────────


def normalize_counting_answer(answer: Any) -> str:
    """Counting 用の回答を正規化して文字列にする。"""
    if pd.isna(answer):
        return ""

    text = str(answer).strip()
    if not text:
        return ""

    raw_numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not raw_numbers:
        return ""

    normalized: List[str] = []
    for item in raw_numbers:
        if "." in item:
            value = str(float(item)).rstrip("0").rstrip(".")
        else:
            value = str(int(item))
        normalized.append(value)

    unique_numbers = sorted(set(normalized), key=lambda x: float(x))
    return ", ".join(unique_numbers)


def parse_numbers_from_normalized(text: str) -> List[float]:
    """Parse floats from a comma-separated normalized string."""
    if not text:
        return []
    try:
        return [float(item) for item in text.split(", ")]
    except ValueError:
        return []


# ────────────────────────────────────────────────────────────
# Enumeration Utilities
# ────────────────────────────────────────────────────────────


def create_enumeration_prompt(true_answer: str, generated_answer: str) -> str:
    """Enumeration 判定用のプロンプトを作成する。"""
    template = PROMPTS["judge_enumeration"]["template"]
    return template.format(
        true_answer=true_answer,
        generated_answer=generated_answer,
    )


def parse_enumeration_response(response_text: str) -> Dict[str, Any]:
    """LLM 応答から JSON を抽出して dict に変換する。"""
    text = response_text.strip()

    if "```" in text:
        start = text.find("```json")
        if start == -1:
            start = text.find("```")
        end = text.find("```", start + 3)
        if start != -1 and end != -1:
            text = text[start + 3 : end].strip()

    data = json.loads(text)

    def ensure_list(key: str) -> List[str]:
        value = data.get(key, [])
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return []

    parsed = {
        "tp_items": ensure_list("tp_items"),
        "fp_items": ensure_list("fp_items"),
        "fn_items": ensure_list("fn_items"),
        "reasoning": str(data.get("reasoning", "")).strip(),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
    }
    parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
    return parsed


def judge_enumeration(
    true_answer: str,
    generated_answer: str,
    model: str,
    max_retries: int,
    retry_interval: float,
) -> Dict[str, Any]:
    """Enumeration タスクを LLM で判定し、統計値を返す。"""
    client = _get_client()
    prompt = create_enumeration_prompt(true_answer, generated_answer)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict evaluator that outputs JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = response.choices[0].message.content or ""
            parsed = parse_enumeration_response(response_text)

            tp = len(parsed["tp_items"])
            fp = len(parsed["fp_items"])
            fn = len(parsed["fn_items"])

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "precision_is_one": math.isclose(precision, 1.0),
                "reasoning": parsed["reasoning"],
                "confidence": parsed["confidence"],
                "tp_items": parsed["tp_items"],
                "fp_items": parsed["fp_items"],
                "fn_items": parsed["fn_items"],
            }

        except Exception as error:  # pylint: disable=broad-except
            last_error = error
            tqdm.write(f"Enumeration 判定失敗 (試行 {attempt}/{max_retries}): {error}")
            if attempt < max_retries:
                time.sleep(retry_interval)

    raise RuntimeError(f"Enumeration 判定に失敗しました: {last_error}")  # noqa: TRY003


# ────────────────────────────────────────────────────────────
# Clip Interval Utilities
# ────────────────────────────────────────────────────────────

TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")


def hms_to_seconds(match: Sequence[str] | tuple[str, ...]) -> int:
    hours, minutes, seconds = (int(part) for part in match)
    return hours * 3600 + minutes * 60 + seconds


def to_seconds(value: str) -> int | None:
    match = TIME_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    return hms_to_seconds(match.groups())


def parse_clip_intervals(raw: Any) -> List[Tuple[int, int]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []

    if isinstance(raw, list):
        intervals: List[Tuple[int, int]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                start = to_seconds(str(item[0]))
                end = to_seconds(str(item[1]))
                if start is not None and end is not None and end >= start:
                    intervals.append((start, end))
        return intervals

    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []

    try:
        parsed = literal_eval(text)
        if isinstance(parsed, list):
            return parse_clip_intervals(parsed)
    except Exception:
        pass

    tokens = TIME_PATTERN.findall(text)
    intervals = []
    for i in range(0, len(tokens), 2):
        if i + 1 >= len(tokens):
            break
        start = hms_to_seconds(tokens[i])
        end = hms_to_seconds(tokens[i + 1])
        if end >= start:
            intervals.append((start, end))
    return intervals


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_iou(
    intervals_a: List[Tuple[int, int]], intervals_b: List[Tuple[int, int]]
) -> float | None:
    if not intervals_a and not intervals_b:
        return None
    if not intervals_a or not intervals_b:
        return 0.0

    merged_a = merge_intervals(intervals_a)
    merged_b = merge_intervals(intervals_b)

    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        start_a, end_a = merged_a[i]
        start_b, end_b = merged_b[j]
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        if overlap_end > overlap_start:
            intersection += overlap_end - overlap_start
        if end_a < end_b:
            i += 1
        else:
            j += 1

    duration_a = sum(end - start for start, end in merged_a)
    duration_b = sum(end - start for start, end in merged_b)
    union = duration_a + duration_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ────────────────────────────────────────────────────────────
# DataFrame Evaluation (unified superset)
# ────────────────────────────────────────────────────────────


def _build_detail_entry(
    base_detail: Dict[str, Any],
    *,
    task_type: str = "",
    normalized_true: str = "",
    normalized_generated: str = "",
    is_match: bool | None = None,
    tp: int | None = None,
    fp: int | None = None,
    fn: int | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    precision_is_one: bool | None = None,
    reasoning: str = "",
    confidence: float | None = None,
    tp_items: str = "",
    fp_items: str = "",
    fn_items: str = "",
    counting_value_true: float | None = None,
    counting_value_generated: float | None = None,
    counting_abs_error: float | None = None,
    counting_sq_error: float | None = None,
) -> Dict[str, Any]:
    """detail entry の dict を統一的に構築する。"""
    return {
        **base_detail,
        "task_type": task_type,
        "normalized_true": normalized_true,
        "normalized_generated": normalized_generated,
        "is_match": is_match,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_is_one": precision_is_one,
        "reasoning": reasoning,
        "confidence": confidence,
        "tp_items": tp_items,
        "fp_items": fp_items,
        "fn_items": fn_items,
        "counting_value_true": counting_value_true,
        "counting_value_generated": counting_value_generated,
        "counting_abs_error": counting_abs_error,
        "counting_sq_error": counting_sq_error,
    }


def _compute_summary(
    *,
    counting_total: int,
    counting_match: int,
    counting_abs_error_sum: float,
    counting_sq_error_sum: float,
    counting_error_count: int,
    enumeration_total: int,
    enumeration_tp: int,
    enumeration_fp: int,
    enumeration_fn: int,
    enumeration_agreement: int,
    clip_iou_sum: float,
    clip_iou_count: int,
    junre_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """集計統計を計算して summary dict を返す。"""
    counting_rate = counting_match / counting_total * 100 if counting_total > 0 else 0.0
    counting_mae = (
        counting_abs_error_sum / counting_error_count if counting_error_count > 0 else 0.0
    )
    counting_rmse = (
        math.sqrt(counting_sq_error_sum / counting_error_count) if counting_error_count > 0 else 0.0
    )

    precision = (
        enumeration_tp / (enumeration_tp + enumeration_fp)
        if enumeration_tp + enumeration_fp > 0
        else 0.0
    )
    recall = (
        enumeration_tp / (enumeration_tp + enumeration_fn)
        if enumeration_tp + enumeration_fn > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    agreement_rate = (
        enumeration_agreement / enumeration_total * 100 if enumeration_total > 0 else 0.0
    )

    overall_total = counting_total + enumeration_total
    overall_correct = counting_match + enumeration_agreement
    overall_rate = overall_correct / overall_total * 100 if overall_total > 0 else 0.0

    clip_miou = clip_iou_sum / clip_iou_count if clip_iou_count > 0 else 0.0

    def calc_junre_rate(key: str) -> float:
        stats = junre_stats.get(key)
        if not stats or stats["total"] == 0:
            return 0.0
        return stats["correct"] / stats["total"] * 100

    return {
        "counting_total": counting_total,
        "counting_match": counting_match,
        "counting_match_rate_percent": counting_rate,
        "counting_mae": counting_mae,
        "counting_rmse": counting_rmse,
        "enumeration_total": enumeration_total,
        "enumeration_tp_total": enumeration_tp,
        "enumeration_fp_total": enumeration_fp,
        "enumeration_fn_total": enumeration_fn,
        "enumeration_precision": precision,
        "enumeration_recall": recall,
        "enumeration_f1": f1,
        "enumeration_agreement_count": enumeration_agreement,
        "enumeration_agreement_rate_percent": agreement_rate,
        "overall_accuracy_percent": overall_rate,
        "clip_miou": clip_miou,
        "A": calc_junre_rate("A"),
        "B": calc_junre_rate("B"),
        "C": calc_junre_rate("C"),
        "D": calc_junre_rate("D"),
        "E": calc_junre_rate("E"),
        "F": calc_junre_rate("F"),
    }


def evaluate_dataframe(
    df: pd.DataFrame,
    answer_column: str,
    model: str,
    max_retries: int,
    retry_interval: float,
    true_column: str = "answer",
    clips_column: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """DataFrame を評価し、詳細結果と集計値を返す。

    Parameters
    ----------
    df : DataFrame with task_type, true_column, answer_column columns.
    answer_column : Column name containing generated/human answers.
    model : OpenAI model name for enumeration judging.
    max_retries : Max retries for enumeration LLM calls.
    retry_interval : Sleep between retries (seconds).
    true_column : Column name for ground-truth answers (default "answer").
    clips_column : Optional column with predicted clip intervals for IoU.
    """
    counting_total = 0
    counting_match = 0
    counting_abs_error_sum = 0.0
    counting_sq_error_sum = 0.0
    counting_error_count = 0

    enumeration_total = 0
    enumeration_tp = 0
    enumeration_fp = 0
    enumeration_fn = 0
    enumeration_agreement = 0

    clip_iou_sum = 0.0
    clip_iou_count = 0

    junre_stats: Dict[str, Dict[str, float]] = {}

    def update_junre_stats(junre: str, is_correct: bool) -> None:
        if not junre:
            return
        stats = junre_stats.setdefault(junre, {"total": 0, "correct": 0})
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1

    detail_entries: List[Tuple[int, Dict[str, Any]]] = []
    enumeration_tasks: List[Dict[str, Any]] = []
    order_counter = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Counting 判定中"):
        generated_answer_raw = row.get(answer_column, "")
        if pd.isna(generated_answer_raw) or str(generated_answer_raw).strip() == "":
            continue

        true_answer_raw = row.get(true_column, "")
        if pd.isna(true_answer_raw) or str(true_answer_raw).strip() == "":
            continue

        true_answer = str(true_answer_raw)
        generated_answer = str(generated_answer_raw)
        q_tag = str(row.get("task_type", "") or "")
        q_junre = str(row.get("question_type", "") or "")

        # Clip IoU (when clips_column is provided)
        clip_iou = None
        if clips_column and clips_column in df.columns:
            true_intervals = parse_clip_intervals(row.get("clips"))
            pred_intervals = parse_clip_intervals(row.get(clips_column))
            if true_intervals or pred_intervals:
                clip_iou = compute_iou(true_intervals, pred_intervals)
                if clip_iou is not None:
                    clip_iou_sum += clip_iou
                    clip_iou_count += 1

        base_detail: Dict[str, Any] = {
            "index": index,
            "video_id": row.get("video_id", ""),
            "question_id": row.get("question_id", ""),
            "question_type": q_junre,
            "task_type": q_tag,
            "true_answer": true_answer,
            "generated_answer": generated_answer,
            "clip_iou": clip_iou,
        }

        if q_tag == "Counting":
            counting_total += 1
            normalized_true = normalize_counting_answer(true_answer)
            normalized_generated = normalize_counting_answer(generated_answer)
            is_match = (
                bool(normalized_true)
                and bool(normalized_generated)
                and normalized_true == normalized_generated
            )
            if is_match:
                counting_match += 1

            update_junre_stats(q_junre, is_match)

            true_numbers = parse_numbers_from_normalized(normalized_true)
            generated_numbers = parse_numbers_from_normalized(normalized_generated)
            counting_abs_error = None
            counting_sq_error = None
            if true_numbers and generated_numbers:
                diff = generated_numbers[0] - true_numbers[0]
                counting_abs_error = abs(diff)
                counting_sq_error = diff ** 2
                counting_abs_error_sum += counting_abs_error
                counting_sq_error_sum += counting_sq_error
                counting_error_count += 1

            detail_entries.append((
                order_counter,
                _build_detail_entry(
                    base_detail,
                    task_type="Counting",
                    normalized_true=normalized_true,
                    normalized_generated=normalized_generated,
                    is_match=is_match,
                    counting_value_true=true_numbers[0] if true_numbers else None,
                    counting_value_generated=generated_numbers[0] if generated_numbers else None,
                    counting_abs_error=counting_abs_error,
                    counting_sq_error=counting_sq_error,
                ),
            ))
        elif q_tag == "Enumeration":
            enumeration_total += 1
            enumeration_tasks.append(
                {
                    "order": order_counter,
                    "base_detail": base_detail,
                    "true_answer": true_answer,
                    "generated_answer": generated_answer,
                    "q_junre": q_junre,
                }
            )
        else:
            detail_entries.append((
                order_counter,
                _build_detail_entry(
                    base_detail,
                    task_type="Unsupported",
                    reasoning="Unsupported task_type",
                ),
            ))
        order_counter += 1

    # ── Enumeration (parallel LLM judging) ──
    enumeration_detail_entries: List[Tuple[int, Dict[str, Any]]] = []
    if enumeration_tasks:
        max_workers = min(len(enumeration_tasks), os.cpu_count() or 1)

        def process_enum_task(task: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = judge_enumeration(
                    task["true_answer"],
                    task["generated_answer"],
                    model=model,
                    max_retries=max_retries,
                    retry_interval=retry_interval,
                )
            except Exception as error:  # pylint: disable=broad-except
                tqdm.write(f"Enumeration 判定エラー: {error}")
                result = {
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "precision_is_one": False,
                    "reasoning": f"ERROR: {error}",
                    "confidence": 0.0,
                    "tp_items": [], "fp_items": [], "fn_items": [],
                }
            return {**task, "result": result}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_enum_task, task): task for task in enumeration_tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Enumeration 判定中",
            ):
                outcome = future.result()
                result = outcome["result"]
                q_junre = outcome["q_junre"]

                enumeration_tp += result["tp"]
                enumeration_fp += result["fp"]
                enumeration_fn += result["fn"]
                if result["precision_is_one"]:
                    enumeration_agreement += 1

                update_junre_stats(q_junre, result["precision_is_one"])

                enumeration_detail_entries.append((
                    outcome["order"],
                    _build_detail_entry(
                        outcome["base_detail"],
                        task_type="Enumeration",
                        tp=result["tp"],
                        fp=result["fp"],
                        fn=result["fn"],
                        precision=result["precision"],
                        recall=result["recall"],
                        f1=result["f1"],
                        precision_is_one=result["precision_is_one"],
                        reasoning=result["reasoning"],
                        confidence=result["confidence"],
                        tp_items="; ".join(result["tp_items"]),
                        fp_items="; ".join(result["fp_items"]),
                        fn_items="; ".join(result["fn_items"]),
                    ),
                ))

    detail_entries.extend(enumeration_detail_entries)
    details = [detail for _, detail in sorted(detail_entries, key=lambda x: x[0])]

    summary = _compute_summary(
        counting_total=counting_total,
        counting_match=counting_match,
        counting_abs_error_sum=counting_abs_error_sum,
        counting_sq_error_sum=counting_sq_error_sum,
        counting_error_count=counting_error_count,
        enumeration_total=enumeration_total,
        enumeration_tp=enumeration_tp,
        enumeration_fp=enumeration_fp,
        enumeration_fn=enumeration_fn,
        enumeration_agreement=enumeration_agreement,
        clip_iou_sum=clip_iou_sum,
        clip_iou_count=clip_iou_count,
        junre_stats=junre_stats,
    )

    return details, summary


# ────────────────────────────────────────────────────────────
# Ground-truth merging
# ────────────────────────────────────────────────────────────


def ensure_true_answers(df: pd.DataFrame, true_answer_csv: Any) -> pd.DataFrame:
    """Ensure DataFrame has ground-truth answers in column 'answer'."""
    from pathlib import Path

    true_answer_csv = Path(true_answer_csv)

    def _is_blank(series: pd.Series) -> pd.Series:
        str_series = series.astype(str).str.strip()
        lower_series = str_series.str.lower()
        return (
            series.isna()
            | str_series.eq("")
            | lower_series.eq("nan")
            | lower_series.eq("none")
        )

    needs_merge = "answer" not in df.columns
    partial_missing = False
    if not needs_merge:
        blank_mask = _is_blank(df["answer"])
        needs_merge = blank_mask.all()
        partial_missing = blank_mask.any() and not needs_merge

    if not needs_merge and not partial_missing:
        return df

    if not true_answer_csv.exists():
        raise FileNotFoundError(
            f"正解データセットが見つかりません: {true_answer_csv}"
        )

    # Determine which columns to load from ground truth
    gt_all = pd.read_csv(true_answer_csv)
    gt_use_cols = ["question_id", "answer"]
    for col in ["video_id", "task_type", "question_type"]:
        if col in gt_all.columns:
            gt_use_cols.append(col)
    ground_truth = gt_all[gt_use_cols].drop_duplicates(subset=["question_id"])

    # Choose merge key: use question_id only when video_id is missing from input
    has_video_id = "video_id" in df.columns
    merge_keys = ["video_id", "question_id"] if has_video_id else ["question_id"]

    if "answer" in df.columns:
        merged = df.merge(
            ground_truth, on=merge_keys, how="left", suffixes=("", "_truth")
        )
        if "answer_truth" in merged.columns:
            blank_mask = _is_blank(merged["answer"])
            merged["answer"] = merged["answer"].where(~blank_mask, merged["answer_truth"])
            merged = merged.drop(columns=["answer_truth"])
        # Backfill missing metadata columns from ground truth
        for col in ["video_id", "task_type", "question_type"]:
            if f"{col}_truth" in merged.columns:
                if col not in df.columns:
                    merged[col] = merged[f"{col}_truth"]
                else:
                    blank = _is_blank(merged[col])
                    merged[col] = merged[col].where(~blank, merged[f"{col}_truth"])
                merged = merged.drop(columns=[f"{col}_truth"])
            elif col not in merged.columns and col in ground_truth.columns:
                merged = merged.merge(
                    ground_truth[["question_id", col]].drop_duplicates(),
                    on="question_id", how="left", suffixes=("", "_gt"),
                )
                if f"{col}_gt" in merged.columns:
                    merged[col] = merged[f"{col}_gt"]
                    merged = merged.drop(columns=[f"{col}_gt"])
    else:
        merged = df.merge(ground_truth, on=merge_keys, how="left")

    missing_after = merged["answer"].isna().sum()
    if missing_after > 0:
        tqdm.write(f"警告: 正解が見つからなかった行が {missing_after} 件あります。")

    return merged


# ────────────────────────────────────────────────────────────
# Column detection utilities
# ────────────────────────────────────────────────────────────


def detect_answer_columns(df: pd.DataFrame, explicit: List[str] | None) -> List[str]:
    """Return answer column(s) to evaluate, auto-detecting ``*_A`` if not specified."""
    if explicit:
        unique = list(dict.fromkeys(explicit))
        missing = [column for column in unique if column not in df.columns]
        if missing:
            available = ", ".join(df.columns)
            raise ValueError(
                f"指定された回答カラムが見つかりません: {', '.join(missing)}。利用可能: {available}"
            )
        return unique

    detected = [col for col in df.columns if col.endswith("_A") and col not in ("A", "answer")]
    if not detected:
        raise ValueError("`*_A` 形式の回答カラムが見つかりませんでした。")
    return detected


def column_to_model_name(column: str) -> str:
    """Strip trailing ``_A`` to derive a model name from a column name."""
    if column.endswith("_A"):
        return column[:-2]
    return column
