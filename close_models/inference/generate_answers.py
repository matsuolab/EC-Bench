# このスクリプトはサンプリングフレームと音声文字起こしを用いてGPT/Gemini回答を生成するツールです
"""
dataset.csvのデータをもとに、サンプリングフレームと音声文字起こしを使用して
GPT-5またはGeminiに生成的解答を作成させるプログラム。

機能:
1. dataset.csvを読み込み
2. 各video_idについて動画をダウンロード
3. 指定フレーム数で等間隔サンプリング（256x256にリサイズ）
4. Whisperで音声文字起こし
5. そのvideo_idに関連するすべての質問に対してGPT-5/Geminiで解答を生成
6. 結果をCSVに保存

パラメータ:
- フレーム数: 指定数で等間隔サンプリング（64, 128など、または1fps, 2fpsなど）
- タイムスタンプの有無: フレームのタイムスタンプを含めるかどうか
- 文字起こしの有無: 音声文字起こし（タイムスタンプ付き）を含めるかどうか
- モデル: GPT-5またはGemini
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from inference.utils import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_TARGET_SIZE,
    DEVICE,
    FORMAT_OPTIONS,
    GEN_CONFIG,
    PROJECT_ROOT,
    VERTEX_PROJECT_ID,
    VERTEX_REGION,
    calculate_frame_indices,
    download_video_from_youtube,
    format_clips,
    format_timestamp,
    parse_api_response,
    sample_frames,
    transcribe_audio,
    yt_dlp_progress_hook,
)
from inference.prompts import PROMPTS

# ────────────────────────────────────────────────────────────
# 設定定数
# ────────────────────────────────────────────────────────────
INPUT_CSV = PROJECT_ROOT / "data" / "dataset.csv"
TRANSCRIPT_CSV = PROJECT_ROOT / "data" / "transcript.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TMP_VIDEO = Path("temp_video.mp4")
TMP_AUDIO = Path("temp_audio.wav")
FRAMES_DIR = Path("temp_frames")

TRANSCRIPT_COLUMNS = [
    "video_id",
    "whisper_model",
    "include_timestamps",
    "text",
    "timestamped_text",
    "segments_json",
    "updated_at",
]

# API設定
OPENAI_MODEL = "gpt-5"
GEMINI_MODEL = "gemini-2.5-flash"

# デフォルト Whisper モデルサイズ（CLI 引数で上書き可能）
_DEFAULT_WHISPER_MODEL_SIZE = "large-v3"


# ────────────────────────────────────────────────────────────
# FPS モード用ユーティリティ（generate_answers.py 固有）
# ────────────────────────────────────────────────────────────
def get_video_info(video_path: Path) -> Dict:
    """動画の基本情報を取得"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


def calculate_frame_indices_by_fps(duration: float, target_fps: float) -> Tuple[List[float], int]:
    """指定FPSでフレームタイムスタンプを計算"""
    if duration <= 0:
        return [], 0
    timestamps = []
    current_time = 0.0
    step = 1.0 / target_fps
    while current_time < duration:
        timestamps.append(current_time)
        current_time += step
    return timestamps, len(timestamps)


def extract_frames_by_fps(
    video_path: Path,
    target_fps: float,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    include_timestamps: bool = True,
) -> List[Dict]:
    """指定FPSで動画からフレームを抽出"""
    video_info = get_video_info(video_path)
    timestamps, _ = calculate_frame_indices_by_fps(video_info["duration"], target_fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        tqdm.write(f"警告: 動画ファイルを開けません: {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frames_data = []

    for timestamp in timestamps:
        frame_idx = int(timestamp * original_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            tqdm.write(f"警告: タイムスタンプ {timestamp:.3f}s (フレーム {frame_idx}) の読み込みに失敗しました")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        frame_data: Dict = {
            "frame_index": frame_idx,
            "image_base64": img_base64,
        }
        if include_timestamps:
            frame_data["timestamp"] = timestamp
            frame_data["timestamp_str"] = format_timestamp(timestamp)
        frames_data.append(frame_data)

    cap.release()
    return frames_data


# ────────────────────────────────────────────────────────────
# 正規化ヘルパー
# ────────────────────────────────────────────────────────────
def _normalize_bool(value) -> bool:
    """CSVから読み込んだ値を真偽値に正規化する"""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    value_str = str(value).strip().lower()
    return value_str in {"1", "true", "t", "yes"}


# ────────────────────────────────────────────────────────────
# 文字起こしキャッシュ
# ────────────────────────────────────────────────────────────
def _ensure_transcript_cache_exists() -> None:
    """transcript.csv が存在し、必要なカラムを持つことを保証する"""
    if not TRANSCRIPT_CSV.exists():
        df = pd.DataFrame(columns=TRANSCRIPT_COLUMNS)
        df.to_csv(TRANSCRIPT_CSV, index=False, encoding="utf-8")
        return
    df = pd.read_csv(TRANSCRIPT_CSV)
    missing_columns = [col for col in TRANSCRIPT_COLUMNS if col not in df.columns]
    if missing_columns:
        for col in missing_columns:
            df[col] = pd.NA
        df = df[TRANSCRIPT_COLUMNS]
        df.to_csv(TRANSCRIPT_CSV, index=False, encoding="utf-8")


def load_transcript_from_cache(
    video_id: str, whisper_model: str, include_timestamps: bool
) -> Optional[Dict]:
    """transcript.csv から該当動画の文字起こしを取得"""
    _ensure_transcript_cache_exists()
    df = pd.read_csv(TRANSCRIPT_CSV)
    if df.empty:
        return None
    for col in TRANSCRIPT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    video_mask = df["video_id"].astype(str) == str(video_id)
    model_mask = df["whisper_model"].astype(str) == str(whisper_model)
    timestamps_mask = df["include_timestamps"].apply(_normalize_bool) == bool(include_timestamps)
    filtered = df[video_mask & model_mask & timestamps_mask]
    if filtered.empty:
        return None

    row = filtered.iloc[-1]
    text_value = row["text"] if pd.notna(row["text"]) else ""
    timestamped_value = row["timestamped_text"] if pd.notna(row["timestamped_text"]) else ""
    segments_value: List[Dict] = []
    if pd.notna(row["segments_json"]) and str(row["segments_json"]).strip():
        try:
            segments_value = json.loads(row["segments_json"])
        except json.JSONDecodeError:
            segments_value = []
    return {
        "text": text_value,
        "timestamped_text": timestamped_value,
        "segments": segments_value,
    }


def save_transcript_to_cache(
    video_id: str, transcription: Dict, whisper_model: str, include_timestamps: bool
) -> None:
    """新しい文字起こし結果を transcript.csv に保存"""
    _ensure_transcript_cache_exists()
    df = pd.read_csv(TRANSCRIPT_CSV)
    for col in TRANSCRIPT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    include_mask = df["include_timestamps"].apply(_normalize_bool)
    mask = (
        (df["video_id"].astype(str) == str(video_id))
        & (df["whisper_model"].astype(str) == str(whisper_model))
        & (include_mask == bool(include_timestamps))
    )
    df = df[~mask]

    new_row = {
        "video_id": video_id,
        "whisper_model": whisper_model,
        "include_timestamps": bool(include_timestamps),
        "text": transcription.get("text", ""),
        "timestamped_text": transcription.get("timestamped_text", ""),
        "segments_json": json.dumps(transcription.get("segments", []), ensure_ascii=False),
        "updated_at": datetime.utcnow().isoformat(),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df[TRANSCRIPT_COLUMNS]
    df.to_csv(TRANSCRIPT_CSV, index=False, encoding="utf-8")


# ────────────────────────────────────────────────────────────
# プロンプト作成関数
# ────────────────────────────────────────────────────────────
def create_sampling_prompt(
    queries: List[Dict],
    frames_data: List[Dict],
    transcription: Dict,
    include_timestamps: bool = True,
    include_transcription: bool = True,
) -> str:
    """サンプリングフレーム用プロンプトを生成"""
    queries_text = ""
    for i, query_info in enumerate(queries, 1):
        query_text = query_info["query"]
        tag = query_info["tag"]
        queries_text += f"**Query{i}**: {query_text}\n**Type{i}**: {tag} (Counting: numerical answer / Enumeration: list answer)\n\n"

    frames_info = f"\n## Video Frames Information\nTotal sampled frames: {len(frames_data)}\n"
    if include_timestamps and frames_data:
        frames_info += "Frame timestamps:\n"
        for i, frame in enumerate(frames_data):
            if "timestamp_str" in frame:
                frames_info += f"Frame {i+1}: {frame['timestamp_str']}\n"

    transcription_info = ""
    if include_transcription and transcription:
        transcription_info = "\n## Audio Transcription\n"
        if include_timestamps and transcription.get("timestamped_text"):
            transcription_info += transcription["timestamped_text"]
        else:
            transcription_info += transcription.get("text", "")

    template = PROMPTS["generate_sampling_openQA_with_clips"]["template"]
    return template.format(
        queries_section=queries_text.strip(),
        frames_section=frames_info.strip(),
        transcription_section=transcription_info.strip(),
    )


# ────────────────────────────────────────────────────────────
# API呼び出し関数（raw テキストを返す版 — バッチモードで必要）
# ────────────────────────────────────────────────────────────
def _call_api_raw(prompt: str, frames_data: List[Dict], model_type: str) -> str:
    """model_type に応じて OpenAI / Gemini を呼び出し、生テキストを返す。"""
    if model_type.lower() == "openai":
        import openai

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        for frame in frames_data:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame['image_base64']}"},
                }
            )
        response = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        return response.choices[0].message.content if response.choices else ""
    else:
        from google import genai
        from google.genai import types

        client = genai.Client(vertexai=True, project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
        content_parts: list = [prompt]
        for frame in frames_data:
            img_data = base64.b64decode(frame["image_base64"])
            content_parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=content_parts, config=GEN_CONFIG
        )
        return response.text if response and response.text else ""


# ────────────────────────────────────────────────────────────
# 動画準備ヘルパー（ダウンロード→情報取得→フレーム抽出→文字起こし）
# ────────────────────────────────────────────────────────────
def _prepare_video_for_queries(
    video_id: Union[str, int],
    video_url: str,
    frame_count: Union[int, float],
    is_fps_mode: bool,
    include_timestamps: bool,
    include_transcription: bool,
    whisper_model_size: str,
) -> Tuple[List[Dict], Dict, Optional[float]]:
    """動画のダウンロード、フレーム抽出、文字起こしを一括で行う。

    Returns
    -------
    (frames_data, transcription, duration_seconds)
    """
    # 1) ダウンロード
    local_path = download_video_from_youtube(video_url, TMP_VIDEO)

    # 2) 動画情報取得
    video_info = get_video_info(local_path)
    duration_seconds = float(video_info["duration"]) if "duration" in video_info else None
    tqdm.write(f"動画情報: {video_info['duration']:.2f}秒, {video_info['fps']:.2f}fps")

    # 3) フレームサンプリング
    tqdm.write("フレームサンプリング開始...")
    if is_fps_mode:
        frames_data = extract_frames_by_fps(local_path, frame_count, DEFAULT_TARGET_SIZE, include_timestamps)
        tqdm.write(f"FPSモード: {frame_count}fps で {len(frames_data)}フレーム抽出")
    else:
        frames_data, actual_count, _, _, _ = sample_frames(local_path, int(frame_count), include_timestamps)
        tqdm.write(f"フレーム数モード: {len(frames_data)}フレーム抽出")

    # 4) 文字起こし
    transcription: Dict = {}
    if include_transcription:
        cached = load_transcript_from_cache(str(video_id), whisper_model_size, include_timestamps)
        if cached:
            transcription = cached
            tqdm.write("キャッシュ済み文字起こしを使用 (transcript.csv)")
        else:
            tqdm.write("音声文字起こし開始...")
            transcription = transcribe_audio(
                local_path, include_timestamps,
                whisper_model_size=whisper_model_size, tmp_audio=TMP_AUDIO,
            )
            tqdm.write(f"文字起こし完了: {len(transcription.get('text', ''))}文字")
            save_transcript_to_cache(str(video_id), transcription, whisper_model_size, include_timestamps)
            tqdm.write("文字起こし結果を transcript.csv に保存")

    return frames_data, transcription, duration_seconds


# ────────────────────────────────────────────────────────────
# メイン処理関数
# ────────────────────────────────────────────────────────────
def process_video_queries_sampling(
    video_id: str,
    video_url: str,
    queries: List[Dict],
    frame_count: Union[int, float] = DEFAULT_FRAME_COUNT,
    is_fps_mode: bool = False,
    include_timestamps: bool = True,
    include_transcription: bool = True,
    model_type: str = "gemini",
    whisper_model_size: str = _DEFAULT_WHISPER_MODEL_SIZE,
) -> Dict:
    """全質問を1リクエストで送信するバッチモード。"""
    results: Dict = {}
    duration_seconds: Optional[float] = None

    try:
        frames_data, transcription, duration_seconds = _prepare_video_for_queries(
            video_id, video_url, frame_count, is_fps_mode,
            include_timestamps, include_transcription, whisper_model_size,
        )

        # プロンプト生成
        prompt = create_sampling_prompt(
            queries, frames_data, transcription, include_timestamps, include_transcription
        )

        # API 呼び出し（リトライ付き）
        tqdm.write(f"{model_type.upper()} API 呼び出し開始: {len(queries)}個のクエリを処理中...")
        response_text: Optional[str] = None
        max_retry = 3
        for attempt in range(max_retry):
            try:
                response_text = _call_api_raw(prompt, frames_data, model_type)
                break
            except Exception as e:
                tqdm.write(f"API エラー (try {attempt+1}/{max_retry}): {e}")
                if attempt < max_retry - 1:
                    wait = 10 * 2**attempt
                    tqdm.write(f"{wait} 秒後にリトライ ...")
                    time.sleep(wait)
                else:
                    raise

        if response_text is None:
            raise RuntimeError("API 応答無し")

        # レスポンス解析
        try:
            tqdm.write(f"API レスポンス受信: {len(response_text)} 文字")
            json_start = response_text.find("```json")
            json_end = response_text.find("```", json_start + 7)
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start + 7 : json_end].strip()
            else:
                json_str = response_text.strip()

            parsed_response = json.loads(json_str)
            if "results" in parsed_response:
                for result in parsed_response["results"]:
                    query_id = result.get("query_id", 0)
                    answer = result.get("answer", "")
                    clips = format_clips(result.get("clips"))
                    results[query_id - 1] = {"answer": answer, "clips": clips}

            tqdm.write(f"解析完了: {len(results)}個の回答を取得")

        except json.JSONDecodeError as e:
            tqdm.write(f"JSON パースエラー: {e}")
            tqdm.write(f"レスポンス内容: {response_text[:500]}...")
            for i in range(len(queries)):
                results[i] = {"answer": "", "clips": ""}

    except Exception as e:
        error_msg = str(e)
        tqdm.write(f"動画処理エラー (ID: {video_id}): {error_msg}")
        if "only images are available" in error_msg.lower():
            tqdm.write(f"⚠ Video ID {video_id}: この動画は画像のみで動画フォーマットが利用できません")
        elif any(msg in error_msg.lower() for msg in ["private video", "video unavailable", "has been removed"]):
            tqdm.write(f"⚠ Video ID {video_id}: この動画は視聴できません")
        for i in range(len(queries)):
            results[i] = {"answer": ""}

    finally:
        if TMP_VIDEO.exists():
            TMP_VIDEO.unlink()
        if TMP_AUDIO.exists():
            TMP_AUDIO.unlink()
        if FRAMES_DIR.exists():
            shutil.rmtree(FRAMES_DIR)

    return {"answers": results, "duration": duration_seconds}


def process_video_queries_by1(
    video_id: Union[str, int],
    video_url: str,
    queries: List[Dict],
    frame_count: Union[int, float] = DEFAULT_FRAME_COUNT,
    is_fps_mode: bool = False,
    include_timestamps: bool = True,
    include_transcription: bool = True,
    model_type: str = "gemini",
    whisper_model_size: str = _DEFAULT_WHISPER_MODEL_SIZE,
) -> Dict:
    """各質問ごとに1リクエスト送信する逐次モード。"""
    results: Dict[int, Dict[str, str]] = {}
    duration_seconds: Optional[float] = None

    try:
        frames_data, transcription, duration_seconds = _prepare_video_for_queries(
            video_id, video_url, frame_count, is_fps_mode,
            include_timestamps, include_transcription, whisper_model_size,
        )

        for query_index, query_info in enumerate(queries):
            prompt = create_sampling_prompt(
                [query_info], frames_data, transcription, include_timestamps, include_transcription
            )
            tqdm.write(
                f"{model_type.upper()} API 呼び出し開始: 動画 {video_id} の質問 {query_index + 1}/{len(queries)} を処理中..."
            )

            response_text: Optional[str] = None
            max_retry = 3
            for attempt in range(max_retry):
                try:
                    response_text = _call_api_raw(prompt, frames_data, model_type)
                    break
                except Exception as err:
                    tqdm.write(f"API エラー (try {attempt + 1}/{max_retry}): {err}")
                    if attempt < max_retry - 1:
                        wait_seconds = 10 * 2**attempt
                        tqdm.write(f"{wait_seconds} 秒後にリトライ ...")
                        time.sleep(wait_seconds)
                    else:
                        response_text = None

            if response_text is None:
                tqdm.write("API 応答無し: 空回答を記録")
                results[query_index] = {"answer": "", "clips": ""}
                continue

            tqdm.write(f"API レスポンス受信（{len(response_text)} 文字）")
            parsed = parse_api_response(response_text)
            results[query_index] = {"answer": parsed["answer"], "clips": parsed["clips"]}
            tqdm.write(f"解析完了: 質問 {query_index + 1} の回答 -> {parsed['answer'][:80]}")

    except Exception as err:
        error_msg = str(err)
        tqdm.write(f"動画処理エラー (ID: {video_id}): {error_msg}")
        if "only images are available" in error_msg.lower():
            tqdm.write(f"⚠ Video ID {video_id}: この動画は画像のみで動画フォーマットが利用できません")
        elif any(msg in error_msg.lower() for msg in ["private video", "video unavailable", "has been removed"]):
            tqdm.write(f"⚠ Video ID {video_id}: この動画は視聴できません")
        for query_index in range(len(queries)):
            results[query_index] = {"answer": ""}

    finally:
        if TMP_VIDEO.exists():
            TMP_VIDEO.unlink()
        if TMP_AUDIO.exists():
            TMP_AUDIO.unlink()
        if FRAMES_DIR.exists():
            shutil.rmtree(FRAMES_DIR)

    return {"answers": results, "duration": duration_seconds}


# ────────────────────────────────────────────────────────────
# メイン実行関数
# ────────────────────────────────────────────────────────────
def main(
    frame_count: Union[int, float] = DEFAULT_FRAME_COUNT,
    is_fps_mode: bool = False,
    include_timestamps: bool = True,
    include_transcription: bool = True,
    model_type: str = "gemini",
    batch: bool = False,
    whisper_model_size: str = _DEFAULT_WHISPER_MODEL_SIZE,
):
    """メイン処理"""
    # 出力ファイル名を設定に応じて変更
    mode_str = f"{frame_count}fps" if is_fps_mode else f"{int(frame_count)}frames"
    timestamp_str = "with_ts" if include_timestamps else "no_ts"
    transcription_str = "with_audio" if include_transcription else "no_audio"
    batch_str = "batch_" if batch else ""
    output_filename = f"dataset_{model_type}_{batch_str}{mode_str}_{timestamp_str}_{transcription_str}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename

    # ─── 1) CSV読み込み ─────────────────
    answer_col = f"{model_type}_A"
    clips_col = f"{model_type}_clips"

    if output_path.exists():
        tqdm.write(f"既存の処理結果を読み込み: {output_path}")
        df = pd.read_csv(output_path)
        if answer_col not in df.columns:
            df[answer_col] = ""
        if clips_col not in df.columns:
            df[clips_col] = ""
        if "duration" not in df.columns:
            df["duration"] = pd.NA
    else:
        tqdm.write(f"初回処理 - 元データを読み込み: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        df[answer_col] = ""
        df[clips_col] = ""
        df["duration"] = pd.NA

    df[answer_col] = df[answer_col].astype("string").fillna("")
    df[clips_col] = df[clips_col].astype("string").fillna("")

    tqdm.write(f"データ数: {len(df)}行")
    tqdm.write(f"設定: {mode_str}, タイムスタンプ: {include_timestamps}, 音声: {include_transcription}, モデル: {model_type}")

    # GPU情報を表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        tqdm.write(f"GPU使用: {gpu_name} ({gpu_memory:.1f}GB)")
    tqdm.write(f"Whisperモデル: {whisper_model_size} on {DEVICE}")

    # ─── 2) video_id毎にグループ化して処理 ─────────────────
    grouped = df.groupby("video_id")

    for video_id, group in tqdm(grouped, desc="動画処理中"):
        tqdm.write(f"\n--- Video ID: {video_id} ---")

        if all(pd.notna(group[answer_col]) & (group[answer_col] != "")):
            tqdm.write("既に処理済みのためスキップ")
            continue

        video_url = None
        for url in group["video_url"]:
            if pd.notna(url) and url.strip():
                video_url = url.strip()
                break
        if not video_url:
            tqdm.write("動画URLが見つからないためスキップ")
            continue

        queries = []
        query_indices = []
        for idx, row in group.iterrows():
            if pd.notna(row["question"]) and row["question"].strip():
                if pd.notna(row[answer_col]) and str(row[answer_col]).strip() != "":
                    tqdm.write(f"  質問「{row['question'][:50]}...」は既に処理済みのためスキップ")
                    continue
                queries.append({
                    "query": row["question"].strip(),
                    "tag": row.get("task_type", "Counting"),
                })
                query_indices.append(idx)

        if not queries:
            tqdm.write("全て処理済みのためスキップ")
            continue

        tqdm.write(f"質問数: {len(queries)}個{'（バッチ処理）' if batch else '（逐次処理）'}")

        # ─── 3) 動画とクエリを処理 ─────────────────
        try:
            process_fn = process_video_queries_sampling if batch else process_video_queries_by1
            process_result = process_fn(
                video_id, video_url, queries,
                frame_count, is_fps_mode, include_timestamps, include_transcription,
                model_type, whisper_model_size,
            )
            results = process_result.get("answers", {})
            duration_seconds = process_result.get("duration", None)

            # ─── 4) 結果をDataFrameに反映 ─────────────────
            for query_index, df_index in enumerate(query_indices):
                if query_index in results:
                    df.loc[df_index, answer_col] = results[query_index]["answer"]
                    df.loc[df_index, clips_col] = results[query_index].get("clips", "")
                    tqdm.write(f"  結果反映: インデックス{df_index} -> {results[query_index]['answer'][:50]}...")

            if duration_seconds is not None:
                df.loc[group.index, "duration"] = float(duration_seconds)

            tqdm.write(f"Video ID {video_id} の処理完了")

            # 進捗保存
            df.to_csv(output_path, index=False, encoding="utf-8")
            tqdm.write(f"進捗保存完了: {output_path}")

        except Exception as e:
            tqdm.write(f"Video ID {video_id} の処理でエラー: {e}")
            continue

    # ─── 5) 最終保存 ─────────────────
    df.to_csv(output_path, index=False, encoding="utf-8")
    tqdm.write(f"\n処理完了！結果を保存: {output_path}")

    total_rows = len(df)
    completed_rows = len(df[df[answer_col] != ""])
    tqdm.write(f"処理済み行数: {completed_rows}/{total_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="サンプリングフレームを使用した動画質問応答システム")
    parser.add_argument("--frame_count", type=float, default=DEFAULT_FRAME_COUNT,
                        help="フレーム数またはFPS (デフォルト: 128)")
    parser.add_argument("--is_fps_mode", type=lambda x: x.lower() == "true", default=False,
                        help="FPSモードを使用するかどうか (デフォルト: False)")
    parser.add_argument("--include_timestamps", type=lambda x: x.lower() == "true", default=True,
                        help="タイムスタンプを含めるかどうか (デフォルト: True)")
    parser.add_argument("--include_transcription", type=lambda x: x.lower() == "true", default=True,
                        help="音声文字起こしを含めるかどうか (デフォルト: True)")
    parser.add_argument("--model_type", type=str, default="gemini", choices=["gemini", "openai"],
                        help="使用するモデル (デフォルト: gemini)")
    parser.add_argument("--whisper_model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisperモデルサイズ (デフォルト: base)")
    parser.add_argument("--batch", action="store_true", default=False,
                        help="バッチモード: 全質問を1リクエストで送信 (デフォルト: False)")

    args = parser.parse_args()

    main(
        frame_count=args.frame_count,
        is_fps_mode=args.is_fps_mode,
        include_timestamps=args.include_timestamps,
        include_transcription=args.include_transcription,
        model_type=args.model_type,
        batch=args.batch,
        whisper_model_size=args.whisper_model,
    )
