"""
inference パッケージ共通ユーティリティ。

フレームサンプリング、音声文字起こし、動画ダウンロード、API 呼び出しなど、
generate_answers.py / generate_answers_compare.py / generate_queries.py で
重複していた関数・定数をここに集約する。
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import torch
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from tqdm import tqdm
from yt_dlp import YoutubeDL

# ────────────────────────────────────────────────────────────
# パス設定
# ────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
INFERENCE_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")

# ────────────────────────────────────────────────────────────
# 定数
# ────────────────────────────────────────────────────────────
DEFAULT_FRAME_COUNT = 128
DEFAULT_TARGET_SIZE = (256, 256)

FORMAT_OPTIONS = [
    "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]",
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
    "bestvideo+bestaudio/best",
    "best[ext=mp4]/best",
    "best",
]

# 環境変数（fallback 付き）
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID", "")
VERTEX_REGION = os.getenv("VERTEX_REGION") or os.getenv("REGION", "global")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COOKIES_PATH = PROJECT_ROOT / "cookies.txt"

GEN_CONFIG = {
    "top_p": 1,
    "top_k": 1,
    "response_mime_type": "application/json",
}


# ────────────────────────────────────────────────────────────
# タイムスタンプ
# ────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    """秒数を HH:MM:SS.mmm 形式に変換する。"""
    total_seconds = max(seconds, 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# ────────────────────────────────────────────────────────────
# フレームサンプリング
# ────────────────────────────────────────────────────────────
def calculate_frame_indices(total_frames: int, target_count: int) -> List[int]:
    """等間隔でフレームインデックスを計算する。"""
    if target_count <= 0 or total_frames <= 0:
        return []
    if target_count >= total_frames:
        return list(range(total_frames))
    step = total_frames / target_count
    return sorted({min(int(round(i * step)), total_frames - 1) for i in range(target_count)})


def sample_frames(
    video_path: Path,
    target_count: int,
    include_timestamps: bool,
) -> Tuple[List[Dict], int, Optional[float], int, float]:
    """動画からフレームをサンプリングし、base64 エンコードして返す。

    Returns
    -------
    (frames_data, actual_count, duration, total_frames, fps)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = total_frames / fps if fps > 0 else None

    indices = calculate_frame_indices(total_frames, target_count)
    frames_data: List[Dict] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success or frame is None:
            continue
        resized = cv2.resize(frame, DEFAULT_TARGET_SIZE, interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", resized)
        if not success:
            continue
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        entry: Dict[str, Union[str, int, float]] = {
            "image_base64": encoded,
            "frame_index": int(frame_index),
        }
        if include_timestamps and fps > 0:
            timestamp = frame_index / fps
            entry["timestamp"] = timestamp
            entry["timestamp_str"] = format_timestamp(timestamp)
        frames_data.append(entry)

    cap.release()
    return frames_data, len(frames_data), duration, total_frames, fps


# ────────────────────────────────────────────────────────────
# 音声文字起こし
# ────────────────────────────────────────────────────────────
def transcribe_audio(
    video_path: Path,
    include_timestamps: bool,
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    tmp_audio: Optional[Path] = None,
) -> Dict:
    """faster-whisper で音声文字起こしを実行する。

    Parameters
    ----------
    whisper_model_size : Whisper モデルサイズ（例: "base", "large-v3"）。
    tmp_audio : 一時音声ファイルパス。省略時はカレントディレクトリに作成。
    """
    if tmp_audio is None:
        tmp_audio = Path("temp_audio.wav")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-err_detect", "ignore_err",
        "-y", str(tmp_audio),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)  # noqa: PLW1510
    if result.returncode != 0 and not tmp_audio.exists():
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

    compute_type = "float16" if DEVICE == "cuda" else "int8"
    try:
        tqdm.write(f"faster-whisper 読み込み中 ({whisper_model_size} on {DEVICE}, {compute_type})")
        model = WhisperModel(whisper_model_size, device=DEVICE, compute_type=compute_type)
    except Exception as err:  # pylint: disable=broad-except
        if DEVICE == "cuda":
            tqdm.write(f"GPU 読み込み失敗のため CPU(int8) で再試行: {err}")
            model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        else:
            raise

    segments_iter, _ = model.transcribe(str(tmp_audio), beam_size=5)
    segments_list = list(segments_iter)

    full_text = "".join(seg.text for seg in segments_list if seg.text).strip()
    converted_segments = [
        {
            "start": float(seg.start or 0.0),
            "end": float(seg.end or 0.0),
            "text": (seg.text or "").strip(),
        }
        for seg in segments_list
    ]

    if include_timestamps:
        lines = []
        for seg in converted_segments:
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            lines.append(f"[{start_str} - {end_str}] {seg['text']}")
        timestamped_text = "\n".join(lines)
    else:
        timestamped_text = ""

    return {
        "text": full_text,
        "timestamped_text": timestamped_text,
        "segments": converted_segments,
    }


# ────────────────────────────────────────────────────────────
# 動画ダウンロード
# ────────────────────────────────────────────────────────────
def yt_dlp_progress_hook(status: Dict) -> None:
    """yt-dlp の進捗を簡易表示する。"""
    if status.get("status") == "downloading":
        downloaded = status.get("downloaded_bytes") or 0
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        if total:
            percent = downloaded / total * 100
            tqdm.write(f"  ダウンロード中... {percent:.1f}%")
    elif status.get("status") == "finished":
        tqdm.write("  ダウンロード完了、後処理中 ...")


def download_video_from_youtube(
    video_url: str,
    output_path: Path,
    video_key: str = "video",
) -> Path:
    """YouTube から動画をダウンロードし、保存先パスを返す。

    FORMAT_OPTIONS を順に試行し、最初に成功したものを採用する。
    output_path がディレクトリの場合は ``{video_key}.%(ext)s`` テンプレートを使用し、
    ファイルの場合はそのまま outtmpl に設定する。

    Returns
    -------
    Path : ダウンロードされた動画ファイルのパス。
    """
    tqdm.write(f"動画ダウンロード開始: {video_url}")
    last_error: Optional[str] = None

    is_directory = output_path.is_dir()

    for attempt_index, format_opt in enumerate(FORMAT_OPTIONS, 1):
        tqdm.write(f"  フォーマット試行 {attempt_index}/{len(FORMAT_OPTIONS)}: {format_opt}")

        if is_directory:
            outtmpl = str(output_path / f"{video_key}.%(ext)s")
        else:
            outtmpl = str(output_path)

        ydl_opts: Dict = {
            "forceipv4": True,
            "rm_cachedir": True,
            "retries": 3,
            "fragment_retries": 10,
            "format": format_opt,
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "noprogress": False,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [yt_dlp_progress_hook],
        }
        if COOKIES_PATH.exists():
            ydl_opts["cookiefile"] = str(COOKIES_PATH)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as err:  # pylint: disable=broad-except
            last_error = str(err)
            tqdm.write(f"    ✗ 失敗: {last_error[:150]}")
            continue

        # ダウンロード結果を探す
        if is_directory:
            video_files = [
                path
                for path in output_path.glob(f"{video_key}.*")
                if path.is_file()
                and not path.name.endswith((".part", ".ytdl", ".info.json", ".json"))
            ]
            if video_files:
                video_files.sort(key=lambda p: p.stat().st_mtime)
                return video_files[-1]
        elif output_path.exists():
            return output_path

        last_error = "ダウンロード後に動画ファイルが見つかりません"

    raise FileNotFoundError(
        f"動画の取得に失敗しました: {last_error or '原因不明'}"
    )


# ────────────────────────────────────────────────────────────
# API レスポンス解析
# ────────────────────────────────────────────────────────────
def format_clips(clips_raw) -> str:
    """clips フィールド（リスト of [start, end]）を CSV 格納用文字列に変換する。

    入力例: [["00:04:12","00:07:23"],["00:12:12","00:12:56"]]
    出力例: "[[00:04:12, 00:07:23], [00:12:12, 00:12:56]]"
    """
    if clips_raw in (None, "", []):
        return ""
    if not clips_raw:
        return ""

    segments: List[str] = []
    if isinstance(clips_raw, (list, tuple)):
        for clip in clips_raw:
            if isinstance(clip, (list, tuple)) and len(clip) == 2:
                start = str(clip[0]).strip()
                end = str(clip[1]).strip()
                if start and end:
                    segments.append(f"[{start}, {end}]")
                else:
                    combined = ", ".join(part for part in (start, end) if part)
                    if combined:
                        segments.append(f"[{combined}]")
            else:
                clip_str = str(clip).strip()
                if clip_str:
                    if not (clip_str.startswith("[") and clip_str.endswith("]")):
                        clip_str = f"[{clip_str}]"
                    segments.append(clip_str)
    else:
        clip_str = str(clips_raw).strip()
        if clip_str and not (clip_str.startswith("[") and clip_str.endswith("]")):
            clip_str = f"[{clip_str}]"
        if clip_str:
            segments.append(clip_str)

    if not segments:
        return ""

    return "[" + ", ".join(segments) + "]"


def parse_api_response(response_text: str) -> Dict[str, str]:
    """API レスポンスから回答とクリップを抽出する。

    Returns
    -------
    {"answer": str, "clips": str}
    """
    empty: Dict[str, str] = {"answer": "", "clips": ""}
    if not response_text:
        return empty

    json_str = ""
    start = response_text.find("```json")
    if start != -1:
        end = response_text.find("```", start + 7)
        if end != -1:
            json_str = response_text[start + 7 : end].strip()
    if not json_str:
        json_str = response_text.strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return {"answer": response_text.strip(), "clips": ""}

    if isinstance(parsed, dict):
        results = parsed.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    answer = item.get("answer")
                    if isinstance(answer, str):
                        return {"answer": answer.strip(), "clips": format_clips(item.get("clips"))}
                elif isinstance(item, str):
                    return {"answer": item.strip(), "clips": ""}
        answer = parsed.get("answer")
        if isinstance(answer, str):
            return {"answer": answer.strip(), "clips": format_clips(parsed.get("clips"))}
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                answer = item.get("answer")
                if isinstance(answer, str):
                    return {"answer": answer.strip(), "clips": format_clips(item.get("clips"))}
            elif isinstance(item, str):
                return {"answer": item.strip(), "clips": ""}

    return {"answer": response_text.strip(), "clips": ""}


# ────────────────────────────────────────────────────────────
# API 呼び出し
# ────────────────────────────────────────────────────────────
def call_openai_api_raw(prompt: str, frames_data: List[Dict], model: str) -> str:
    """OpenAI API を呼び出し、生テキストを返す。"""
    import openai

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    for frame in frames_data:
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['image_base64']}",
                },
            }
        )

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content if response.choices else ""


def call_gemini_api_raw(prompt: str, frames_data: List[Dict], model: str) -> str:
    """Gemini API を呼び出し、生テキストを返す。"""
    client = genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT_ID or None,
        location=VERTEX_REGION,
    )

    content_parts: List = [prompt]
    for frame in frames_data:
        image_bytes = base64.b64decode(frame["image_base64"])
        content_parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=model,
        contents=content_parts,
        config=GEN_CONFIG,
    )
    return response.text if response and response.text else ""


def call_openai_api(prompt: str, frames_data: List[Dict], model: str) -> Dict[str, str]:
    """OpenAI API を呼び出し、parse_api_response を通した結果を返す。"""
    text = call_openai_api_raw(prompt, frames_data, model)
    return parse_api_response(text)


def call_gemini_api(prompt: str, frames_data: List[Dict], model: str) -> Dict[str, str]:
    """Gemini API を呼び出し、parse_api_response を通した結果を返す。"""
    text = call_gemini_api_raw(prompt, frames_data, model)
    return parse_api_response(text)
