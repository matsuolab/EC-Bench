"""
動画ファイルをGemini 2.5 Proへ渡してカウンティングクエリを生成するスクリプト。

手順:
1. videosディレクトリ内のすべてのmp4を順番に処理
2. 各動画をGCSにtemp.mp4としてアップロード
3. prompts.pyに定義されたプロンプトと動画をgemini-2.5-proへ入力して応答を取得
4. 応答JSONから12個のクエリを抽出し、動画IDと共にCSVへ保存
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Sequence, Set, Tuple

from google import genai
from google.cloud import storage
from google.genai import types

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from inference.query_prompts import PROMPT  # noqa: E402
from inference.utils import format_clips  # noqa: E402

# ────────────────────────────────────────────────────────────
# 設定値
# ────────────────────────────────────────────────────────────
DEFAULT_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "your-gcp-project-id")
DEFAULT_REGION = os.getenv("VERTEX_REGION", "global")
DEFAULT_BUCKET = os.getenv("VERTEX_BUCKET", "your-gcs-bucket")
DEFAULT_MODEL = "gemini-2.5-pro"
EXPECTED_QUERY_COUNT = 12

VIDEOS_DIR = SCRIPT_DIR / "videos"
OUTPUT_CSV = SCRIPT_DIR / "video_queries.csv"
TEMP_BLOB_NAME = "temp.mp4"
OUTPUT_COLUMNS = ["video_id", "query", "task_type", "answer", "category", "clips"]

GEN_CONFIG = {
    "response_mime_type": "application/json",
}


# ────────────────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────────────────
def iter_videos(video_dir: Path) -> Iterable[Path]:
    """videosディレクトリから拡張子mp4のファイルを取得"""
    if not video_dir.exists():
        raise FileNotFoundError(f"動画ディレクトリが存在しません: {video_dir}")

    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".mp4":
            yield path


def upload_video(bucket: storage.Bucket, video_path: Path) -> storage.Blob:
    """動画をGCSにtemp.mp4としてアップロード"""
    blob = bucket.blob(TEMP_BLOB_NAME)
    blob.upload_from_filename(str(video_path))
    return blob


def delete_blob(blob: storage.Blob) -> None:
    """アップロードした一時ファイルを削除"""
    if blob is None:
        return
    try:
        blob.delete()
    except Exception as error:  # noqa: BLE001
        print(f"[WARN] GCS一時ファイルの削除に失敗: {error}")


def call_gemini_with_video(
    client: genai.Client,
    model: str,
    gcs_uri: str,
    prompt: str,
    max_retries: int = 3,
) -> str:
    """動画URIとプロンプトを用いてGeminiへ問い合わせ"""
    video_part = types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4")

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, video_part],
                config=GEN_CONFIG,
            )
            return response.text
        except Exception as error:  # noqa: BLE001
            wait_seconds = 5 * attempt
            print(f"[WARN] Gemini呼び出し失敗({attempt}/{max_retries}): {error}")
            if attempt == max_retries:
                raise
            print(f"       {wait_seconds}秒後に再試行します")
            time.sleep(wait_seconds)
    raise RuntimeError("Gemini呼び出しに失敗しました")


def _extract_json_text(raw_text: str) -> str:
    """Gemini応答からJSON文字列部分を抽出"""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Gemini応答が空です")

    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                return part

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def parse_queries(response_text: str, expected_count: int = EXPECTED_QUERY_COUNT) -> List[dict]:
    """応答文字列からクエリ情報を抽出"""
    json_text = _extract_json_text(response_text)
    data = json.loads(json_text)

    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError("JSONにqueriesキーが存在しません")

    queries_raw = data["queries"]
    if not isinstance(queries_raw, Sequence):
        raise ValueError("queriesフィールドが配列ではありません")

    queries: List[dict] = []
    for entry in queries_raw:
        if isinstance(entry, dict):
            query_text = entry.get("query")
            if not isinstance(query_text, str):
                continue
            queries.append(
                {
                    "query": query_text.strip(),
                    "task_type": str(entry.get("tag", "")).strip(),
                    "answer": str(entry.get("answer", "")).strip(),
                    "category": str(entry.get("category", "")).strip(),
                    "clips": format_clips(entry.get("clips", "")),
                }
            )
        elif isinstance(entry, str):
            queries.append(
                {
                    "query": entry.strip(),
                    "task_type": "",
                    "answer": "",
                    "category": "",
                    "clips": "",
                }
            )

    queries = [q for q in queries if q["query"]]
    if len(queries) != expected_count:
        raise ValueError(f"クエリ数が{expected_count}ではありません: {len(queries)}件")

    return queries


def _normalize_rows(rows: List[dict]) -> List[dict]:
    """Ensure each row has all required columns."""
    normalized = []
    for row in rows:
        normalized.append({col: row.get(col, "") for col in OUTPUT_COLUMNS})
    return normalized


def load_existing_rows(output_path: Path) -> Tuple[List[dict], Counter]:
    """既存CSVの内容を読み込み、各video_idの件数を返す"""
    if not output_path.exists():
        return [], Counter()

    rows: List[dict] = []
    counts: Counter = Counter()

    with output_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return [], Counter()
        for row in reader:
            rows.append(row)
            video_id = (row.get("video_id") or "").strip()
            if video_id:
                counts[video_id] += 1

    return rows, counts


def write_rows_with_header(output_path: Path, rows: List[dict]) -> None:
    """ヘッダー付きでCSV全体を書き直す"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        if rows:
            writer.writerows(_normalize_rows(rows))


def append_rows_to_csv(rows: List[dict], output_path: Path) -> None:
    """CSVに行を追記。ヘッダーが無ければ作成"""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if file_exists else "w"

    with output_path.open(mode, encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(_normalize_rows(rows))


def process_single_video(
    video_path: Path,
    client: genai.Client,
    bucket: storage.Bucket,
    model: str,
    max_retries: int,
) -> List[dict]:
    """1本の動画からクエリを生成し文字列リストを返す"""
    print(f"\n[INFO] 動画処理開始: {video_path.name}")
    blob = None
    try:
        blob = upload_video(bucket, video_path)
        gcs_uri = f"gs://{bucket.name}/{blob.name}"
        print(f"[INFO] アップロード完了: {gcs_uri}")

        response_text = call_gemini_with_video(
            client=client,
            model=model,
            gcs_uri=gcs_uri,
            prompt=PROMPT,
            max_retries=max_retries,
        )
        queries = parse_queries(response_text)
        print(f"[INFO] クエリ生成成功: {len(queries)}件")
        return queries
    finally:
        delete_blob(blob)


def generate_queries_for_all_videos(
    videos_dir: Path,
    output_csv: Path,
    project_id: str,
    region: str,
    bucket_name: str,
    model: str,
    max_retries: int,
) -> None:
    """videosディレクトリ内の動画をすべて処理してCSVに出力"""
    videos = list(iter_videos(videos_dir))
    if not videos:
        raise RuntimeError(f"mp4動画が見つかりませんでした: {videos_dir}")

    print(f"[INFO] 対象動画: {len(videos)}本")

    client = genai.Client(vertexai=True, project=project_id, location=region)
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    existing_rows, counts = load_existing_rows(output_csv)
    partial_videos = {vid for vid, cnt in counts.items() if 0 < cnt < EXPECTED_QUERY_COUNT}
    if partial_videos:
        print(f"[WARN] 不完全な動画結果を検出したため再処理します: {sorted(partial_videos)}")
        filtered_rows = [
            row for row in existing_rows if (row.get("video_id") or "").strip() not in partial_videos
        ]
        write_rows_with_header(output_csv, filtered_rows)
        counts = Counter()
        for row in filtered_rows:
            vid = (row.get("video_id") or "").strip()
            if vid:
                counts[vid] += 1
        existing_rows = filtered_rows

    processed_video_ids: Set[str] = {
        vid for vid, cnt in counts.items() if cnt >= EXPECTED_QUERY_COUNT
    }
    if processed_video_ids:
        print(f"[INFO] 既存の結果をスキップ: {len(processed_video_ids)}件")

    newly_processed = 0
    for video_path in videos:
        video_id = video_path.stem
        if video_id in processed_video_ids:
            print(f"[INFO] 既存結果ありのためスキップ: {video_id}")
            continue

        try:
            query_items = process_single_video(
                video_path=video_path,
                client=client,
                bucket=bucket,
                model=model,
                max_retries=max_retries,
            )
        except Exception as error:  # noqa: BLE001
            print(f"[ERROR] {video_path.name}の処理で失敗: {error}")
            continue

        new_rows = [
            {
                "video_id": video_id,
                "query": item["query"],
                "task_type": item.get("task_type", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "clips": item.get("clips", ""),
            }
            for item in query_items
        ]

        append_rows_to_csv(new_rows, output_csv)
        processed_video_ids.add(video_id)
        newly_processed += 1
        print(f"[INFO] 出力へ追記完了: {video_id}")

    if newly_processed == 0:
        print(f"[INFO] 新規に処理する動画はありませんでした。出力: {output_csv}")
    else:
        print(f"[INFO] 新規{newly_processed}本の動画を処理しました。出力: {output_csv}")


# ────────────────────────────────────────────────────────────
# エントリポイント
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="videosディレクトリの動画をGemini 2.5 Proで解析しクエリを生成します。",
    )
    parser.add_argument(
        "--videos_dir",
        type=Path,
        default=VIDEOS_DIR,
        help="処理対象の動画ディレクトリ (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="生成結果を書き出すCSVファイルパス (default: %(default)s)",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        default=DEFAULT_PROJECT_ID,
        help="Vertex AIのプロジェクトID (default: %(default)s)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        help="Vertex AIのリージョン (default: %(default)s)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_BUCKET,
        help="動画をアップロードするGCSバケット名 (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="使用するGeminiモデル名 (default: %(default)s)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Gemini呼び出し失敗時のリトライ回数 (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_queries_for_all_videos(
        videos_dir=args.videos_dir.expanduser().resolve(),
        output_csv=args.output.expanduser().resolve(),
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket,
        model=args.model,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
