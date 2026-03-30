"""Unified LLM-as-a-Judge evaluation CLI.

Replaces the former ``llm_as_judge.py``, ``whole_judge.py``, and
``evaluate_human.py`` scripts with a single entry point.

Usage examples::

    # Single answer column (was llm_as_judge.py)
    python -m evaluation.evaluate --input results.csv --answer-column gemini_A

    # Auto-detect *_A columns, skip GT merge (was whole_judge.py)
    python -m evaluation.evaluate --input whole_exp_data.csv --no-gt-merge

    # Human evaluation with custom columns (was evaluate_human.py)
    python -m evaluation.evaluate --input human.csv \
        --answer-column "Human Answer" --true-col true_answer --no-gt-merge
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from evaluation.judge_core import (  # noqa: E402
    DEFAULT_MODEL,
    column_to_model_name,
    detect_answer_columns,
    ensure_true_answers,
    evaluate_dataframe,
)

DEFAULT_GT_CSV = Path(__file__).resolve().parent / "dataset.csv"


def run(
    input_csv: Path,
    output_dir: Path | None,
    answer_columns: List[str] | None,
    true_col: str,
    gt_csv: Path,
    no_gt_merge: bool,
    model: str,
    max_retries: int,
    retry_interval: float,
    q_num: int | None,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"入力 CSV が見つかりません: {input_csv}")

    output_dir = output_dir if output_dir is not None else input_csv.parent
    stem = input_csv.stem

    tqdm.write(f"CSV 読み込み: {input_csv}")
    df = pd.read_csv(input_csv)

    # GT merge (only when true_col is default "answer" and merge is requested)
    if not no_gt_merge and true_col == "answer":
        df = ensure_true_answers(df, gt_csv)

    columns = detect_answer_columns(df, answer_columns)
    tqdm.write(f"評価対象カラム: {', '.join(columns)}")

    df_subset = df
    if q_num is not None:
        if q_num <= 0:
            raise ValueError("--q-num は 1 以上を指定してください。")
        df_subset = df_subset.head(q_num)
        tqdm.write(f"テスト実行: 先頭 {len(df_subset)} 行を処理します (--q-num={q_num})")

    output_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"出力先: {output_dir}")

    multi = len(columns) > 1

    def evaluate_column(column: str) -> None:
        model_name = column_to_model_name(column)
        tqdm.write(f"[{model_name}] 判定開始")

        valid_mask = df_subset[column].notna() & (df_subset[column].astype(str).str.strip() != "")
        truth_mask = df_subset[true_col].notna() & (df_subset[true_col].astype(str).str.strip() != "")
        valid_df = df_subset[valid_mask & truth_mask]
        tqdm.write(f"[{model_name}] 評価対象行数: {len(valid_df)} / {len(df_subset)}")

        # Auto-detect clips column
        clips_column_name = f"{model_name}_clips"
        clips_column = clips_column_name if clips_column_name in df_subset.columns else None

        details, summary = evaluate_dataframe(
            valid_df,
            answer_column=column,
            model=model,
            max_retries=max_retries,
            retry_interval=retry_interval,
            true_column=true_col,
            clips_column=clips_column,
        )

        if multi:
            s_name = f"{model_name}_{stem}_judge_summary.csv"
            d_name = f"{model_name}_{stem}_judge_detail.csv"
        else:
            s_name = f"{stem}_judge_summary.csv"
            d_name = f"{stem}_judge_detail.csv"

        summary_path = output_dir / s_name
        details_path = output_dir / d_name

        if details:
            pd.DataFrame(details).to_csv(details_path, index=False, encoding="utf-8")
            tqdm.write(f"[{model_name}] 詳細結果を保存: {details_path}")
        else:
            tqdm.write(f"[{model_name}] 詳細結果は空です。CSV 出力をスキップします。")

        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8")
        tqdm.write(f"[{model_name}] 集計結果を保存: {summary_path}")

    if len(columns) == 1:
        evaluate_column(columns[0])
    else:
        max_workers = min(len(columns), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_column, col): col for col in columns}
            for future in tqdm(as_completed(futures), total=len(futures), desc="モデル評価中"):
                col = futures[future]
                try:
                    future.result()
                except Exception as error:  # pylint: disable=broad-except
                    tqdm.write(f"[{column_to_model_name(col)}] 評価中にエラー: {error}")
                    raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified LLM-as-a-Judge evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: input's parent)")
    parser.add_argument(
        "--answer-column",
        dest="answer_columns",
        nargs="+",
        default=None,
        metavar="COL",
        help="Answer column(s) to evaluate (default: auto-detect *_A)",
    )
    parser.add_argument(
        "--true-col",
        default="answer",
        help="Ground truth column name (default: answer)",
    )
    parser.add_argument(
        "--gt-csv",
        default=str(DEFAULT_GT_CSV),
        help="GT CSV for merging when true col is missing (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--no-gt-merge",
        action="store_true",
        default=False,
        help="Skip ground-truth merge",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge LLM model (default: gpt-5)")
    parser.add_argument("--max-retries", type=int, default=3, help="Enumeration judge max retries")
    parser.add_argument("--retry-interval", type=float, default=2.0, help="Retry interval (seconds)")
    parser.add_argument("--q-num", type=int, default=None, help="Limit rows for testing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        answer_columns=args.answer_columns,
        true_col=args.true_col,
        gt_csv=Path(args.gt_csv),
        no_gt_merge=args.no_gt_merge,
        model=args.model,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
        q_num=args.q_num,
    )


if __name__ == "__main__":
    main()
