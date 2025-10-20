#!/usr/bin/env python3
"""
Week 4: 比較データセット作成スクリプト

目的: 人間評価済みのモデル出力から、DPOトレーニング用の
     chosen/rejected ペアを作成

入力: collect_model_outputs.py の出力（人間評価済み）
出力: DPO形式の比較データセット
"""

import json
import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/create_comparison_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定パラメータ
# ============================================================================

# 入力ファイル（人間評価済み）
INPUT_FILE = None  # コマンドライン引数で指定

# 出力ファイル
OUTPUT_DIR = "data/comparison"
COMPARISON_DATASET_FILE = f"{OUTPUT_DIR}/comparison_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# DPO用のJSONL形式
DPO_DATASET_FILE = f"{OUTPUT_DIR}/dpo_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# ============================================================================
# データセット作成
# ============================================================================

def load_evaluated_data(input_file: str) -> List[Dict]:
    """人間評価済みデータを読み込み"""

    logger.info("=" * 60)
    logger.info("評価済みデータ読み込み")
    logger.info("=" * 60)
    logger.info(f"入力ファイル: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"✅ {len(data)}件のデータを読み込みました")

    return data


def create_comparison_pairs(data: List[Dict]) -> List[Dict]:
    """比較ペアを作成"""

    logger.info("\n" + "=" * 60)
    logger.info("比較ペア作成")
    logger.info("=" * 60)

    comparison_pairs = []
    skipped = 0

    for item in data:
        question = item["question"]
        model_response = item["model_response"]
        is_good = item.get("is_good_response")
        better_response = item.get("better_response")

        # 評価が未記入の場合はスキップ
        if is_good is None:
            skipped += 1
            continue

        # 悪い応答の場合、better_responseが必須
        if not is_good:
            if not better_response:
                logger.warning(f"⚠️  質問「{question}」: better_responseが未記入")
                skipped += 1
                continue

            # chosen (良い応答) と rejected (悪い応答) のペア
            comparison_pairs.append({
                "prompt": question,
                "chosen": better_response,
                "rejected": model_response,
                "question_id": item["question_id"],
                "human_rating": item.get("human_rating"),
                "issues": item.get("issues", []),
            })

        else:
            # 良い応答の場合、そのまま使用
            # ただし、rejected として使える悪い応答例が必要
            # → 別の質問の悪い応答を使うか、手動で作成
            pass

    logger.info(f"✅ {len(comparison_pairs)}ペアを作成")
    logger.info(f"   スキップ: {skipped}件")

    return comparison_pairs


def save_comparison_dataset(pairs: List[Dict]):
    """比較データセットを保存"""

    logger.info("\n" + "=" * 60)
    logger.info("比較データセット保存")
    logger.info("=" * 60)

    # JSON形式で保存
    with open(COMPARISON_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ JSON形式保存完了: {COMPARISON_DATASET_FILE}")

    # JSONL形式（DPO用）で保存
    with open(DPO_DATASET_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            # DPO形式: prompt, chosen, rejected
            dpo_item = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            f.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")

    logger.info(f"✅ JSONL形式保存完了: {DPO_DATASET_FILE}")

    # 統計情報
    logger.info("\n📊 データセット統計:")
    logger.info(f"   総ペア数: {len(pairs)}")

    # 品質分析
    ratings = [p.get("human_rating") for p in pairs if p.get("human_rating")]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        logger.info(f"   平均評価: {avg_rating:.2f} / 5.0")

    # 問題点の集計
    all_issues = []
    for p in pairs:
        all_issues.extend(p.get("issues", []))

    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues)
        logger.info("\n   頻出問題:")
        for issue, count in issue_counts.most_common(5):
            logger.info(f"     - {issue}: {count}件")


def main():
    import sys

    logger.info("=" * 80)
    logger.info("Week 4: 比較データセット作成")
    logger.info("=" * 80)

    # コマンドライン引数チェック
    if len(sys.argv) < 2:
        logger.error("❌ 使用方法: python create_comparison_dataset.py <評価済みファイル>")
        logger.error("   例: python create_comparison_dataset.py data/comparison/model_outputs_20251020_123456.json")
        sys.exit(1)

    input_file = sys.argv[1]

    if not Path(input_file).exists():
        logger.error(f"❌ ファイルが見つかりません: {input_file}")
        sys.exit(1)

    logger.info(f"\n入力ファイル: {input_file}")
    logger.info(f"出力ディレクトリ: {OUTPUT_DIR}\n")

    # データ読み込み
    evaluated_data = load_evaluated_data(input_file)

    # 比較ペア作成
    comparison_pairs = create_comparison_pairs(evaluated_data)

    if not comparison_pairs:
        logger.error("❌ 比較ペアが作成できませんでした")
        logger.error("   人間評価（is_good_response, better_response）を記入してください")
        sys.exit(1)

    # データセット保存
    save_comparison_dataset(comparison_pairs)

    logger.info("\n" + "=" * 80)
    logger.info("🎉 Week 4: 比較データセット作成完了")
    logger.info("=" * 80)
    logger.info("\n次のステップ (Week 5):")
    logger.info("  1. DPOトレーニングの実施:")
    logger.info(f"     python scripts/training/dpo_training.py")
    logger.info("  2. または PPOトレーニング:")
    logger.info(f"     python scripts/training/ppo_training.py")


if __name__ == "__main__":
    main()
