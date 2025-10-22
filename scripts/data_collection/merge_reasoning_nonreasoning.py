#!/usr/bin/env python3
"""
推論データと非推論データのマージスクリプト

目的:
  - 推論データ（50サンプル、固有名詞認識特化）
  - 非推論データ（455サンプル、既存データ）
  をマージして第6次学習用データセットを作成

構成:
  - 推論データ比率: 9.9% (50/505)
  - 目標: 固有名詞認識問題の解決
"""

import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# 設定
# ============================================================================

REASONING_DATA_PATH = "dataset/takaichi_sanae_reasoning_v2_proper_nouns.jsonl"
NON_REASONING_DATA_PATH = "data/processed/merged_collection.json"
OUTPUT_PATH = "dataset/takaichi_sanae_mixed_v6.jsonl"

# ============================================================================
# メイン処理
# ============================================================================

def load_reasoning_data(path: str):
    """推論データ（JSONL形式）を読み込み"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_non_reasoning_data(path: str):
    """非推論データ（JSON形式）を読み込み、Harmony形式に変換"""
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    harmony_data = []
    for item in raw_data:
        question = item.get('question', '')
        answer = item.get('answer', '')

        if not question or not answer:
            continue

        # Harmony形式に変換（Analysis channelなし）
        harmony_item = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        harmony_data.append(harmony_item)

    return harmony_data


def merge_datasets(reasoning_data, non_reasoning_data):
    """推論データと非推論データをマージ"""
    # シャッフルせず、推論データを先にして順番に配置
    # これにより、推論能力を優先的に学習できる可能性
    merged = reasoning_data + non_reasoning_data
    return merged


def main():
    print("=" * 80)
    print("推論データと非推論データのマージ")
    print("=" * 80)

    # 推論データ読み込み
    print(f"\n推論データ読み込み: {REASONING_DATA_PATH}")
    reasoning_data = load_reasoning_data(REASONING_DATA_PATH)
    print(f"  推論データ: {len(reasoning_data)}サンプル")

    # 非推論データ読み込み
    print(f"\n非推論データ読み込み: {NON_REASONING_DATA_PATH}")
    non_reasoning_data = load_non_reasoning_data(NON_REASONING_DATA_PATH)
    print(f"  非推論データ: {len(non_reasoning_data)}サンプル")

    # マージ
    print("\nマージ中...")
    merged_data = merge_datasets(reasoning_data, non_reasoning_data)

    # 統計
    total = len(merged_data)
    reasoning_ratio = (len(reasoning_data) / total) * 100

    print(f"\n【マージ結果】")
    print(f"  総サンプル数: {total}")
    print(f"  推論データ: {len(reasoning_data)} ({reasoning_ratio:.1f}%)")
    print(f"  非推論データ: {len(non_reasoning_data)} ({100-reasoning_ratio:.1f}%)")

    # 保存
    print(f"\n保存中: {OUTPUT_PATH}")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ マージ完了")
    print(f"\n【データセット構成】")
    print(f"  配置順: 推論データ（先頭50件）→ 非推論データ（残り455件）")
    print(f"  推論データの特徴:")
    print(f"    - 固有名詞の日英対応を明示")
    print(f"    - Analysis channel使用")
    print(f"    - 「高市早苗 = Sanae Takaichi」を常に明記")
    print(f"\n【次のステップ】")
    print(f"  1. 第6次学習スクリプトを作成")
    print(f"  2. 第5次のハイパーパラメータを継承")
    print(f"  3. 混合データセットで学習")
    print(f"  4. 固有名詞認識率の検証")
    print(f"\n【期待される効果】")
    print(f"  固有名詞認識率: 0% → 95%以上")
    print(f"  「高市早苗」→ \"Sanae Takaichi\" 認識: 100%")
    print(f"  基本的事実の正答率: 90%以上")


if __name__ == "__main__":
    main()
