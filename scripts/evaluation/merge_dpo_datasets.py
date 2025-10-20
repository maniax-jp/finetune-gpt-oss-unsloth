#!/usr/bin/env python3
"""
Week 5 タスク1: DPOデータセットマージスクリプト

目的:
- 既存のDPOデータセット（7ペア）と新規データセット（41ペア）をマージ
- 重複削除と品質チェックを実施
- 最終的なDPOデータセット（data/comparison/dpo_dataset_final.jsonl）を生成
"""

import json
from typing import List, Dict
from datetime import datetime
import os
import argparse

# ============================================================================
# ヘルパー関数
# ============================================================================

def load_jsonl(file_path: str) -> List[Dict]:
    """JSONL形式のファイルを読み込み"""
    if not os.path.exists(file_path):
        print(f"⚠️  ファイルが見つかりません: {file_path}")
        return []

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def remove_duplicates(datasets: List[List[Dict]]) -> List[Dict]:
    """重複削除（promptが同じものを除外）"""
    seen_prompts = set()
    unique_data = []

    for dataset in datasets:
        for item in dataset:
            prompt = item.get("prompt", "")
            if prompt and prompt not in seen_prompts:
                seen_prompts.add(prompt)
                unique_data.append(item)
            elif prompt in seen_prompts:
                print(f"  - 重複削除: {prompt[:50]}...")

    return unique_data


def validate_dpo_pair(item: Dict) -> bool:
    """DPOペアの品質チェック"""
    # 必須フィールドの確認
    if "prompt" not in item or "chosen" not in item or "rejected" not in item:
        return False

    prompt = item["prompt"].strip()
    chosen = item["chosen"].strip()
    rejected = item["rejected"].strip()

    # 空文字列チェック
    if not prompt or not chosen or not rejected:
        return False

    # chosenとrejectedが同じ場合は無効
    if chosen == rejected:
        print(f"  - 無効なペア（chosen=rejected）: {prompt[:50]}...")
        return False

    # 最小文字数チェック
    if len(prompt) < 5 or len(chosen) < 5 or len(rejected) < 5:
        print(f"  - 無効なペア（短すぎる）: {prompt[:50]}...")
        return False

    return True


def categorize_pairs(data: List[Dict]) -> Dict[str, int]:
    """ペアをカテゴリー別に分類して統計情報を生成"""
    categories = {
        "生年月日": 0,
        "出身地": 0,
        "学歴・大学": 0,
        "選挙区": 0,
        "大臣経験": 0,
        "初当選": 0,
        "総裁選": 0,
        "政策・主張": 0,
        "その他": 0,
    }

    for item in data:
        prompt = item["prompt"].lower()
        categorized = False

        if "生年月日" in prompt or "いつ生まれ" in prompt or "何年生まれ" in prompt:
            categories["生年月日"] += 1
            categorized = True
        elif "出身" in prompt and ("地" in prompt or "どこ" in prompt):
            categories["出身地"] += 1
            categorized = True
        elif "学歴" in prompt or "大学" in prompt or "出身大学" in prompt:
            categories["学歴・大学"] += 1
            categorized = True
        elif "選挙区" in prompt:
            categories["選挙区"] += 1
            categorized = True
        elif "大臣" in prompt:
            categories["大臣経験"] += 1
            categorized = True
        elif "初当選" in prompt:
            categories["初当選"] += 1
            categorized = True
        elif "総裁選" in prompt:
            categories["総裁選"] += 1
            categorized = True
        elif "政策" in prompt or "主張" in prompt or "経済" in prompt or "安全保障" in prompt:
            categories["政策・主張"] += 1
            categorized = True

        if not categorized:
            categories["その他"] += 1

    return categories


# ============================================================================
# メイン処理
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DPOデータセットマージスクリプト")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/comparison/dpo_dataset_20251020_151239.jsonl",
            "data/comparison/dpo_dataset_expanded_20251020_152655.jsonl"
        ],
        help="入力するJSONLファイルのリスト"
    )
    parser.add_argument(
        "--output",
        default="data/comparison/dpo_dataset_final.jsonl",
        help="出力先のJSONLファイル"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Week 5 タスク1: DPOデータセットマージ")
    print("=" * 80)

    # データセット読み込み
    datasets = []
    total_items = 0

    for i, input_file in enumerate(args.inputs, 1):
        print(f"\n[{i}/{len(args.inputs)}] 読み込み中: {input_file}")
        data = load_jsonl(input_file)
        print(f"  ✅ {len(data)}ペア読み込み完了")
        datasets.append(data)
        total_items += len(data)

    print(f"\n総読み込みペア数: {total_items}")

    # 重複削除
    print("\n重複削除中...")
    merged_data = remove_duplicates(datasets)
    print(f"  ✅ 重複削除後: {len(merged_data)}ペア")

    # 品質チェック
    print("\n品質チェック中...")
    valid_data = [item for item in merged_data if validate_dpo_pair(item)]
    invalid_count = len(merged_data) - len(valid_data)
    if invalid_count > 0:
        print(f"  ⚠️  無効なペアを{invalid_count}件除外")
    print(f"  ✅ 有効なペア: {len(valid_data)}ペア")

    # カテゴリー別統計
    print("\nカテゴリー別統計:")
    categories = categorize_pairs(valid_data)
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / len(valid_data)) * 100
            print(f"  - {category}: {count}件 ({percentage:.1f}%)")

    # 保存
    print(f"\n保存中: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 保存完了: {args.output}")
    print(f"   最終ペア数: {len(valid_data)}")

    # サンプル表示
    print("\n📋 サンプル（最初の3ペア）:")
    for i, item in enumerate(valid_data[:3], 1):
        print(f"\n--- ペア {i} ---")
        print(f"質問: {item['prompt']}")
        print(f"正解: {item['chosen'][:80]}...")
        print(f"誤答: {item['rejected'][:80]}...")

    # サマリー
    print("\n" + "=" * 80)
    print("📊 マージ完了サマリー")
    print("=" * 80)
    print(f"入力ファイル数: {len(args.inputs)}")
    print(f"総入力ペア数: {total_items}")
    print(f"重複削除後: {len(merged_data)}ペア")
    print(f"品質チェック後: {len(valid_data)}ペア")
    print(f"最終出力: {args.output}")

    # 目標達成チェック
    if len(valid_data) >= 100:
        print("\n🎉 目標達成！ 100ペア以上のDPOデータセットを作成しました！")
    elif len(valid_data) >= 50:
        print(f"\n✅ 良好な進捗です。現在{len(valid_data)}ペア（目標100ペアまであと{100-len(valid_data)}ペア）")
    else:
        print(f"\n⚠️  追加データ収集が必要です。現在{len(valid_data)}ペア（目標100ペアまであと{100-len(valid_data)}ペア）")

    print("=" * 80)


if __name__ == "__main__":
    main()
