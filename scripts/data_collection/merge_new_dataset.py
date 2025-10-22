#!/usr/bin/env python3
"""
新規データセットのマージスクリプト

目的:
- 新しく収集したQAデータ（agent-takaichi形式）を既存形式に変換
- 既存のmerged_collection.jsonとマージ
- 重複チェックと品質検証を実施
"""

import json
from datetime import datetime
from typing import List, Dict
import os

# ============================================================================
# 設定
# ============================================================================

# 入力ファイル
NEW_DATA_PATH = "/home/maniax/dev/agent-takaichi/output/高市早苗氏について_20251021_203326.json"
EXISTING_DATA_PATH = "data/processed/merged_collection.json"

# 出力ファイル
OUTPUT_PATH = f"data/processed/merged_collection_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# カテゴリーマッピング
CATEGORY_MAPPING = {
    "プロフィール": "CAT-01",  # 基本情報
    "学歴": "CAT-02",  # 学歴・経歴
    "政治経歴": "CAT-03",  # 政治活動
    "政策": "CAT-04",  # 政策・主張
    "実績": "CAT-05",  # 実績・業績
    "その他": "CAT-99",  # その他
}

# ============================================================================
# ヘルパー関数
# ============================================================================

def load_new_data(file_path: str) -> List[Dict]:
    """新規データを読み込み"""
    print(f"新規データ読み込み中: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  総QAペア数: {data.get('total_qa_pairs', 0)}")
    return data.get('qa_pairs', [])


def load_existing_data(file_path: str) -> List[Dict]:
    """既存データを読み込み"""
    print(f"\n既存データ読み込み中: {file_path}")
    if not os.path.exists(file_path):
        print(f"  既存データが見つかりません。新規作成します。")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  既存QAペア数: {len(data)}")
    return data


def convert_to_standard_format(new_item: Dict, index: int, existing_count: int) -> Dict:
    """新規データを標準形式に変換"""
    # カテゴリーマッピング
    category = new_item.get('category', 'その他')
    cat_code = CATEGORY_MAPPING.get(category, 'CAT-99')

    # ID生成（既存データの続きから）
    qa_id = f"TAKAICHI-QA-{str(existing_count + index + 1).zfill(4)}"

    # 回答を自然な文体に変換
    answer = new_item.get('answer', '')
    question = new_item.get('question', '')

    # 回答が短い場合は丁寧な文体に変換
    if answer and not answer.endswith('。') and not answer.endswith('です') and not answer.endswith('ます'):
        # 既に完結している回答はそのまま、そうでない場合は文体を整える
        if len(answer) < 50:
            answer = f"{answer}です。"

    # ソースURL
    source_url = new_item.get('source', '')
    source_type = 'wikipedia' if 'wikipedia' in source_url else 'web'

    return {
        "id": qa_id,
        "category": cat_code,
        "question": question,
        "answer": answer,
        "source": {
            "type": source_type,
            "url": source_url,
            "access_date": new_item.get('date', datetime.now().strftime('%Y-%m-%d')),
            "reliability": "B"  # デフォルトでB評価
        },
        "verification": {
            "verified": False,
            "verified_by": None,
            "verification_date": None,
            "cross_check_sources": []
        },
        "metadata": {
            "created_date": datetime.now().strftime('%Y-%m-%d'),
            "last_updated": datetime.now().strftime('%Y-%m-%d'),
            "version": "1.0",
            "tags": new_item.get('keywords', []),
            "original_category": category
        }
    }


def check_duplicate(new_item: Dict, existing_data: List[Dict]) -> bool:
    """重複チェック（質問の類似度で判定）"""
    new_question = new_item['question'].strip().lower()

    for existing_item in existing_data:
        existing_question = existing_item['question'].strip().lower()

        # 完全一致
        if new_question == existing_question:
            return True

        # 非常に類似（簡易的な類似度判定）
        # 質問の主要部分が一致する場合は重複とみなす
        new_q_simple = new_question.replace('？', '').replace('?', '').replace('さん', '').replace('氏', '')
        existing_q_simple = existing_question.replace('？', '').replace('?', '').replace('さん', '').replace('氏', '')

        if new_q_simple == existing_q_simple:
            return True

    return False


def categorize_stats(data: List[Dict]) -> Dict[str, int]:
    """カテゴリー別統計"""
    stats = {}
    for item in data:
        cat = item.get('category', 'CAT-99')
        if cat not in stats:
            stats[cat] = 0
        stats[cat] += 1
    return stats


# ============================================================================
# メイン処理
# ============================================================================

def main():
    print("=" * 80)
    print("新規データセットマージスクリプト")
    print("=" * 80)

    # データ読み込み
    new_data = load_new_data(NEW_DATA_PATH)
    existing_data = load_existing_data(EXISTING_DATA_PATH)

    existing_count = len(existing_data)

    # 新規データを標準形式に変換
    print("\n新規データを標準形式に変換中...")
    converted_data = []
    duplicates = 0

    for i, item in enumerate(new_data):
        # 標準形式に変換
        converted_item = convert_to_standard_format(item, i, existing_count)

        # 重複チェック
        if check_duplicate(converted_item, existing_data + converted_data):
            duplicates += 1
            print(f"  [重複] {converted_item['question'][:50]}...")
            continue

        converted_data.append(converted_item)

    print(f"\n✅ 変換完了")
    print(f"   新規データ: {len(new_data)}ペア")
    print(f"   変換成功: {len(converted_data)}ペア")
    print(f"   重複除外: {duplicates}ペア")

    # マージ
    print("\nデータマージ中...")
    merged_data = existing_data + converted_data

    print(f"✅ マージ完了")
    print(f"   既存データ: {existing_count}ペア")
    print(f"   追加データ: {len(converted_data)}ペア")
    print(f"   合計: {len(merged_data)}ペア")

    # カテゴリー別統計
    print("\nカテゴリー別統計:")
    stats = categorize_stats(merged_data)
    for cat, count in sorted(stats.items()):
        percentage = (count / len(merged_data)) * 100
        print(f"  {cat}: {count}ペア ({percentage:.1f}%)")

    # 保存
    print(f"\n保存中: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存完了")

    # サンプル表示
    print("\n📋 追加されたデータのサンプル（最初の5件）:")
    for i, item in enumerate(converted_data[:5], 1):
        print(f"\n--- サンプル {i} ---")
        print(f"ID: {item['id']}")
        print(f"カテゴリー: {item['category']} ({item['metadata'].get('original_category', '')})")
        print(f"質問: {item['question']}")
        print(f"回答: {item['answer'][:80]}...")

    # 既存のmerged_collection.jsonをバックアップして上書き
    print("\n既存ファイルのバックアップと更新...")
    if os.path.exists(EXISTING_DATA_PATH):
        backup_path = f"{EXISTING_DATA_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(EXISTING_DATA_PATH, backup_path)
        print(f"  バックアップ: {backup_path}")

    # 新しいデータを既存パスに保存
    with open(EXISTING_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"  更新: {EXISTING_DATA_PATH}")

    print("\n" + "=" * 80)
    print("🎉 データセットマージ完了")
    print("=" * 80)
    print(f"最終データセット: {EXISTING_DATA_PATH}")
    print(f"総QAペア数: {len(merged_data)}")
    print(f"バックアップファイル: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
