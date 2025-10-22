#!/usr/bin/env python3
"""
非推論データにanalysisチャンネル自動付与スクリプト

Phase 10-2-B: 全データに固有名詞の日英対応を含める

機能:
1. 既存の非推論データ（455サンプル）を読み込み
2. 各サンプルに簡易analysisチャンネルを自動生成
3. 固有名詞辞書を使って日英対応を自動挿入
4. 推論データ（50サンプル）とマージ
5. 第7次訓練用データセットとして保存
"""

import json
import re
from typing import Dict, List, Tuple
from datetime import datetime

# ========================================
# 固有名詞辞書
# ========================================

PROPER_NOUNS = {
    "高市早苗": "Sanae Takaichi",
    "岸田文雄": "Fumio Kishida",
    "安倍晋三": "Shinzo Abe",
    "自由民主党": "Liberal Democratic Party (LDP)",
    "サナエノミクス": "Sanaenomics",
    "総務大臣": "Minister of Internal Affairs and Communications",
    "経済安全保障": "economic security",
    "内閣府特命担当大臣": "Minister of State for Special Missions",
    "奈良県": "Nara Prefecture",
    "早稲田大学": "Waseda University",
    "神戸大学": "Kobe University",
    "イリノイ大学": "University of Illinois",
    "松下政経塾": "Matsushita Institute of Government and Management",
    "防衛大臣": "Minister of Defense",
    "デジタル大臣": "Minister of Digital Affairs",
    "経済産業大臣": "Minister of Economy, Trade and Industry",
    "女性活躍": "women's empowerment",
    "地方創生": "regional revitalization",
    "憲法改正": "constitutional amendment",
    "靖国神社": "Yasukuni Shrine",
    "拉致問題": "abduction issue",
    "北朝鮮": "North Korea",
    "中国": "China",
    "アメリカ": "United States",
    "台湾": "Taiwan"
}

# ========================================
# 固有名詞検出関数
# ========================================

def detect_proper_nouns(text: str) -> List[Tuple[str, str]]:
    """
    テキストから固有名詞を検出し、日英対応のリストを返す

    Args:
        text: 検出対象のテキスト

    Returns:
        List[Tuple[str, str]]: [(日本語, 英語), ...] のリスト
    """
    detected = []
    for japanese, english in PROPER_NOUNS.items():
        if japanese in text:
            detected.append((japanese, english))
    return detected


# ========================================
# analysisチャンネル生成関数
# ========================================

def generate_analysis_channel(question: str, answer: str) -> str:
    """
    質問と回答から簡易analysisチャンネルを自動生成

    Args:
        question: ユーザーの質問
        answer: アシスタントの回答

    Returns:
        str: 生成されたanalysisチャンネルの内容
    """
    # 固有名詞を検出
    detected_nouns = detect_proper_nouns(question + " " + answer)

    # analysisチャンネルの構築
    analysis_parts = []

    # 1. 固有名詞の日英対応を明記
    if detected_nouns:
        noun_list = ", ".join([f"{jp} ({en})" for jp, en in detected_nouns])
        analysis_parts.append(f"Key entities: {noun_list}.")

    # 2. 質問の要約（日本語から英語へ）
    analysis_parts.append(f'User asks in Japanese: "{question}"')

    # 3. 回答の方針
    if "高市早苗" in question:
        analysis_parts.append("This question is about Sanae Takaichi, a Japanese politician and member of the Liberal Democratic Party (LDP).")

    # 4. 簡潔な回答戦略
    analysis_parts.append(f"I will provide a concise answer in Japanese based on the available information.")

    # analysisチャンネルの結合
    analysis = "\n\n".join(analysis_parts)

    return analysis


# ========================================
# データ変換関数
# ========================================

def convert_to_harmony_with_analysis(item: Dict) -> Dict:
    """
    非推論データをHarmony形式（analysisチャンネル付き）に変換

    Args:
        item: 元のデータアイテム {"question": "...", "answer": "..."}

    Returns:
        Dict: Harmony形式のデータアイテム
    """
    question = item.get("question", "")
    answer = item.get("answer", "")

    # analysisチャンネルを生成
    analysis = generate_analysis_channel(question, answer)

    # Harmony形式に変換
    harmony_item = {
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": (
                    f"<|channel|>analysis<|message|>{analysis}<|end|>"
                    f"<|channel|>final<|message|>{answer}"
                )
            }
        ]
    }

    return harmony_item


# ========================================
# メイン処理
# ========================================

def main():
    print(f"\n{'='*60}")
    print("非推論データにanalysisチャンネル自動付与")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 1. 既存の推論データをロード（50サンプル）
    print("[ステップ1] 推論データをロード...")
    reasoning_data = []
    reasoning_path = "dataset/takaichi_sanae_reasoning_v2_proper_nouns.jsonl"

    with open(reasoning_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reasoning_data.append(json.loads(line))

    print(f"  ✓ 推論データ: {len(reasoning_data)} サンプル\n")

    # 2. 非推論データをロード（455サンプル）
    print("[ステップ2] 非推論データをロード...")
    non_reasoning_path = "data/processed/merged_collection.json"

    with open(non_reasoning_path, 'r', encoding='utf-8') as f:
        non_reasoning_raw = json.load(f)

    print(f"  ✓ 非推論データ: {len(non_reasoning_raw)} サンプル\n")

    # 3. 非推論データにanalysisチャンネルを自動付与
    print("[ステップ3] analysisチャンネルを自動付与中...")
    non_reasoning_with_analysis = []

    for idx, item in enumerate(non_reasoning_raw, 1):
        harmony_item = convert_to_harmony_with_analysis(item)
        non_reasoning_with_analysis.append(harmony_item)

        if idx % 50 == 0:
            print(f"  処理中: {idx}/{len(non_reasoning_raw)} サンプル")

    print(f"  ✓ 変換完了: {len(non_reasoning_with_analysis)} サンプル\n")

    # 4. サンプル表示
    print("[ステップ4] 変換サンプルの確認...")
    sample = non_reasoning_with_analysis[0]
    print("  --- サンプル1 ---")
    print(f"  質問: {sample['messages'][0]['content']}")
    print(f"  回答（抜粋）: {sample['messages'][1]['content'][:200]}...")
    print()

    # 5. 推論データと非推論データをマージ
    print("[ステップ5] データをマージ...")
    # 推論データを先頭、非推論データを後ろに配置
    merged_data = reasoning_data + non_reasoning_with_analysis
    print(f"  ✓ マージ完了: {len(merged_data)} サンプル")
    print(f"    - 推論データ（手動作成）: {len(reasoning_data)} サンプル ({len(reasoning_data)/len(merged_data)*100:.1f}%)")
    print(f"    - 非推論データ（自動付与）: {len(non_reasoning_with_analysis)} サンプル ({len(non_reasoning_with_analysis)/len(merged_data)*100:.1f}%)")
    print()

    # 6. JSONL形式で保存
    print("[ステップ6] データセット保存...")
    output_path = "dataset/takaichi_sanae_full_reasoning_v7.jsonl"

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  ✓ 保存完了: {output_path}\n")

    # 7. 統計情報
    print(f"{'='*60}")
    print("変換統計")
    print(f"{'='*60}")
    print(f"総サンプル数: {len(merged_data)}")
    print(f"推論データ（手動）: {len(reasoning_data)} ({len(reasoning_data)/len(merged_data)*100:.1f}%)")
    print(f"非推論データ（自動）: {len(non_reasoning_with_analysis)} ({len(non_reasoning_with_analysis)/len(merged_data)*100:.1f}%)")
    print(f"analysisチャンネル付与率: 100.0%")
    print(f"{'='*60}\n")

    # 8. 固有名詞カバレッジ確認
    print("[ステップ7] 固有名詞カバレッジ確認...")
    takaichi_count = 0
    for item in merged_data:
        content = item['messages'][1]['content']
        if "高市早苗 (Sanae Takaichi)" in content or "Sanae Takaichi" in content:
            takaichi_count += 1

    print(f"  「高市早苗 (Sanae Takaichi)」を含むサンプル: {takaichi_count}/{len(merged_data)} ({takaichi_count/len(merged_data)*100:.1f}%)")
    print()

    print(f"{'='*60}")
    print("処理完了")
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    print("次のステップ:")
    print("  1. 第7次ファインチューニングスクリプトを作成")
    print("  2. 第7次ファインチューニングを実行")
    print("  3. 固有名詞認識率を検証（目標: 95%以上）")
    print()


if __name__ == "__main__":
    main()
