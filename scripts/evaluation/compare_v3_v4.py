#!/usr/bin/env python3
"""
第3次モデル（DPO適用後）と第4次モデル（データセット拡充後）の比較評価

目的: 両モデルの出力を比較し、データセット拡充の効果を検証する

比較観点:
  1. 基本的事実の正確性
  2. 応答の一貫性
  3. 応答の詳細度
  4. ハルシネーション（幻覚）の発生頻度
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

# ============================================================================
# 設定
# ============================================================================

V3_OUTPUT = "data/comparison/model_outputs_v3_20251020_155521.json"
V4_OUTPUT = "data/comparison/model_outputs_v4_20251022_101056.json"

# 既知の事実（検証用）
KNOWN_FACTS = {
    "生年月日": "1961年3月7日",
    "出身地": "奈良県",
    "学歴": "神戸大学経営学部",
    "初当選": "1993年",
    "政党": "自由民主党",
    "選挙区": "奈良2区",
}

# エラーパターン（第2次モデルで確認されたもの）
ERROR_PATTERNS = {
    "生年月日": ["1960年", "1962年", "1970年", "1959年"],
    "出身地": ["岩手県", "名古屋", "東京", "大阪", "京都"],
    "学歴": ["東京大学", "早稲田大学", "日本大学", "慶應義塾大学"],
    "初当選": ["2009年", "2003年", "2000年", "1996年"],
    "選挙区": ["東京都", "大阪府", "神奈川県"],
}

# ============================================================================
# データ読み込み
# ============================================================================

def load_outputs():
    """両モデルの出力を読み込み"""
    print("=" * 80)
    print("第3次 vs 第4次モデル比較評価")
    print("=" * 80)

    with open(V3_OUTPUT, 'r', encoding='utf-8') as f:
        v3_data = json.load(f)

    with open(V4_OUTPUT, 'r', encoding='utf-8') as f:
        v4_data = json.load(f)

    # 新形式（metadataあり）と旧形式（配列のみ）の両方に対応
    v3_results = v3_data.get('results', v3_data) if isinstance(v3_data, dict) else v3_data
    v4_results = v4_data.get('results', v4_data) if isinstance(v4_data, dict) else v4_data

    print(f"\n第3次モデル出力: {len(v3_results)}件")
    print(f"第4次モデル出力: {len(v4_results)}件")

    return v3_results, v4_results


# ============================================================================
# 比較分析
# ============================================================================

def check_factual_errors(response: str, question: str) -> List[str]:
    """事実誤認エラーをチェック"""
    errors = []

    for fact_type, error_list in ERROR_PATTERNS.items():
        for error_value in error_list:
            if error_value in response:
                errors.append(f"{fact_type}の誤り: {error_value}")

    return errors


def calculate_similarity(text1: str, text2: str) -> float:
    """2つのテキストの類似度を計算"""
    return SequenceMatcher(None, text1, text2).ratio()


def compare_responses(v3_results: List[Dict], v4_results: List[Dict]):
    """応答を詳細比較"""
    print("\n" + "=" * 80)
    print("詳細比較分析")
    print("=" * 80)

    comparison_results = []
    v3_errors_total = 0
    v4_errors_total = 0

    for v3, v4 in zip(v3_results, v4_results):
        question = v3['question']
        v3_response = v3.get('model_response', '')
        v4_response = v4.get('model_response', '')

        # エラーチェック
        v3_errors = check_factual_errors(v3_response, question)
        v4_errors = check_factual_errors(v4_response, question)

        v3_errors_total += len(v3_errors)
        v4_errors_total += len(v4_errors)

        # 類似度計算
        similarity = calculate_similarity(v3_response, v4_response)

        # 応答長
        v3_length = len(v3_response)
        v4_length = len(v4_response)

        comparison_results.append({
            "question": question,
            "v3_response": v3_response,
            "v4_response": v4_response,
            "v3_errors": v3_errors,
            "v4_errors": v4_errors,
            "similarity": similarity,
            "v3_length": v3_length,
            "v4_length": v4_length,
            "improved": len(v4_errors) < len(v3_errors),
            "degraded": len(v4_errors) > len(v3_errors),
        })

    return comparison_results, v3_errors_total, v4_errors_total


def print_summary(comparison_results: List[Dict], v3_errors: int, v4_errors: int):
    """サマリーを表示"""
    print("\n" + "=" * 80)
    print("📊 比較サマリー")
    print("=" * 80)

    total = len(comparison_results)
    improved = sum(1 for r in comparison_results if r['improved'])
    degraded = sum(1 for r in comparison_results if r['degraded'])
    unchanged = total - improved - degraded

    avg_similarity = sum(r['similarity'] for r in comparison_results) / total
    avg_v3_length = sum(r['v3_length'] for r in comparison_results) / total
    avg_v4_length = sum(r['v4_length'] for r in comparison_results) / total

    print(f"\n【総合統計】")
    print(f"  総質問数: {total}")
    print(f"  改善: {improved} ({improved/total*100:.1f}%)")
    print(f"  悪化: {degraded} ({degraded/total*100:.1f}%)")
    print(f"  変化なし: {unchanged} ({unchanged/total*100:.1f}%)")

    print(f"\n【エラー数】")
    print(f"  第3次モデル: {v3_errors}件")
    print(f"  第4次モデル: {v4_errors}件")
    print(f"  変化: {v4_errors - v3_errors:+d}件 ({(v4_errors - v3_errors)/v3_errors*100:+.1f}%)")

    print(f"\n【応答特性】")
    print(f"  平均類似度: {avg_similarity:.3f}")
    print(f"  平均応答長（第3次）: {avg_v3_length:.1f}文字")
    print(f"  平均応答長（第4次）: {avg_v4_length:.1f}文字")
    print(f"  応答長変化: {avg_v4_length - avg_v3_length:+.1f}文字")


def print_detailed_differences(comparison_results: List[Dict]):
    """詳細な差分を表示"""
    print("\n" + "=" * 80)
    print("🔍 改善・悪化した質問の詳細")
    print("=" * 80)

    # 改善例
    improved = [r for r in comparison_results if r['improved']]
    if improved:
        print(f"\n✅ 改善された質問（{len(improved)}件）")
        print("-" * 80)
        for i, r in enumerate(improved[:5], 1):  # 最初の5件のみ
            print(f"\n{i}. {r['question']}")
            print(f"   第3次エラー: {r['v3_errors']}")
            print(f"   第4次エラー: {r['v4_errors']}")
            print(f"   第3次応答: {r['v3_response'][:100]}...")
            print(f"   第4次応答: {r['v4_response'][:100]}...")

    # 悪化例
    degraded = [r for r in comparison_results if r['degraded']]
    if degraded:
        print(f"\n❌ 悪化した質問（{len(degraded)}件）")
        print("-" * 80)
        for i, r in enumerate(degraded[:5], 1):  # 最初の5件のみ
            print(f"\n{i}. {r['question']}")
            print(f"   第3次エラー: {r['v3_errors']}")
            print(f"   第4次エラー: {r['v4_errors']}")
            print(f"   第3次応答: {r['v3_response'][:100]}...")
            print(f"   第4次応答: {r['v4_response'][:100]}...")


def save_detailed_report(comparison_results: List[Dict], v3_errors: int, v4_errors: int):
    """詳細レポートを保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/comparison/v3_v4_comparison_{timestamp}.json"

    total = len(comparison_results)
    improved = sum(1 for r in comparison_results if r['improved'])
    degraded = sum(1 for r in comparison_results if r['degraded'])

    report = {
        "metadata": {
            "comparison_date": datetime.now().isoformat(),
            "v3_output": V3_OUTPUT,
            "v4_output": V4_OUTPUT,
            "total_questions": total,
        },
        "summary": {
            "improved_count": improved,
            "improved_percentage": improved / total * 100,
            "degraded_count": degraded,
            "degraded_percentage": degraded / total * 100,
            "unchanged_count": total - improved - degraded,
            "v3_errors_total": v3_errors,
            "v4_errors_total": v4_errors,
            "error_change": v4_errors - v3_errors,
            "error_change_percentage": (v4_errors - v3_errors) / v3_errors * 100 if v3_errors > 0 else 0,
        },
        "detailed_results": comparison_results,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 詳細レポート保存: {output_file}")


# ============================================================================
# メイン処理
# ============================================================================

def main():
    """メイン処理"""
    # データ読み込み
    v3_results, v4_results = load_outputs()

    # 比較分析
    comparison_results, v3_errors, v4_errors = compare_responses(v3_results, v4_results)

    # サマリー表示
    print_summary(comparison_results, v3_errors, v4_errors)

    # 詳細差分表示
    print_detailed_differences(comparison_results)

    # レポート保存
    save_detailed_report(comparison_results, v3_errors, v4_errors)

    print("\n" + "=" * 80)
    print("🎯 結論")
    print("=" * 80)

    improved_pct = sum(1 for r in comparison_results if r['improved']) / len(comparison_results) * 100
    error_change = v4_errors - v3_errors

    if error_change < 0:
        print(f"\n✅ データセット拡充により、エラーが{abs(error_change)}件減少しました")
        print(f"   改善率: {improved_pct:.1f}%")
        print(f"\n   第4次モデルは第3次モデルより優れています。")
    elif error_change == 0:
        print(f"\n⚠️  エラー数は変化なし")
        print(f"   データセット拡充の効果は限定的です。")
    else:
        print(f"\n❌ データセット拡充にも関わらず、エラーが{error_change}件増加しました")
        print(f"   これは以下の可能性を示唆します：")
        print(f"   1. 追加データの質が低い")
        print(f"   2. モデルが過学習している")
        print(f"   3. 基本的事実の学習に別のアプローチが必要")

    print("\n次のステップの提案:")
    print("  1. データの質の再確認")
    print("  2. 基本的事実のみに特化した追加トレーニング")
    print("  3. より強力なDPO訓練の実施")
    print("=" * 80)


if __name__ == "__main__":
    main()
