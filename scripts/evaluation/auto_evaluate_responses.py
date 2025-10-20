#!/usr/bin/env python3
"""
Week 4: 自動評価スクリプト

目的: 収集したモデル出力に対して、既知の事実に基づいた自動評価を実施
     正解データと比較して、is_good_response、human_rating、issues、better_responseを自動付与
"""

import json
import logging
from typing import Dict, List, Tuple

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ============================================================================
# 高市早苗氏に関する正解データ
# ============================================================================

CORRECT_FACTS = {
    "所属政党": "自由民主党",
    "生年月日": "1961年3月7日",
    "出身地": "奈良県",
    "学歴": "神戸大学経営学部卒業",
    "選挙区": "奈良2区",
    "初当選": "1993年",
    "大臣経験": ["総務大臣", "内閣府特命担当大臣（経済安全保障）"],
    "派閥": "清和政策研究会（安倍派）",
    "総裁選出馬": ["2021年", "2024年"],
}

# ============================================================================
# 評価関数
# ============================================================================

def evaluate_response(question: str, response: str) -> Tuple[bool, int, List[str], str]:
    """
    応答を評価

    Returns:
        (is_good_response, human_rating, issues, better_response)
    """
    issues = []
    better_response = None
    is_good = True
    rating = 5

    # 質問ごとの評価
    if "何党" in question or "所属" in question:
        if "自由民主党" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 1
            issues.append("所属政党が誤り")
            better_response = "高市早苗さんは自由民主党（自民党）に所属しています。"

    elif "生年月日" in question:
        if "1961" in response and "3月7日" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 1
            issues.append("生年月日が誤り")
            better_response = "高市早苗さんの生年月日は1961年3月7日です。"

    elif "出身地" in question:
        if "奈良" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 1
            issues.append("出身地が誤り")
            better_response = "高市早苗さんの出身地は奈良県です。"

    elif "学歴" in question:
        if "神戸大学" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 1
            issues.append("学歴が誤り")
            better_response = "高市早苗さんは神戸大学経営学部を卒業しています。"

    elif "選挙区" in question:
        if "奈良" in response and "2区" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 1
            issues.append("選挙区が誤り")
            better_response = "高市早苗さんは奈良2区から立候補しています。"

    elif "初当選" in question:
        if "1993" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 1
            issues.append("初当選年が誤り")
            better_response = "高市早苗さんの初当選は1993年です。"

    elif "大臣" in question and "ポスト" in question:
        if "総務大臣" in response or "総務相" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 2
            issues.append("大臣経験に誤りあり")
            better_response = "高市早苗さんは総務大臣や内閣府特命担当大臣（経済安全保障）などを歴任しています。"

    elif "派閥" in question:
        if "清和" in response or "安倍" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 2
            issues.append("派閥情報が不正確")
            better_response = "高市早苗さんは清和政策研究会（安倍派）に所属しています。"

    elif "総裁選" in question and "出馬" in question:
        if "2021" in response or "2024" in response:
            is_good = True
            rating = 5
        else:
            is_good = False
            rating = 2
            issues.append("総裁選出馬歴が不正確")
            better_response = "高市早苗さんは2021年と2024年の自民党総裁選挙に出馬しています。"

    # 一般的な評価基準
    if len(response) < 10:
        issues.append("応答が短すぎる")
        rating = min(rating, 2)
        is_good = False

    # 応答が空の場合
    if not response or response.strip() == "":
        is_good = False
        rating = 1
        issues.append("応答が空")
        better_response = "適切な情報を提供できませんでした。"

    return is_good, rating, issues, better_response


def auto_evaluate(input_file: str, output_file: str):
    """自動評価を実施"""

    logger.info("=" * 80)
    logger.info("Week 4: 自動評価開始")
    logger.info("=" * 80)
    logger.info(f"入力: {input_file}")
    logger.info(f"出力: {output_file}")

    # データ読み込み
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"\n総質問数: {len(data)}")

    # 評価実施
    evaluated_count = 0
    good_count = 0
    bad_count = 0

    for item in data:
        question = item['question']
        response = item['model_response']

        # 評価
        is_good, rating, issues, better_response = evaluate_response(question, response)

        # 結果を記録
        item['is_good_response'] = is_good
        item['human_rating'] = rating
        item['issues'] = issues
        item['better_response'] = better_response

        evaluated_count += 1
        if is_good:
            good_count += 1
        else:
            bad_count += 1

        # ログ出力
        status = "✅" if is_good else "❌"
        logger.info(f"{status} Q{item['question_id']}: {question[:40]}... (評価: {rating}/5)")
        if issues:
            logger.info(f"   問題点: {', '.join(issues)}")

    # 結果保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("📊 評価統計")
    logger.info("=" * 80)
    logger.info(f"  評価済み: {evaluated_count}/{len(data)}")
    logger.info(f"  良い応答: {good_count} ({good_count/len(data)*100:.1f}%)")
    logger.info(f"  悪い応答: {bad_count} ({bad_count/len(data)*100:.1f}%)")

    # 平均評価
    avg_rating = sum(item['human_rating'] for item in data) / len(data)
    logger.info(f"  平均評価: {avg_rating:.2f}/5.0")

    logger.info("\n" + "=" * 80)
    logger.info("✅ 自動評価完了")
    logger.info("=" * 80)
    logger.info(f"出力ファイル: {output_file}")

    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("使用方法: python auto_evaluate_responses.py <入力ファイル>")
        logger.error("例: python auto_evaluate_responses.py data/comparison/model_outputs_20251020_150646.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace('.json', '_evaluated.json')

    auto_evaluate(input_file, output_file)
