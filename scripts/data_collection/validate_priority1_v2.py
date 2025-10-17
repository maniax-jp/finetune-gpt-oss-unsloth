#!/usr/bin/env python3
"""
Validate Priority 1 Collection V2
"""

import json
import logging
from data_validator import QAValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    logger.info("=== Validating Priority 1 Collection V2 ===")

    # V2データを読み込み
    v2_file = "data/processed/priority1_collection_v2.json"
    with open(v2_file, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)

    logger.info(f"Loaded {len(v2_data)} QA pairs from {v2_file}")

    # バリデーター初期化
    validator = QAValidator()

    # 検証実行
    results = validator.validate_dataset(v2_data)

    # レポート生成
    report = validator.generate_report(results)
    print("\n" + report)

    # レポート保存
    report_file = "data/metadata/priority1_validation_report_v2.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"\nReport saved to {report_file}")

    # 重複チェック
    duplicates = validator.check_duplicates(v2_data)
    if duplicates:
        logger.warning(f"\n⚠️ Found {len(duplicates)} duplicate(s)")
        for dup in duplicates:
            logger.warning(f"  Duplicate: {dup['question']}")
    else:
        logger.info("\n✅ No duplicates found")

    # 品質チェックサマリー
    logger.info("\n=== Quality Check Summary ===")

    if results['invalid'] == 0:
        logger.info("✅ All QA pairs are valid")
    else:
        logger.warning(f"⚠️ {results['invalid']} invalid QA pair(s) found")

    stats = results.get('statistics', {})

    # 目標との比較
    logger.info("\n=== Target Comparison ===")
    category_targets = {
        'CAT-01': (30, 40),
        'CAT-03': (40, 60),
        'CAT-04': (60, 80)
    }

    category_dist = stats.get('category_distribution', {})
    total_on_target = 0

    for cat, (min_target, max_target) in category_targets.items():
        actual = category_dist.get(cat, 0)
        logger.info(f"{cat}: {actual} samples (target: {min_target}-{max_target})")

        if min_target <= actual <= max_target:
            logger.info(f"  ✅ Within target range")
            total_on_target += 1
        elif actual < min_target:
            logger.warning(f"  ⚠️ Below target (need {min_target - actual} more)")
        else:
            logger.info(f"  ℹ️ Above target range (+{actual - max_target})")

    logger.info(f"\nCategories on target: {total_on_target}/3")

    # 信頼性レベルチェック
    rel_dist = stats.get('reliability_distribution', {})
    total = stats.get('total_samples', 0)
    if total:
        a_b_count = rel_dist.get('A', 0) + rel_dist.get('B', 0)
        a_b_ratio = a_b_count / total * 100
        logger.info(f"\nReliability A+B ratio: {a_b_ratio:.1f}%")

        if a_b_ratio >= 85:
            logger.info("✅ High reliability ratio (target: ≥85%)")
        else:
            logger.warning(f"⚠️ Low reliability ratio (target: ≥85%, actual: {a_b_ratio:.1f}%)")

    # 警告表示
    if results['warnings']:
        logger.info("\n=== Warnings ===")
        for warning in results['warnings']:
            logger.warning(f"  {warning}")

    logger.info("\n=== Validation Complete ===")

    # 結果を返す
    return {
        "valid": results['invalid'] == 0 and not duplicates,
        "total_samples": stats.get('total_samples', 0),
        "validation_results": results,
        "duplicates": duplicates,
        "on_target_categories": total_on_target
    }


if __name__ == "__main__":
    result = main()

    if result['valid'] and result['on_target_categories'] == 3:
        print("\n" + "=" * 80)
        print("✅ PRIORITY 1 COLLECTION V2 PASSED ALL QUALITY CHECKS")
        print(f"✅ ALL 3 CATEGORIES MEET TARGET RANGES")
        print("=" * 80)
    elif result['valid']:
        print("\n" + "=" * 80)
        print("✅ PRIORITY 1 COLLECTION V2 IS VALID")
        print(f"ℹ️ {result['on_target_categories']}/3 categories meet target ranges")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️ PRIORITY 1 COLLECTION V2 HAS QUALITY ISSUES")
        print("=" * 80)
