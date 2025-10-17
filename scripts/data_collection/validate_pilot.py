#!/usr/bin/env python3
"""
Validate Pilot Collection Data
パイロット収集データの検証
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
    logger.info("=== Validating Pilot Collection Data ===")

    # パイロットデータを読み込み
    pilot_file = "data/processed/pilot_collection.json"
    with open(pilot_file, 'r', encoding='utf-8') as f:
        pilot_data = json.load(f)

    logger.info(f"Loaded {len(pilot_data)} QA pairs from {pilot_file}")

    # バリデーター初期化
    validator = QAValidator()

    # 検証実行
    results = validator.validate_dataset(pilot_data)

    # レポート生成
    report = validator.generate_report(results)
    print("\n" + report)

    # レポートをファイルに保存
    report_file = "data/metadata/pilot_validation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"\nReport saved to {report_file}")

    # 重複チェック
    duplicates = validator.check_duplicates(pilot_data)
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
    logger.info(f"Target samples: 15+")
    logger.info(f"Actual samples: {stats.get('total_samples', 0)}")

    if stats.get('total_samples', 0) >= 15:
        logger.info("✅ Target achieved")
    else:
        logger.warning("⚠️ Below target")

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

    # 警告があれば表示
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
        "duplicates": duplicates
    }


if __name__ == "__main__":
    result = main()

    if result['valid']:
        print("\n" + "=" * 80)
        print("✅ PILOT COLLECTION PASSED ALL QUALITY CHECKS")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️ PILOT COLLECTION HAS QUALITY ISSUES - REVIEW REQUIRED")
        print("=" * 80)
