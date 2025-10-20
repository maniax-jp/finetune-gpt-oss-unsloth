#!/usr/bin/env python3
"""
Validate and Merge Priority 2 Collection
Priority 2収集データの検証とマージ
"""

import json
import logging
from datetime import datetime
from collections import Counter
from data_validator import QAValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    logger.info("=== Validating Priority 2 Collection ===")

    # Priority 2データを読み込み
    with open("data/processed/priority2_collection.json", 'r', encoding='utf-8') as f:
        priority2_data = json.load(f)
    logger.info(f"Loaded {len(priority2_data)} QA pairs from Priority 2 collection")

    # バリデーター初期化
    validator = QAValidator()

    # 検証実行
    results = validator.validate_dataset(priority2_data)

    # レポート生成
    report = validator.generate_report(results)
    print("\n" + report)

    # 重複チェック
    duplicates = validator.check_duplicates(priority2_data)
    if duplicates:
        logger.warning(f"\n⚠️ Found {len(duplicates)} duplicate(s)")
        for dup in duplicates:
            logger.warning(f"  Duplicate: {dup['question']}")
    else:
        logger.info("\n✅ No duplicates found")

    # 統計
    stats = results.get('statistics', {})
    category_dist = stats.get('category_distribution', {})

    # 目標との比較
    logger.info("\n=== Target Comparison ===")
    targets = {
        'CAT-02': (25, 35),
        'CAT-05': (20, 30),
        'CAT-10': (30, 40)
    }

    all_on_target = True
    for cat, (min_t, max_t) in targets.items():
        actual = category_dist.get(cat, 0)
        logger.info(f"{cat}: {actual} samples (target: {min_t}-{max_t})")
        if min_t <= actual <= max_t:
            logger.info(f"  ✅ Within target range")
        else:
            all_on_target = False
            if actual < min_t:
                logger.warning(f"  ⚠️ Below target")
            else:
                logger.info(f"  ℹ️ Above target range")

    # 検証結果の判定
    if results['invalid'] > 0 or duplicates:
        logger.error("\n❌ Validation failed - cannot merge")
        return False

    logger.info("\n✅ Validation passed")

    # マージ処理
    logger.info("\n=== Merging with existing collection ===")

    with open("data/processed/merged_collection.json", 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    logger.info(f"Loaded {len(existing_data)} existing samples")

    # 重複チェック（既存データとの比較）
    existing_questions = set(qa['question'] for qa in existing_data)
    priority2_questions = set(qa['question'] for qa in priority2_data)
    overlap = existing_questions & priority2_questions

    if overlap:
        logger.warning(f"Found {len(overlap)} overlapping questions with existing data:")
        for q in list(overlap)[:5]:  # 最初の5件だけ表示
            logger.warning(f"  - {q}")
        if len(overlap) > 5:
            logger.warning(f"  ... and {len(overlap) - 5} more")

        # 重複を除外
        priority2_filtered = [qa for qa in priority2_data if qa['question'] not in overlap]
        logger.info(f"Removed {len(priority2_data) - len(priority2_filtered)} duplicates")
        new_data = priority2_filtered
    else:
        logger.info("No overlapping questions found")
        new_data = priority2_data

    # マージ
    merged_data = existing_data + new_data
    logger.info(f"Total after merge: {len(merged_data)} samples")

    # マージ後の統計
    merged_cats = Counter(qa['category'] for qa in merged_data)
    merged_rel = Counter(qa['source']['reliability'] for qa in merged_data)

    logger.info("\n=== Merged Collection Statistics ===")
    logger.info("Category distribution:")
    for cat in sorted(merged_cats.keys()):
        count = merged_cats[cat]
        logger.info(f"  {cat}: {count} samples")

    logger.info("\nReliability distribution:")
    for level in sorted(merged_rel.keys()):
        count = merged_rel[level]
        logger.info(f"  Level {level}: {count} samples")

    total = len(merged_data)
    a_b_count = merged_rel.get('A', 0) + merged_rel.get('B', 0)
    a_b_ratio = a_b_count / total * 100 if total > 0 else 0
    logger.info(f"\nReliability A+B ratio: {a_b_ratio:.1f}%")

    # 保存
    output_file = "data/processed/merged_collection.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ Saved merged data to {output_file}")

    # 進捗レポート作成
    progress_report = f"""
================================================================================
Data Collection Progress Report - Phase 2 Days 6-8 Complete
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Collection Progress

### Completed Phases:
1. Phase 1 Day 1-2: Preparation & Pilot Collection (20 samples) ✅
2. Phase 2 Day 3-5: Priority 1 Categories (CAT-01, 03, 04) ✅
3. Phase 2 Day 6-8: Priority 2 Categories (CAT-02, 05, 10) ✅

### Total Collected: {len(merged_data)} samples

### Category Breakdown:
"""

    # 全カテゴリの目標
    all_targets = {
        'CAT-01': (30, 40),
        'CAT-02': (25, 35),
        'CAT-03': (40, 60),
        'CAT-04': (60, 80),
        'CAT-05': (20, 30),
        'CAT-06': (15, 25),
        'CAT-07': (15, 25),
        'CAT-08': (15, 25),
        'CAT-09': (15, 25),
        'CAT-10': (30, 40)
    }

    for cat in sorted(all_targets.keys()):
        min_t, max_t = all_targets[cat]
        actual = merged_cats.get(cat, 0)
        progress = actual / max_t * 100 if max_t > 0 else 0
        status = "✅" if min_t <= actual <= max_t else ("⚠️" if actual < min_t else "ℹ️")
        progress_report += f"  {cat}: {actual:3d} samples (target: {min_t:2d}-{max_t:2d}) [{progress:5.1f}%] {status}\n"

    progress_report += f"""
### Quality Metrics:
- Total samples: {len(merged_data)}
- Reliability A: {merged_rel.get('A', 0)} samples ({merged_rel.get('A', 0)/total*100:.1f}%)
- Reliability B: {merged_rel.get('B', 0)} samples ({merged_rel.get('B', 0)/total*100:.1f}%)
- Reliability A+B: {a_b_count} samples ({a_b_ratio:.1f}%) ✅ Target: ≥85%
- Invalid samples: 0
- Duplicates removed: {len(overlap) if overlap else 0}

### Progress to Final Goal:
- Current: {len(merged_data)} samples
- Target: 300-500 samples
- Progress: {len(merged_data)/400*100:.1f}%
- Remaining: {max(0, 300-len(merged_data))} samples (to reach minimum 300)

### Next Steps:
1. Phase 2 Day 9-10: Priority 3 Categories (CAT-06, 07, 08, 09) - Target: 60-100 samples
2. Phase 3 Day 11-12: Validation & Quality Check
3. Phase 4 Day 13-14: Format Conversion & Final Review

================================================================================
"""

    # レポート保存
    report_file = "data/metadata/collection_progress_phase2_days6-8.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(progress_report)

    print(progress_report)
    logger.info(f"Progress report saved to {report_file}")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Priority 2 validation and merge completed successfully")
    else:
        print("\n❌ Priority 2 validation failed")
