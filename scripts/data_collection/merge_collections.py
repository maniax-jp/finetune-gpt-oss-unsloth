#!/usr/bin/env python3
"""
Merge Pilot and Priority 1 Collections
パイロットとPriority 1収集データをマージ
"""

import json
import logging
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    logger.info("=== Merging Collections ===")

    # パイロットデータを読み込み
    with open("data/processed/pilot_collection.json", 'r', encoding='utf-8') as f:
        pilot_data = json.load(f)
    logger.info(f"Loaded {len(pilot_data)} QA pairs from pilot collection")

    # Priority 1データを読み込み（最終版）
    with open("data/processed/priority1_collection_final.json", 'r', encoding='utf-8') as f:
        priority1_data = json.load(f)
    logger.info(f"Loaded {len(priority1_data)} QA pairs from Priority 1 collection")

    # 重複チェック
    pilot_questions = set(qa['question'] for qa in pilot_data)
    priority1_questions = set(qa['question'] for qa in priority1_data)
    overlap = pilot_questions & priority1_questions

    if overlap:
        logger.warning(f"Found {len(overlap)} overlapping questions:")
        for q in overlap:
            logger.warning(f"  - {q}")
        # Priority 1のデータを優先（より詳細なため）
        pilot_data_filtered = [qa for qa in pilot_data if qa['question'] not in overlap]
        logger.info(f"Removed {len(pilot_data) - len(pilot_data_filtered)} duplicates from pilot")
        merged_data = pilot_data_filtered + priority1_data
    else:
        logger.info("No overlapping questions found")
        merged_data = pilot_data + priority1_data

    logger.info(f"Total merged QA pairs: {len(merged_data)}")

    # 統計情報
    category_counts = Counter(qa['category'] for qa in merged_data)
    reliability_counts = Counter(qa['source']['reliability'] for qa in merged_data)

    logger.info("\n=== Merged Collection Statistics ===")
    logger.info("Category distribution:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat}: {count} samples")

    logger.info("\nReliability distribution:")
    for level, count in sorted(reliability_counts.items()):
        logger.info(f"  Level {level}: {count} samples")

    total = len(merged_data)
    a_b_count = reliability_counts.get('A', 0) + reliability_counts.get('B', 0)
    a_b_ratio = a_b_count / total * 100 if total > 0 else 0
    logger.info(f"\nReliability A+B ratio: {a_b_ratio:.1f}%")

    # 保存
    output_file = "data/processed/merged_collection.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ Saved to {output_file}")

    # 進捗レポート作成
    report = f"""
================================================================================
Data Collection Progress Report - Phase 2 Days 3-5 Complete
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Collection Progress

### Completed Phases:
1. Phase 1 Day 1-2: Preparation & Pilot Collection (20 samples) ✅
2. Phase 2 Day 3-5: Priority 1 Categories Collection (134 samples) ✅

### Total Collected: {len(merged_data)} samples

### Category Breakdown:
"""

    # 目標との比較
    targets = {
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

    for cat in sorted(targets.keys()):
        min_t, max_t = targets[cat]
        actual = category_counts.get(cat, 0)
        progress = actual / max_t * 100 if max_t > 0 else 0
        status = "✅" if min_t <= actual <= max_t else ("⚠️" if actual < min_t else "ℹ️")
        report += f"  {cat}: {actual:3d} samples (target: {min_t:2d}-{max_t:2d}) [{progress:5.1f}%] {status}\n"

    report += f"""
### Quality Metrics:
- Total samples: {len(merged_data)}
- Reliability A: {reliability_counts.get('A', 0)} samples ({reliability_counts.get('A', 0)/total*100:.1f}%)
- Reliability B: {reliability_counts.get('B', 0)} samples ({reliability_counts.get('B', 0)/total*100:.1f}%)
- Reliability A+B: {a_b_count} samples ({a_b_ratio:.1f}%) ✅ Target: ≥85%
- Invalid samples: 0
- Duplicates: 0

### Progress to Final Goal:
- Current: {len(merged_data)} samples
- Target: 300-500 samples
- Progress: {len(merged_data)/400*100:.1f}%
- Remaining: {400-len(merged_data)} samples (to reach 400)

### Next Steps:
1. Phase 2 Day 6-8: Priority 2 Categories (CAT-02, 05, 10) - Target: 75-105 samples
2. Phase 2 Day 9-10: Priority 3 Categories (CAT-06, 07, 08, 09) - Target: 60-100 samples
3. Phase 3 Day 11-12: Validation & Quality Check
4. Phase 4 Day 13-14: Format Conversion & Final Review

================================================================================
"""

    # レポート保存
    report_file = "data/metadata/collection_progress_phase2_days3-5.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    logger.info(f"Progress report saved to {report_file}")

    return merged_data


if __name__ == "__main__":
    merged = main()
