#!/usr/bin/env python3
"""
Convert to Harmony Format for Phase 4
Phase 4: Harmony形式への変換とtrain/validationセットの分割
"""

import json
import logging
import random
from datetime import datetime
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_harmony_format(qa_data: List[Dict]) -> List[Dict]:
    """QAデータをHarmony形式に変換"""
    logger.info("=== Converting to Harmony Format ===")

    harmony_data = []

    for qa in qa_data:
        # Harmony形式: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        harmony_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": qa['question']
                },
                {
                    "from": "gpt",
                    "value": qa['answer']
                }
            ]
        }

        harmony_data.append(harmony_item)

    logger.info(f"Converted {len(harmony_data)} QA pairs to Harmony format")
    return harmony_data


def split_train_validation(data: List[Dict], train_ratio: float = 0.9, seed: int = 42) -> tuple:
    """データをtrain/validationセットに分割"""
    logger.info(f"\n=== Splitting Data (train: {train_ratio*100:.0f}%, validation: {(1-train_ratio)*100:.0f}%) ===")

    # シードを設定して再現性を確保
    random.seed(seed)

    # データをシャッフル
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 分割
    split_index = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_index]
    validation_data = shuffled_data[split_index:]

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(validation_data)}")

    return train_data, validation_data


def add_metadata(data: List[Dict], dataset_type: str) -> Dict:
    """メタデータを追加"""
    metadata = {
        "dataset_name": "takaichi-sanae-qa-dataset",
        "dataset_type": dataset_type,
        "version": "2.0",
        "creation_date": datetime.now().strftime('%Y-%m-%d'),
        "total_samples": len(data),
        "description": "高市早苗氏に関するQAデータセット - 第2次ファインチューニング用",
        "source": "Phase 2 Data Collection (Days 1-10)",
        "format": "Harmony (ChatML compatible)",
        "language": "Japanese",
        "categories": [
            "CAT-01: 基本プロフィール",
            "CAT-02: 経歴・学歴・職歴",
            "CAT-03: 政治歴・役職",
            "CAT-04: 政策・主張",
            "CAT-05: 大臣経験・実績",
            "CAT-06: 選挙・選挙区・得票",
            "CAT-07: 発言・スピーチ・著作",
            "CAT-08: 人間関係・交友",
            "CAT-09: 評価・批判・論争",
            "CAT-10: その他・雑学"
        ]
    }

    return {
        "metadata": metadata,
        "data": data
    }


def main():
    """メイン実行関数"""
    logger.info("=" * 80)
    logger.info("Phase 4: Format Conversion & Final Review")
    logger.info("=" * 80)

    # マージ済みデータを読み込み
    logger.info("\nLoading merged collection...")
    with open("data/processed/merged_collection.json", 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    logger.info(f"Loaded {len(qa_data)} QA pairs")

    # Harmony形式に変換
    harmony_data = convert_to_harmony_format(qa_data)

    # train/validationセットに分割
    train_data, validation_data = split_train_validation(harmony_data, train_ratio=0.9, seed=42)

    # メタデータを追加
    train_with_metadata = add_metadata(train_data, "train")
    validation_with_metadata = add_metadata(validation_data, "validation")

    # 保存
    logger.info("\n=== Saving Files ===")

    # Trainセット保存
    train_file = "data/processed/train_dataset.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_with_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Train dataset saved: {train_file}")

    # Validationセット保存
    validation_file = "data/processed/validation_dataset.json"
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_with_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Validation dataset saved: {validation_file}")

    # Harmony形式の全データも保存（メタデータなし、純粋なHarmony形式）
    full_harmony_file = "data/processed/full_harmony_dataset.json"
    with open(full_harmony_file, 'w', encoding='utf-8') as f:
        json.dump(harmony_data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Full Harmony dataset saved: {full_harmony_file}")

    # サマリーレポート作成
    summary_report = f"""
================================================================================
Phase 4: Format Conversion Complete
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary

### Total Samples: {len(qa_data)}
- Train: {len(train_data)} samples (90%)
- Validation: {len(validation_data)} samples (10%)

### Format
- Original: Custom QA format with metadata
- Converted: Harmony format (ChatML compatible)
- Structure: {{"conversations": [{{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}

### Files Created
1. {train_file}
   - Train dataset with metadata
   - {len(train_data)} samples

2. {validation_file}
   - Validation dataset with metadata
   - {len(validation_data)} samples

3. {full_harmony_file}
   - Full dataset in pure Harmony format
   - {len(harmony_data)} samples

### Dataset Version
- Version: 2.0
- Name: takaichi-sanae-qa-dataset
- Language: Japanese
- Purpose: 第2次ファインチューニング用

### Next Steps
1. ✅ データ収集完了 (Phase 2)
2. ✅ 包括的検証完了 (Phase 3)
3. ✅ フォーマット変換完了 (Phase 4)
4. ➡️ 第2次ファインチューニング開始準備

### Recommended Training Parameters
- Base Model: GPT-OSS 20B
- Training Samples: {len(train_data)}
- Validation Samples: {len(validation_data)}
- Epochs: 25-30 (previous: 20, insufficient)
- Batch Size: 2-4
- Learning Rate: 5e-5 to 1e-4
- LoRA r: 16
- LoRA alpha: 32

### Expected Improvements
1. データセット規模: 101 → {len(qa_data)} samples (約3倍)
2. カテゴリー網羅性: 6カテゴリー → 10カテゴリー
3. 信頼性: A+B = 100%
4. エポック数増加で知識定着向上

================================================================================
"""

    # サマリーレポート保存
    summary_file = "data/metadata/phase4_conversion_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)

    print(summary_report)
    logger.info(f"\n✅ Summary report saved: {summary_file}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Phase 4 Complete - Ready for 2nd Fine-tuning!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Format conversion successful!")
        print("✅ Dataset is ready for fine-tuning!")
    else:
        print("\n❌ Format conversion failed")
