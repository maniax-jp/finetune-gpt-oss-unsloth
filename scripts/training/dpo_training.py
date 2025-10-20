#!/usr/bin/env python3
"""
Week 5 タスク2: DPOトレーニングスクリプト

目的:
- 第2次ファインチューニング済みモデルに対して、
  DPO (Direct Preference Optimization) を実施し、
  事実の正確性を向上させる

参考: development-plan.md Phase 9
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import PeftModel
import torch
from datetime import datetime
import os

# ============================================================================
# 設定
# ============================================================================

# ベースモデル（第2次ファインチューニング済み）
BASE_MODEL_NAME = "openai/gpt-oss-20b"
ADAPTER_PATH = "outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final"

# DPOデータセット
DPO_DATASET_PATH = "data/comparison/dpo_dataset_final.jsonl"

# 出力ディレクトリ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/gpt-oss-20b-takaichi-v3-dpo-{timestamp}"

# ハイパーパラメータ
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "num_train_epochs": 3,  # 過学習防止のため少なめ
    "learning_rate": 5e-6,  # SFTより低い学習率
    "max_steps": -1,  # エポック数で制御
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.0,
    "lr_scheduler_type": "linear",
    "seed": 42,
    "output_dir": OUTPUT_DIR,
    "save_strategy": "epoch",
    "save_total_limit": 3,
}

# DPO固有パラメータ
DPO_BETA = 0.1  # KLペナルティの強度（0.1-0.5が一般的）

# ============================================================================
# ヘルパー関数
# ============================================================================

def print_section(title: str):
    """セクションヘッダーを表示"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_config():
    """設定情報を表示"""
    print_section("Week 5 タスク2: DPOトレーニング設定")
    print(f"ベースモデル: {BASE_MODEL_NAME}")
    print(f"LoRAアダプター: {ADAPTER_PATH}")
    print(f"DPOデータセット: {DPO_DATASET_PATH}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print(f"\nトレーニング設定:")
    print(f"  - エポック数: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"  - 学習率: {TRAINING_CONFIG['learning_rate']}")
    print(f"  - バッチサイズ: {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  - Gradient Accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  - 実効バッチサイズ: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"\nDPO設定:")
    print(f"  - Beta (KLペナルティ): {DPO_BETA}")
    print("=" * 80)


# ============================================================================
# メイン処理
# ============================================================================

def main():
    print_config()

    # ステップ1: ベースモデル読み込み
    print_section("[1/6] ベースモデル読み込み")
    print("モデル読み込み中...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ ベースモデル読み込み完了")

    # ステップ2: LoRAアダプター読み込み
    print_section("[2/6] LoRAアダプター読み込み")
    print(f"アダプターパス: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ LoRAアダプター読み込み完了")

    # ステップ3: トレーニング可能に設定
    print_section("[3/6] DPOトレーニング用にモデルを設定")
    print("既存のLoRAアダプターをトレーニング可能に設定中...")
    # 既存のLoRAパラメータをトレーニング可能にする
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("✅ モデル設定完了")
    print(f"   トレーニング可能パラメータ: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   総パラメータ数: {total_params:,}")

    # ステップ4: データセット読み込み
    print_section("[4/6] DPOデータセット読み込み")
    print(f"データセットパス: {DPO_DATASET_PATH}")

    if not os.path.exists(DPO_DATASET_PATH):
        raise FileNotFoundError(f"DPOデータセットが見つかりません: {DPO_DATASET_PATH}")

    dataset = load_dataset("json", data_files=DPO_DATASET_PATH, split="train")
    print(f"✅ データセット読み込み完了")
    print(f"   総サンプル数: {len(dataset)}")

    # サンプル表示
    if len(dataset) > 0:
        print(f"\n📋 データセットサンプル:")
        sample = dataset[0]
        print(f"   Prompt: {sample['prompt'][:60]}...")
        print(f"   Chosen: {sample['chosen'][:60]}...")
        print(f"   Rejected: {sample['rejected'][:60]}...")

    # ステップ5: DPOトレーナー設定
    print_section("[5/6] DPOトレーナー設定")
    print("DPOトレーナーを初期化中...")

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOは参照モデルを自動作成
        args=DPOConfig(**TRAINING_CONFIG),
        beta=DPO_BETA,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=2048,
        max_prompt_length=1024,
    )
    print("✅ DPOトレーナー設定完了")

    # ステップ6: トレーニング実行
    print_section("[6/6] DPOトレーニング開始")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n予想トレーニング時間:")
    total_steps = len(dataset) * TRAINING_CONFIG['num_train_epochs'] // (
        TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']
    )
    print(f"  総ステップ数: {total_steps}")
    print(f"  推定時間: {total_steps * 2 // 60}分程度（1ステップ約2秒想定）")
    print("\nトレーニング開始...")
    print("=" * 80)

    try:
        trainer_stats = dpo_trainer.train()

        # トレーニング成功
        print("\n" + "=" * 80)
        print("✅ トレーニング完了")
        print("=" * 80)

        # 統計情報
        print("\n📊 トレーニング統計:")
        if hasattr(trainer_stats, 'training_loss'):
            print(f"  最終Loss: {trainer_stats.training_loss:.4f}")
        if hasattr(trainer_stats, 'global_step'):
            print(f"  総ステップ数: {trainer_stats.global_step}")
        if hasattr(trainer_stats, 'metrics'):
            print(f"  メトリクス: {trainer_stats.metrics}")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ トレーニング中にエラーが発生しました: {e}")
        print("=" * 80)
        raise

    # ステップ7: モデル保存
    print_section("モデル保存")
    final_output_dir = f"{OUTPUT_DIR}/final"
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"保存先: {final_output_dir}")
    print("モデルを保存中...")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"✅ モデル保存完了")

    # 完了サマリー
    print_section("🎉 Week 5 タスク2: DPOトレーニング完了")
    print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print(f"最終モデル: {final_output_dir}")
    print("\n次のステップ:")
    print("  1. モデル評価: scripts/evaluation/compare_models.py")
    print("  2. Week 5完了レポート作成")
    print("=" * 80)


if __name__ == "__main__":
    main()
