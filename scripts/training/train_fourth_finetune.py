#!/usr/bin/env python3
"""
第4次ファインチューニングスクリプト

改善点:
- データセット: 301サンプル → 455サンプル（1.51倍に拡大）
- エポック数: 25（第2次と同じ）
- 学習率: 5e-5（第2次と同じ、安定性重視）
- LoRA rank: 16（第2次と同じ）
- バッチサイズ: 4（GPU並列化）
- 新規データ追加による知識拡充
"""

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import os
from datetime import datetime
import logging
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_fourth_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定パラメータ
# ============================================================================

# モデル設定
MODEL_NAME = "openai/gpt-oss-20b"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA設定
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# トレーニング設定
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # 実効バッチサイズ=16
NUM_TRAIN_EPOCHS = 25
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 1
SAVE_STEPS = 50

# データセットパス
SOURCE_DATA_PATH = "data/processed/merged_collection.json"
OUTPUT_DATASET_PATH = "dataset/takaichi_sanae_qa_harmony_v4.jsonl"

# 出力ディレクトリ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/gpt-oss-20b-takaichi-v4-{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# ============================================================================
# データセット準備関数
# ============================================================================

def convert_to_harmony_format(data_path: str, output_path: str):
    """
    merged_collection.json を Harmony形式のJSONLに変換
    """
    logger.info(f"データセット変換中: {data_path} → {output_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"  総データ数: {len(data)}")

    harmony_data = []
    for item in data:
        question = item.get('question', '')
        answer = item.get('answer', '')

        if not question or not answer:
            continue

        # Harmony形式に変換
        harmony_item = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        harmony_data.append(harmony_item)

    # JSONL形式で保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in harmony_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"✅ 変換完了: {len(harmony_data)}サンプル保存")
    return len(harmony_data)


def formatting_prompts_func(examples, tokenizer):
    """
    Harmony形式の会話データをテキストに変換
    """
    if isinstance(examples["messages"], list) and len(examples["messages"]) > 0:
        if isinstance(examples["messages"][0], list):
            messages_list = examples["messages"]
        else:
            messages_list = [examples["messages"]]
    else:
        messages_list = [examples["messages"]]

    texts = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return texts


# ============================================================================
# メイン処理
# ============================================================================

logger.info("=" * 80)
logger.info("第4次ファインチューニング開始")
logger.info("=" * 80)
logger.info(f"\n【第2次からの改善点】")
logger.info(f"  データセット: 301 → 455 samples (1.51倍)")
logger.info(f"  新規データ: +154サンプル（agent-takaichi収集）")
logger.info(f"  カテゴリー: より詳細な情報を追加")
logger.info(f"  期待効果: 基本的事実の正確性向上\n")
logger.info(f"\n【トレーニング設定】")
logger.info(f"  バッチサイズ: {BATCH_SIZE}")
logger.info(f"  実効バッチサイズ: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"  エポック数: {NUM_TRAIN_EPOCHS}")
logger.info(f"  学習率: {LEARNING_RATE}")
logger.info(f"  LoRA rank: {LORA_RANK}\n")

# ============================================================================
# 0. データセット準備
# ============================================================================

logger.info("=" * 60)
logger.info("データセット準備")
logger.info("=" * 60)

sample_count = convert_to_harmony_format(SOURCE_DATA_PATH, OUTPUT_DATASET_PATH)

# ============================================================================
# 1. モデルとトークナイザーの読み込み
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("モデル読み込み")
logger.info("=" * 60)
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Max sequence length: {MAX_SEQ_LENGTH}")
logger.info(f"4-bit quantization: {LOAD_IN_4BIT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

logger.info("✅ モデル読み込み完了")

# ============================================================================
# 2. LoRAアダプターの追加
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("LoRAアダプター設定")
logger.info("=" * 60)
logger.info(f"LoRA rank (r): {LORA_RANK}")
logger.info(f"LoRA alpha: {LORA_ALPHA}")
logger.info(f"Target modules: {TARGET_MODULES}")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

logger.info("✅ LoRAアダプター設定完了")

# ============================================================================
# 3. データセット読み込み
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("データセット読み込み")
logger.info("=" * 60)
logger.info(f"Dataset path: {OUTPUT_DATASET_PATH}")

dataset = load_dataset("json", data_files=OUTPUT_DATASET_PATH, split="train")
logger.info(f"Dataset size: {len(dataset)}")
logger.info("✅ データセット読み込み完了")

# ============================================================================
# 4. トレーナー設定
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("トレーナー設定")
logger.info("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    formatting_func=lambda examples: formatting_prompts_func(examples, tokenizer),
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
    ),
)

logger.info("✅ トレーナー設定完了")

# ============================================================================
# 5. トレーニング実行
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("トレーニング開始")
logger.info("=" * 80)
logger.info(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 予想時間計算
total_steps = (sample_count * NUM_TRAIN_EPOCHS) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
estimated_time_minutes = (total_steps * 8) // 60  # 1ステップ約8秒想定

logger.info(f"\n予想トレーニング:")
logger.info(f"  総ステップ数: 約{total_steps}")
logger.info(f"  推定時間: 約{estimated_time_minutes}分")
logger.info("=" * 80 + "\n")

try:
    trainer_stats = trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("トレーニング完了")
    logger.info("=" * 80)
    logger.info(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 統計情報
    logger.info("\n📊 トレーニング統計:")
    logger.info(f"  最終Loss: {trainer_stats.training_loss:.4f}")
    logger.info(f"  総ステップ数: {trainer_stats.global_step}")

except Exception as e:
    logger.error(f"\n❌ トレーニング中にエラーが発生: {e}")
    raise

# ============================================================================
# 6. モデル保存
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("モデル保存")
logger.info("=" * 60)

final_output_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_output_dir, exist_ok=True)

logger.info(f"保存先: {final_output_dir}")
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

logger.info("✅ モデル保存完了")

# ============================================================================
# 7. 完了サマリー
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("🎉 第4次ファインチューニング完了")
logger.info("=" * 80)
logger.info(f"出力ディレクトリ: {OUTPUT_DIR}")
logger.info(f"最終モデル: {final_output_dir}")
logger.info(f"データセット: {sample_count}サンプル")
logger.info(f"エポック: {NUM_TRAIN_EPOCHS}")
logger.info("\n次のステップ:")
logger.info("  1. モデル評価: scripts/evaluation/collect_model_outputs_v4.py")
logger.info("  2. 第3次モデルとの比較評価")
logger.info("  3. DPOデータセットの追加拡充")
logger.info("=" * 80)
