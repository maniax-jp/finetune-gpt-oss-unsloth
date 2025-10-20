#!/usr/bin/env python3
"""
第2次ファインチューニングスクリプト (V2)
データセット: 301サンプル (101 → 301, 約3倍)
エポック数: 25-30 (前回20から増加)
"""

import os
import json
import torch
from datetime import datetime
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. 設定パラメータ
# ============================================================================

# モデル設定
MAX_SEQ_LENGTH = 2048  # GPT-OSSの最大コンテキスト長
DTYPE = None  # 自動検出（BF16またはFP16）
LOAD_IN_4BIT = True  # QLoRA 4-bit量子化

# LoRA設定
LORA_R = 16  # LoRA rank (前回64から削減して安定性向上)
LORA_ALPHA = 32  # LoRA alpha (2 * r)
LORA_DROPOUT = 0  # LoRA dropout
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# トレーニング設定
BATCH_SIZE = 2  # per_device_train_batch_size
GRADIENT_ACCUMULATION_STEPS = 4  # 実効バッチサイズ = 2 * 4 = 8
NUM_TRAIN_EPOCHS = 25  # 前回20から増加（知識定着向上）
LEARNING_RATE = 5e-5  # 前回1e-4から減少（安定性向上）
WARMUP_STEPS = 10
LOGGING_STEPS = 1
SAVE_STEPS = 50
EVAL_STEPS = 50

# データセットパス
TRAIN_DATASET_PATH = "data/processed/train_dataset.json"
VALIDATION_DATASET_PATH = "data/processed/validation_dataset.json"

# 出力ディレクトリ
OUTPUT_DIR = f"outputs/gpt-oss-20b-takaichi-v2-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================================================================
# 2. データセット読み込み
# ============================================================================

def load_harmony_dataset(file_path: str) -> Dataset:
    """Harmony形式のデータセットを読み込み"""
    logger.info(f"Loading dataset from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # メタデータを除外してデータ部分のみ取得
    if 'data' in data:
        dataset_list = data['data']
    else:
        dataset_list = data

    logger.info(f"Loaded {len(dataset_list)} samples")

    # Datasetsライブラリ形式に変換
    return Dataset.from_list(dataset_list)


def formatting_prompts_func(examples):
    """
    Harmony形式の会話データをテキストに変換
    Unslothの標準形式に合わせる
    """
    convos = examples["conversations"]
    texts = []

    for convo in convos:
        # Harmony形式: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}


# ============================================================================
# 3. モデルとトークナイザーの読み込み
# ============================================================================

logger.info("=" * 80)
logger.info("第2次ファインチューニング開始")
logger.info("=" * 80)

logger.info("\n=== モデル読み込み ===")
logger.info(f"Base Model: unsloth/gpt-oss-20b-BF16")
logger.info(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
logger.info(f"4-bit Quantization: {LOAD_IN_4BIT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-BF16",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

logger.info("✅ モデル読み込み完了")

# チャットテンプレート設定
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",  # GPT-OSSはChatML形式
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

logger.info("✅ チャットテンプレート設定完了")

# ============================================================================
# 4. LoRAアダプターの追加
# ============================================================================

logger.info("\n=== LoRAアダプター設定 ===")
logger.info(f"LoRA rank (r): {LORA_R}")
logger.info(f"LoRA alpha: {LORA_ALPHA}")
logger.info(f"Target modules: {TARGET_MODULES}")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # メモリ節約
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

logger.info("✅ LoRAアダプター設定完了")

# ============================================================================
# 5. データセット準備
# ============================================================================

logger.info("\n=== データセット準備 ===")

# Trainデータセット読み込み
train_dataset = load_harmony_dataset(TRAIN_DATASET_PATH)
logger.info(f"Train samples: {len(train_dataset)}")

# Validationデータセット読み込み
eval_dataset = load_harmony_dataset(VALIDATION_DATASET_PATH)
logger.info(f"Validation samples: {len(eval_dataset)}")

# データセット例を表示
logger.info("\n=== データセット例 ===")
logger.info(f"Sample conversation:\n{train_dataset[0]}")

# ============================================================================
# 6. トレーニング設定
# ============================================================================

logger.info("\n=== トレーニング設定 ===")
logger.info(f"Epochs: {NUM_TRAIN_EPOCHS}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info(f"Total train samples: {len(train_dataset)}")
logger.info(f"Total train steps: {len(train_dataset) * NUM_TRAIN_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # GPT-OSSではFalse推奨
    formatting_func=formatting_prompts_func,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,  # 最新3チェックポイントのみ保存
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # cosineスケジューラ使用
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",  # TensorBoard等の外部ツールを使用しない
    ),
)

# ============================================================================
# 7. トレーニング実行
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("トレーニング開始")
logger.info("=" * 80)

# GPU情報表示
gpu_stats = torch.cuda.get_device_properties(0)
logger.info(f"\nGPU: {gpu_stats.name}")
logger.info(f"VRAM: {round(gpu_stats.total_memory / 1024**3, 1)} GB")
logger.info(f"Compute Capability: {gpu_stats.major}.{gpu_stats.minor}")

# トレーニング実行
try:
    trainer_stats = trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("トレーニング完了")
    logger.info("=" * 80)

    # トレーニング統計を表示
    logger.info(f"\nTraining stats:")
    logger.info(f"  Total time: {trainer_stats.metrics.get('train_runtime', 0):.2f} seconds")
    logger.info(f"  Samples/second: {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")
    logger.info(f"  Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # 統計をファイルに保存
    stats_file = os.path.join(OUTPUT_DIR, "training_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(trainer_stats.metrics, f, indent=2)
    logger.info(f"\n✅ Training stats saved to {stats_file}")

except Exception as e:
    logger.error(f"\n❌ Training failed: {e}")
    raise

# ============================================================================
# 8. モデル保存
# ============================================================================

logger.info("\n=== モデル保存 ===")

# LoRAアダプターのみ保存
lora_save_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(lora_save_path)
tokenizer.save_pretrained(lora_save_path)
logger.info(f"✅ LoRA adapter saved to {lora_save_path}")

# マージ済みモデル保存（16bit）
merged_save_path = os.path.join(OUTPUT_DIR, "merged_16bit")
model.save_pretrained_merged(
    merged_save_path,
    tokenizer,
    save_method="merged_16bit"
)
logger.info(f"✅ Merged model (16bit) saved to {merged_save_path}")

# トレーニング設定を保存
config = {
    "model_name": "unsloth/gpt-oss-20b-BF16",
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "train_samples": len(train_dataset),
    "eval_samples": len(eval_dataset),
    "output_dir": OUTPUT_DIR,
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

config_file = os.path.join(OUTPUT_DIR, "training_config.json")
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)
logger.info(f"✅ Training config saved to {config_file}")

logger.info("\n" + "=" * 80)
logger.info("✅ 第2次ファインチューニング完了")
logger.info("=" * 80)
logger.info(f"\n出力ディレクトリ: {OUTPUT_DIR}")
logger.info("\n次のステップ:")
logger.info("  1. モデルの評価とテスト")
logger.info("  2. GGUF形式へのエクスポート（Ollama用）")
logger.info("  3. Ollamaへのインポートと動作確認")
