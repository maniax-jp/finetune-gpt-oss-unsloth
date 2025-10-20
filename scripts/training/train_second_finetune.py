#!/usr/bin/env python3
"""
第2次ファインチューニングスクリプト
- データセット: 301サンプル（第1次の101から3倍に拡大）
- エポック数: 25（第1次の20から増加）
- 学習率: 5e-5（第1次の1e-4から減少、安定性向上）
- LoRA rank: 16（第1次の64から削減、メモリ効率向上）
"""

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import os
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_second_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定パラメータ
# ============================================================================

# モデル設定
MODEL_NAME = "openai/gpt-oss-20b"  # 正しいモデル名
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA設定（第1次から改善）
LORA_RANK = 16  # 64→16（メモリ効率・安定性向上）
LORA_ALPHA = 16  # rank と同じ値
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# トレーニング設定（第1次から改善）
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # 実効バッチサイズ=8
NUM_TRAIN_EPOCHS = 25  # 20→25（知識定着向上）
LEARNING_RATE = 5e-5  # 1e-4→5e-5（安定性向上）
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 1
SAVE_STEPS = 50

# データセットパス
DATASET_PATH = "dataset/takaichi_sanae_qa_harmony_v2.jsonl"

# 出力ディレクトリ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/gpt-oss-20b-takaichi-v2-{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================================================================
# 関数定義
# ============================================================================

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
logger.info("第2次ファインチューニング開始")
logger.info("=" * 80)
logger.info(f"\n【第1次からの改善点】")
logger.info(f"  データセット: 101 → 301 samples (3倍)")
logger.info(f"  エポック数: 20 → {NUM_TRAIN_EPOCHS}")
logger.info(f"  学習率: 1e-4 → {LEARNING_RATE}")
logger.info(f"  LoRA rank: 64 → {LORA_RANK}")
logger.info(f"  カテゴリー: 6 → 10 (全カテゴリー網羅)")
logger.info(f"  信頼性: A+B = 100%\n")

# ============================================================================
# 1. モデルとトークナイザーの読み込み
# ============================================================================

logger.info("=" * 60)
logger.info("モデル読み込み")
logger.info("=" * 60)
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Max sequence length: {MAX_SEQ_LENGTH}")
logger.info(f"4-bit quantization: {LOAD_IN_4BIT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
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

# VRAM使用状況を表示
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    logger.info(f"   VRAM allocated: {memory_allocated:.2f} GB")

# ============================================================================
# 3. データセット読み込み
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("データセット読み込み")
logger.info("=" * 60)
logger.info(f"Dataset path: {DATASET_PATH}")

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

logger.info(f"✅ Dataset loaded: {len(dataset)} conversations")

# ============================================================================
# 4. トレーニング設定
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("トレーニング設定")
logger.info("=" * 60)
logger.info(f"Epochs: {NUM_TRAIN_EPOCHS}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info(f"Warmup ratio: {WARMUP_RATIO}")
logger.info(f"Weight decay: {WEIGHT_DECAY}")
logger.info(f"Total samples: {len(dataset)}")
logger.info(f"Total steps: {len(dataset) * NUM_TRAIN_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
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
    lr_scheduler_type="cosine",
    seed=3407,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,  # 最新3チェックポイント保存
    report_to="none",
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=data_collator,
    formatting_func=lambda examples: formatting_prompts_func(examples, tokenizer),
    args=training_args,
)

# ============================================================================
# 5. トレーニング実行
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("🚀 トレーニング開始")
logger.info("=" * 80)

# Enable native 2x faster training
FastLanguageModel.for_training(model)

# GPU情報表示
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    logger.info(f"\n📊 GPU Stats:")
    logger.info(f"   Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"   Total VRAM: {gpu_stats.total_memory / 1024**3:.2f} GB")
    logger.info(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    logger.info(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    logger.info(f"   Compute: {gpu_stats.major}.{gpu_stats.minor}\n")

# トレーニング実行
try:
    trainer_stats = trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("✅ トレーニング完了")
    logger.info("=" * 80)

    # トレーニング統計
    logger.info(f"\n📈 Training Statistics:")
    logger.info(f"   Train loss: {trainer_stats.training_loss:.4f}")
    logger.info(f"   Train runtime: {trainer_stats.metrics['train_runtime']:.2f}s")
    logger.info(f"   Train samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    logger.info(f"   Train steps/sec: {trainer_stats.metrics['train_steps_per_second']:.2f}")

    # GPU統計（トレーニング後）
    if torch.cuda.is_available():
        logger.info(f"\n📊 GPU Stats (After Training):")
        logger.info(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"   Peak allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

except Exception as e:
    logger.error(f"\n❌ Training failed: {e}")
    raise

# ============================================================================
# 6. モデル保存
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("💾 モデル保存")
logger.info("=" * 60)

# LoRAアダプターのみ保存
final_dir = f"{OUTPUT_DIR}/final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
logger.info(f"✅ Model saved to {final_dir}")

logger.info("\n" + "=" * 80)
logger.info("🎉 第2次ファインチューニング完了")
logger.info("=" * 80)
logger.info(f"\n📁 Output directory: {OUTPUT_DIR}")
logger.info("\n次のステップ:")
logger.info("  1. モデルの評価とテスト")
logger.info("  2. GGUF形式へのエクスポート（Ollama用）")
logger.info("  3. Ollamaへのインポートと動作確認")
