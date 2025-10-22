#!/usr/bin/env python3
"""
第7次ファインチューニングスクリプト - 完全推論データセット版

Phase 10-2-B: 全データにanalysisチャンネル付与

改善点:
1. 完全推論データセットの使用
   - 推論データ（手動作成）: 50サンプル（9.9%）
   - 非推論データ（自動付与）: 455サンプル（90.1%）
   - analysisチャンネル付与率: 100%

2. 第6次の問題点を解決
   - 第6次: analysisチャンネル付与率 9.9% → 固有名詞認識率 0%
   - 第7次: analysisチャンネル付与率 100% → 固有名詞認識率 95%以上を目標

3. 固有名詞認識データの特徴
   - 全データのanalysis channelに「高市早苗 (Sanae Takaichi)」を明示
   - 重要な固有名詞の日英対応を全サンプルに含む
   - 公式ノートブックのHarmony形式に準拠

4. ハイパーパラメータ
   - 第6次のハイパーパラメータを継承
   - 公式ノートブック準拠設定
   - MoE過学習対策
   - train_on_responses_only適用

期待される効果:
- 固有名詞認識率: 0% → 95%以上
- 「高市早苗」→ "Sanae Takaichi" 認識: 100%
- 基本的事実の正答率: 90%以上

参考:
- docs/sixth-generation-analysis.md
- docs/proper_noun_recognition_issue.md
- docs/official_notebook_analysis.md
"""

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
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
        logging.FileHandler('logs/training_seventh_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定パラメータ（公式ノートブック準拠）
# ============================================================================

# モデル設定
MODEL_NAME = "openai/gpt-oss-20b"
MAX_SEQ_LENGTH = 1024  # 公式: 1024（我々は2048を使用していた）
LOAD_IN_4BIT = True

# LoRA設定（公式準拠）
LORA_RANK = 8  # 公式: 8（我々は16を使用していた）
LORA_ALPHA = 16  # 公式: 16
LORA_DROPOUT = 0.1  # 公式: 0だが、MoE過学習対策で0.1に設定
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# トレーニング設定（公式準拠 + MoE最適化）
BATCH_SIZE = 1  # 公式: 1（我々は4を使用していた）
GRADIENT_ACCUMULATION_STEPS = 4  # 実効バッチサイズ=4
MAX_STEPS = 60  # 公式: 30-60（エポック制御ではなくステップ制御）
LEARNING_RATE = 2e-4  # 公式: 2e-4（我々は5e-5を使用していた）
WARMUP_STEPS = 5  # 公式: 5
WEIGHT_DECAY = 0.1  # MoE過学習対策で強化（公式: 0.01）
LOGGING_STEPS = 1
SAVE_STEPS = 20

# データセットパス（第7次: 完全推論データセット）
FULL_REASONING_DATASET_PATH = "dataset/takaichi_sanae_full_reasoning_v7.jsonl"

# 出力ディレクトリ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/gpt-oss-20b-takaichi-v7-full-reasoning-{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# ============================================================================
# データセット準備関数
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
logger.info("第6次ファインチューニング開始 - 固有名詞認識対応版")
logger.info("=" * 80)
logger.info(f"\n【Phase 10-2-A: 固有名詞認識問題の解決】")
logger.info(f"\n【混合データセット】")
logger.info(f"  推論データ: 50サンプル (9.9%)")
logger.info(f"    - 固有名詞の日英対応を明示")
logger.info(f"    - Analysis channel使用")
logger.info(f"    - 「高市早苗 = Sanae Takaichi」を常に明記")
logger.info(f"  非推論データ: 455サンプル (90.1%)")
logger.info(f"  総サンプル数: 505")
logger.info(f"\n【第5次のハイパーパラメータを継承】")
logger.info(f"  max_seq_length: 1024（公式準拠）")
logger.info(f"  LoRA rank: 8、batch_size: 1、learning_rate: 2e-4")
logger.info(f"  LORA_DROPOUT: 0.1、WEIGHT_DECAY: 0.1（MoE対策）")
logger.info(f"  max_steps: 60（過学習防止）")
logger.info(f"\n【期待される効果】")
logger.info(f"  固有名詞認識率: 0% → 95%以上")
logger.info(f"  「高市早苗」→ \"Sanae Takaichi\" 認識: 100%")
logger.info(f"  基本的事実の正答率: 90%以上\n")

# データセットは既にマージ済み（JSONL形式）
sample_count = 505  # 50推論 + 455非推論

# ============================================================================
# 1. モデルとトークナイザーの読み込み
# ============================================================================

logger.info("\n" + "=" * 60)
logger.info("モデル読み込み")
logger.info("=" * 60)
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Max sequence length: {MAX_SEQ_LENGTH} (公式準拠)")
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
logger.info(f"LoRA rank (r): {LORA_RANK} (公式準拠)")
logger.info(f"LoRA alpha: {LORA_ALPHA}")
logger.info(f"LoRA dropout: {LORA_DROPOUT} (MoE過学習対策)")
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
logger.info(f"Dataset path: {FULL_REASONING_DATASET_PATH}")
logger.info(f"  推論データ（手動作成）: 50サンプル（9.9%）")
logger.info(f"  非推論データ（自動付与）: 455サンプル（90.1%）")
logger.info(f"  analysisチャンネル付与率: 100%")

dataset = load_dataset("json", data_files=FULL_REASONING_DATASET_PATH, split="train")
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
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,  # エポック制御ではなくステップ制御
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

# Unsloth最適化: 応答部分のみ学習
logger.info("\n🔧 Unsloth最適化機能を適用...")
logger.info("  train_on_responses_only: 応答部分のみ学習（質問部分はスキップ）")

gpt_oss_kwargs = {
    "instruction_part": "<|start|>user<|message|>",
    "response_part": "<|start|>assistant<|message|>",
}

trainer = train_on_responses_only(
    trainer,
    **gpt_oss_kwargs
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
estimated_time_minutes = (MAX_STEPS * 8) // 60  # 1ステップ約8秒想定

logger.info(f"\n予想トレーニング:")
logger.info(f"  最大ステップ数: {MAX_STEPS}（公式準拠）")
logger.info(f"  推定時間: 約{estimated_time_minutes}分")
logger.info(f"  注意: 第4次は約285ステップだったのに対し、今回は60ステップで早期終了")
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
    logger.info(f"  比較: 第4次最終Loss = 1.3954")

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
logger.info("🎉 第6次ファインチューニング完了 - Phase 10-2-A")
logger.info("=" * 80)
logger.info(f"出力ディレクトリ: {OUTPUT_DIR}")
logger.info(f"最終モデル: {final_output_dir}")
logger.info(f"データセット: {sample_count}サンプル (推論50 + 非推論455)")
logger.info(f"推論データ比率: 9.9%")
logger.info(f"最大ステップ: {MAX_STEPS}")
logger.info("\n【Phase 10-2-A の成果】")
logger.info("  ✅ 固有名詞認識データ50サンプル作成")
logger.info("  ✅ 推論データと非推論データのマージ")
logger.info("  ✅ Analysis channelでの固有名詞日英対応")
logger.info("  ✅ 「高市早苗 = Sanae Takaichi」の明示")
logger.info("\n【第5次からの継承】")
logger.info("  ✅ 公式ノートブック準拠のハイパーパラメータ")
logger.info("  ✅ MoE過学習対策（Dropout 0.1、Weight Decay 0.1）")
logger.info("  ✅ train_on_responses_only適用")
logger.info("\n次のステップ:")
logger.info("  1. 固有名詞認識率の検証（目標: 95%以上）")
logger.info("  2. 「高市早苗」→ \"Sanae Takaichi\" 認識の確認")
logger.info("  3. 第5次との比較評価")
logger.info("  4. 基本的事実の正答率確認（目標: 90%以上）")
logger.info("=" * 80)
