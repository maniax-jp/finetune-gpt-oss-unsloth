#!/usr/bin/env python3
"""
GGUF形式エクスポートスクリプト
ファインチューニング済みモデルをOllama用GGUF形式に変換

注意: GPT-OSSモデルは現在llama.cppでサポートされていないため、
     FP16形式でエクスポートし、Ollama用のModelfileを作成します。
"""

import os
import sys
from pathlib import Path
import logging
import torch
from unsloth import FastLanguageModel

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/export_to_gguf.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定パラメータ
# ============================================================================

# 最適化版モデルのパス（最新のトレーニング結果）
MODEL_DIR = "outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final"

# ベースモデル名
BASE_MODEL_NAME = "openai/gpt-oss-20b"

# エクスポート先ディレクトリ
EXPORT_DIR = "exported_models/takaichi-v2-optimized"

# 設定
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# ============================================================================
# メイン処理
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("モデルエクスポート開始")
    logger.info("=" * 80)
    logger.info(f"\nモデルディレクトリ: {MODEL_DIR}")
    logger.info(f"ベースモデル: {BASE_MODEL_NAME}")
    logger.info(f"エクスポート先: {EXPORT_DIR}\n")

    # エクスポートディレクトリ作成
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # モデルの存在確認
    if not os.path.exists(MODEL_DIR):
        logger.error(f"❌ モデルディレクトリが見つかりません: {MODEL_DIR}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("モデル読み込みとマージ")
    logger.info("=" * 60)

    try:
        # ベースモデルをFP16で読み込み（4-bitから変換）
        logger.info(f"ベースモデル読み込み: {BASE_MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,  # FP16でロード
        )

        # LoRAアダプターを読み込み
        logger.info(f"LoRAアダプター読み込み: {MODEL_DIR}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, MODEL_DIR)

        # LoRAをマージ
        logger.info("LoRAアダプターをベースモデルにマージ中...")
        model = model.merge_and_unload()

        logger.info("✅ モデルとLoRAアダプターの結合完了")

    except Exception as e:
        logger.error(f"❌ モデル結合エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # FP16形式で保存
    logger.info("\n" + "=" * 60)
    logger.info("FP16形式でモデル保存")
    logger.info("=" * 60)

    try:
        merged_model_path = f"{EXPORT_DIR}/merged_model"
        logger.info(f"保存先: {merged_model_path}")

        # モデルとトークナイザーを保存
        model.save_pretrained(merged_model_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_model_path)

        logger.info("✅ FP16形式でモデル保存完了")

        # ファイルサイズ確認
        total_size = 0
        for file_path in Path(merged_model_path).rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        logger.info(f"   総サイズ: {total_size / (1024**3):.2f} GB")

    except Exception as e:
        logger.error(f"❌ モデル保存エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Modelfile作成
    logger.info("\n" + "=" * 60)
    logger.info("Ollama Modelfile作成")
    logger.info("=" * 60)

    modelfile_content = f"""FROM ./merged_model

# Temperature (0.0 = 決定的, 1.0 = より創造的)
PARAMETER temperature 0.7

# Top-p sampling (0.0 = 最も確信度の高い選択, 1.0 = より多様)
PARAMETER top_p 0.9

# Repeat penalty (1.0 = ペナルティなし)
PARAMETER repeat_penalty 1.1

# システムプロンプト
SYSTEM \"\"\"
あなたは高市早苗氏に関する質問に答える専門アシスタントです。
正確で詳細な情報を提供し、不明な点は正直に伝えてください。
\"\"\"
"""

    modelfile_path = f"{EXPORT_DIR}/Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    logger.info(f"✅ Modelfile作成完了: {modelfile_path}")

    logger.info("\n" + "=" * 80)
    logger.info("🎉 モデルエクスポート完了")
    logger.info("=" * 80)
    logger.info(f"\n📁 エクスポートディレクトリ: {EXPORT_DIR}")

    logger.info("\n次のステップ:")
    logger.info("  1. Ollamaへのインポート:")
    logger.info(f"     cd {EXPORT_DIR}")
    logger.info(f"     ollama create takaichi-v2-optimized -f Modelfile")
    logger.info("  2. 動作確認:")
    logger.info("     ollama run takaichi-v2-optimized \"高市早苗氏について教えてください\"")


if __name__ == "__main__":
    main()
