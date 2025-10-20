#!/usr/bin/env python3
"""
Week 4: 比較データセット作成 - モデル出力収集スクリプト

目的: ファインチューニング済みモデルの応答を収集し、
     良い回答/悪い回答のペアを作成するためのデータを生成

開発計画 Phase 8.4の実装
"""

import os
import json
import torch
from unsloth import FastLanguageModel
from datetime import datetime
import logging
from typing import List, Dict
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_model_outputs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定パラメータ
# ============================================================================

# モデルパス（第2次ファインチューニング済みモデル）
MODEL_DIR = "outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final"
BASE_MODEL_NAME = "openai/gpt-oss-20b"

# テスト質問セット
TEST_QUESTIONS = [
    # カテゴリー1: 基本情報
    "高市早苗さんは何党ですか？",
    "高市早苗さんの生年月日は？",
    "高市早苗さんの出身地はどこですか？",
    "高市早苗さんの学歴について教えてください",
    "高市早苗さんの経歴を教えてください",

    # カテゴリー2: 政治活動
    "高市早苗さんはどの選挙区から立候補していますか？",
    "高市早苗さんが務めた大臣のポストは？",
    "高市早苗さんの政治的な立場は？",
    "高市早苗さんの主な政策は何ですか？",
    "高市早苗さんは何期目の議員ですか？",

    # カテゴリー3: 政策・主張
    "高市早苗さんの経済政策について教えてください",
    "高市早苗さんの安全保障政策は？",
    "高市早苗さんのエネルギー政策について",
    "高市早苗さんは憲法改正についてどう考えていますか？",
    "高市早苗さんの外交政策は？",

    # カテゴリー4: 実績
    "高市早苗さんが総務大臣時代に行ったことは？",
    "高市早苗さんの主な業績は何ですか？",
    "高市早苗さんが推進した法案は？",
    "高市早苗さんの国会での活動について",
    "高市早苗さんが立ち上げた議員連盟は？",

    # カテゴリー5: 発言・著作
    "高市早苗さんの著書は何がありますか？",
    "高市早苗さんの有名な発言は？",
    "高市早苗さんはメディアでどのように発言していますか？",
    "高市早苗さんの最近の発言について",

    # カテゴリー6: 人間関係
    "高市早苗さんと安倍晋三氏の関係は？",
    "高市早苗さんはどの派閥に所属していますか？",
    "高市早苗さんと親しい政治家は？",

    # カテゴリー7: 選挙
    "高市早苗さんの初当選はいつですか？",
    "高市早苗さんの選挙での得票数は？",
    "高市早苗さんは総裁選に出馬したことがありますか？",

    # カテゴリー8: その他
    "高市早苗さんの趣味は何ですか？",
    "高市早苗さんの家族構成は？",
    "高市早苗さんの座右の銘は？",
    "高市早苗さんについて簡単に紹介してください",
    "高市早苗さんの特徴は何ですか？",

    # カテゴリー9: 複雑な質問
    "高市早苗さんの政策は日本の未来にどう影響しますか？",
    "高市早苗さんが首相になったらどんな政策を実施すると思いますか？",
    "高市早苗さんの強みと弱みは何ですか？",
    "高市早苗さんはなぜ人気があるのですか？",

    # カテゴリー10: 比較質問
    "高市早苗さんと岸田文雄氏の政策の違いは？",
    "高市早苗さんと他の自民党議員の違いは何ですか？",
    "高市早苗さんと小泉進次郎氏の考え方の違いは？",

    # 追加の質問（計50問を目指す）
    "高市早苗さんは女性議員としてどのような活動をしていますか？",
    "高市早苗さんの防衛政策について教えてください",
    "高市早苗さんはデジタル政策についてどう考えていますか？",
    "高市早苗さんの地元での評判は？",
    "高市早苗さんの少子化対策について",
    "高市早苗さんは教育政策についてどう考えていますか？",
    "高市早苗さんの環境政策は？",
    "高市早苗さんの社会保障政策について",
    "高市早苗さんは農業政策についてどう考えていますか？",
]

# 生成パラメータ
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# 出力ファイル
OUTPUT_DIR = "data/comparison"
OUTPUT_FILE = f"{OUTPUT_DIR}/model_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ============================================================================
# メイン処理
# ============================================================================

def load_model():
    """モデルとトークナイザーを読み込み"""
    logger.info("=" * 60)
    logger.info("モデル読み込み")
    logger.info("=" * 60)

    try:
        # ベースモデル読み込み
        logger.info(f"ベースモデル: {BASE_MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # LoRAアダプター読み込み
        logger.info(f"LoRAアダプター: {MODEL_DIR}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, MODEL_DIR)

        logger.info("✅ モデル読み込み完了")

        # 推論モードに設定
        FastLanguageModel.for_inference(model)

        return model, tokenizer

    except Exception as e:
        logger.error(f"❌ モデル読み込みエラー: {e}")
        raise


def generate_response(model, tokenizer, question: str) -> str:
    """質問に対する応答を生成"""

    # Harmony形式のプロンプト
    messages = [
        {"role": "user", "content": question}
    ]

    # チャットテンプレート適用
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # トークン化
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # デコード
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # アシスタントの応答部分のみ抽出
    if "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1].strip()
    else:
        response = full_output[len(prompt):].strip()

    return response


def collect_outputs(model, tokenizer, questions: List[str]) -> List[Dict]:
    """すべての質問に対する応答を収集"""

    logger.info("\n" + "=" * 60)
    logger.info(f"応答収集開始（{len(questions)}質問）")
    logger.info("=" * 60)

    results = []

    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] 質問: {question}")

        try:
            response = generate_response(model, tokenizer, question)
            logger.info(f"応答: {response[:100]}...")

            results.append({
                "question_id": i,
                "question": question,
                "model_response": response,
                "timestamp": datetime.now().isoformat(),
                "generation_config": GENERATION_CONFIG,
                # 人間評価用のフィールド（後で記入）
                "is_good_response": None,  # True/False/Null
                "human_rating": None,  # 1-5
                "issues": [],  # 問題点のリスト
                "better_response": None,  # より良い応答の例
            })

        except Exception as e:
            logger.error(f"❌ 生成エラー: {e}")
            results.append({
                "question_id": i,
                "question": question,
                "model_response": f"ERROR: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "is_good_response": False,
                "human_rating": 1,
                "issues": ["generation_error"],
                "better_response": None,
            })

    return results


def save_results(results: List[Dict]):
    """結果を保存"""

    logger.info("\n" + "=" * 60)
    logger.info("結果保存")
    logger.info("=" * 60)

    # ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 保存完了: {OUTPUT_FILE}")

    # 統計情報
    logger.info("\n📊 収集統計:")
    logger.info(f"   総質問数: {len(results)}")

    error_count = sum(1 for r in results if "ERROR" in r["model_response"])
    logger.info(f"   エラー数: {error_count}")
    logger.info(f"   成功数: {len(results) - error_count}")


def main():
    logger.info("=" * 80)
    logger.info("Week 4: モデル出力収集開始")
    logger.info("=" * 80)
    logger.info(f"\nモデル: {MODEL_DIR}")
    logger.info(f"質問数: {len(TEST_QUESTIONS)}")
    logger.info(f"出力先: {OUTPUT_FILE}\n")

    # モデル読み込み
    model, tokenizer = load_model()

    # 応答収集
    results = collect_outputs(model, tokenizer, TEST_QUESTIONS)

    # 結果保存
    save_results(results)

    logger.info("\n" + "=" * 80)
    logger.info("🎉 Week 4: モデル出力収集完了")
    logger.info("=" * 80)
    logger.info("\n次のステップ:")
    logger.info(f"  1. {OUTPUT_FILE} を開いて人間評価を実施")
    logger.info("  2. 各応答に以下を記入:")
    logger.info("     - is_good_response: True/False")
    logger.info("     - human_rating: 1-5 (1=最悪, 5=最高)")
    logger.info("     - issues: 問題点のリスト")
    logger.info("     - better_response: より良い応答の例")
    logger.info("  3. 評価完了後、create_comparison_dataset.pyで比較データセット作成")


if __name__ == "__main__":
    main()
