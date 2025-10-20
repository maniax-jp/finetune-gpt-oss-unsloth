#!/usr/bin/env python3
"""
Week 5 タスク4: 第3次モデル（DPO適用後）の出力収集スクリプト

目的: DPOトレーニング後のモデルの応答を収集し、
     第2次モデルとの比較評価を行うためのデータを生成
"""

import os
import json
import torch
from unsloth import FastLanguageModel
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from peft import PeftModel

# ============================================================================
# 設定パラメータ
# ============================================================================

# モデルパス（第3次DPOトレーニング済みモデル）
MODEL_DIR = "outputs/gpt-oss-20b-takaichi-v3-dpo-20251020_155007/final"
BASE_MODEL_NAME = "openai/gpt-oss-20b"

# テスト質問セット（Week 4と同じ51質問）
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
    "高市早苗さんの初当選はいつですか？",
    "高市早苗さんは総裁選に出馬したことがありますか？",
    "高市早苗さんは何期目の議員ですか？",

    # カテゴリー3: 政策・主張
    "高市早苗さんの経済政策について教えてください",
    "高市早苗さんは消費税についてどう考えていますか？",
    "高市早苗さんの安全保障政策は？",
    "高市早苗さんのエネルギー政策について教えてください",
    "高市早苗さんは憲法改正についてどう考えていますか？",
    "高市早苗さんの外交方針は？",
    "高市早苗さんは移民政策についてどう考えていますか？",

    # カテゴリー4: 政治的立場
    "高市早苗さんの政治的立場は？",
    "高市早苗さんは保守派ですか？",
    "高市早苗さんは靖国神社参拝についてどう考えていますか？",

    # カテゴリー5: 主要政策分野
    "高市早苗さんのデジタル政策について教えてください",
    "高市早苗さんは地方創生についてどう考えていますか？",
    "高市早苗さんの教育政策について教えてください",
    "高市早苗さんは少子化対策についてどう考えていますか？",

    # カテゴリー6: 経済安全保障
    "高市早苗さんの経済安全保障政策について教えてください",
    "高市早苗さんはサプライチェーンについてどう考えていますか？",
    "高市早苗さんは半導体政策についてどう考えていますか？",

    # カテゴリー7: 比較質問
    "高市早苗さんと岸田文雄さんの政策の違いは？",
    "高市早苗さんと安倍晋三さんの関係は？",
    "高市早苗さんは安倍派に所属していますか？",

    # カテゴリー8: 実績・業績
    "高市早苗さんの総務大臣としての実績は？",
    "高市早苗さんが推進した政策は？",
    "高市早苗さんの著書はありますか？",

    # カテゴリー9: 支持基盤
    "高市早苗さんの支持層は？",
    "高市早苗さんは保守系団体と関係がありますか？",

    # カテゴリー10: その他
    "高市早苗さんの趣味は？",
    "高市早苗さんは結婚していますか？",
    "高市早苗さんの家族構成は？",
    "高市早苗さんの座右の銘は？",

    # カテゴリー11: 最近の動向
    "高市早苗さんは2024年の総裁選に出馬しましたか？",
    "高市早苗さんの現在の役職は？",

    # カテゴリー12: 追加の基本情報
    "高市早苗さんの年齢は？",
    "高市早苗さんの身長は？",
    "高市早苗さんの血液型は？",

    # カテゴリー13: 政策詳細
    "高市早苗さんの金融政策について教えてください",
    "高市早苗さんは原発についてどう考えていますか？",
    "高市早苗さんの防衛政策について教えてください",
    "高市早苗さんは女性活躍についてどう考えていますか？",
    "高市早苗さんの社会保障政策について教えてください",
]

# 出力ファイルパス
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"data/comparison/model_outputs_v3_{timestamp}.json"

# 生成設定
GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# ============================================================================
# メイン処理
# ============================================================================

def load_model():
    """モデルとトークナイザーの読み込み"""
    print("=" * 80)
    print("Week 5 タスク4: 第3次モデル（DPO適用後）出力収集")
    print("=" * 80)
    print(f"\nモデルディレクトリ: {MODEL_DIR}")
    print("モデル読み込み中...")

    # ベースモデル読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # DPO済みLoRAアダプター読み込み
    model = PeftModel.from_pretrained(model, MODEL_DIR)

    # 推論モード
    FastLanguageModel.for_inference(model)

    print("✅ モデル読み込み完了")
    return model, tokenizer


def generate_response(model, tokenizer, question: str) -> str:
    """質問に対する応答を生成"""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 入力部分をスキップして応答のみ抽出
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


def main():
    """メイン処理"""
    # モデル読み込み
    model, tokenizer = load_model()

    # 出力ディレクトリ作成
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # 各質問に対する応答を収集
    print(f"\n質問数: {len(TEST_QUESTIONS)}")
    print("応答収集開始...\n")

    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")

        try:
            response = generate_response(model, tokenizer, question)
            results.append({
                "question_id": i,
                "question": question,
                "model_response": response,
                "timestamp": datetime.now().isoformat()
            })
            print(f"  応答: {response[:80]}...")

        except Exception as e:
            print(f"  ❌ エラー: {e}")
            results.append({
                "question_id": i,
                "question": question,
                "model_response": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        print()

    # 結果を保存
    print("=" * 80)
    print("結果を保存中...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存完了: {OUTPUT_FILE}")
    print(f"   総質問数: {len(results)}")
    print(f"   成功: {sum(1 for r in results if 'error' not in r)}")
    print(f"   失敗: {sum(1 for r in results if 'error' in r)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
