#!/usr/bin/env python3
"""
Week 5 タスク1: データセット拡充スクリプト（方法A + 方法B）

目的:
- 既存301サンプルからモデルの誤り応答を抽出（方法A）
- 温度パラメータを上げて多様な誤り応答を生成（方法B）
- 7ペア → 100ペア以上に拡充

戦略:
1. 既存301サンプルの質問に対してモデル応答を生成
2. 温度を1.2に上げて多様性を確保
3. 既知の正解と比較して誤りを自動検出
4. DPO形式（prompt, chosen, rejected）で保存
"""

import json
import torch
from unsloth import FastLanguageModel
from datetime import datetime
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

# ============================================================================
# 設定
# ============================================================================

# モデル設定
BASE_MODEL_NAME = "openai/gpt-oss-20b"
ADAPTER_PATH = "outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final"

# データセット
TRAINING_DATA_PATH = "data/processed/merged_collection.json"
OUTPUT_PATH = f"data/comparison/dpo_dataset_expanded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# 生成設定（方法B: 温度を上げて多様性確保）
GENERATION_CONFIG = {
    "temperature": 1.5,  # さらに高温で多様性確保（2回目の収集）
    "top_p": 0.95,
    "top_k": 50,
    "do_sample": True,
    "max_new_tokens": 200,
    "repetition_penalty": 1.1,
}

# 既知の事実データベース（Week 4で判明した誤りを優先）
CORRECT_FACTS = {
    "生年月日": "1961年3月7日",
    "出身地": "奈良県",
    "学歴": "神戸大学経営学部卒業",
    "選挙区": "奈良2区",
    "所属政党": "自由民主党",
    "大臣経験": ["総務大臣", "経済安全保障担当大臣", "内閣府特命担当大臣"],
    "初当選": "1993年",
    "総裁選出馬": ["2021年", "2024年"],
}

# 誤りを検出するキーワードパターン
ERROR_PATTERNS = {
    "生年月日": ["1960年", "1962年", "11月23日", "1959年"],
    "出身地": ["愛知県", "名古屋市", "東京都", "大阪府", "京都府"],
    "学歴": ["東京大学", "早稲田大学", "慶應義塾大学", "法学部", "経済学部"],
    "選挙区": ["千葉", "東京", "大阪", "愛知", "奈良1区", "奈良3区"],
    "大臣": ["経済産業大臣", "外務大臣", "財務大臣", "防衛大臣"],
    "初当選": ["2003年", "1995年", "2000年", "1990年"],
    "総裁選": ["2018年", "2012年", "2015年", "2020年"],
}

# ============================================================================
# ヘルパー関数
# ============================================================================

def load_model():
    """モデル読み込み"""
    print("モデル読み込み中...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # LoRAアダプター読み込み
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # 推論モード
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_response(model, tokenizer, question: str) -> str:
    """モデル応答生成（温度1.2で多様性確保）"""
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


def detect_error(question: str, model_response: str, correct_answer: str) -> Tuple[bool, str, str]:
    """
    誤りを検出

    Returns:
        (is_error, error_type, better_response)
    """
    q_lower = question.lower()
    r_lower = model_response.lower()

    # 空応答チェック
    if len(model_response.strip()) == 0:
        return True, "空応答", correct_answer

    # 短すぎる応答チェック（10文字未満）
    if len(model_response.strip()) < 10:
        return True, "短すぎる応答", correct_answer

    # 生年月日の誤り
    if "生年月日" in q_lower or "いつ生まれ" in q_lower or "何年生まれ" in q_lower:
        for wrong_date in ERROR_PATTERNS["生年月日"]:
            if wrong_date in r_lower:
                return True, "生年月日の誤り", correct_answer
        # 正解が含まれていない場合も誤り
        if "1961年3月7日" not in model_response and "1961年3月" not in model_response:
            return True, "生年月日の誤り", correct_answer

    # 出身地の誤り
    if "出身" in q_lower and ("地" in q_lower or "どこ" in q_lower):
        for wrong_place in ERROR_PATTERNS["出身地"]:
            if wrong_place in r_lower:
                return True, "出身地の誤り", correct_answer
        if "奈良" not in r_lower:
            return True, "出身地の誤り", correct_answer

    # 学歴の誤り
    if "学歴" in q_lower or "大学" in q_lower or "出身大学" in q_lower:
        for wrong_school in ERROR_PATTERNS["学歴"]:
            if wrong_school in r_lower:
                return True, "学歴の誤り", correct_answer
        if "神戸大学" not in r_lower:
            return True, "学歴の誤り", correct_answer

    # 選挙区の誤り
    if "選挙区" in q_lower:
        for wrong_district in ERROR_PATTERNS["選挙区"]:
            if wrong_district in r_lower:
                return True, "選挙区の誤り", correct_answer
        if "奈良2区" not in r_lower and "奈良県第2区" not in r_lower:
            return True, "選挙区の誤り", correct_answer

    # 大臣経験の誤り
    if "大臣" in q_lower:
        for wrong_minister in ERROR_PATTERNS["大臣"]:
            if wrong_minister in r_lower:
                return True, "大臣経験の誤り", correct_answer

    # 初当選の誤り
    if "初当選" in q_lower:
        for wrong_year in ERROR_PATTERNS["初当選"]:
            if wrong_year in r_lower:
                return True, "初当選の誤り", correct_answer
        if "1993年" not in r_lower and "1993" not in r_lower:
            return True, "初当選の誤り", correct_answer

    # 総裁選の誤り
    if "総裁選" in q_lower:
        for wrong_year in ERROR_PATTERNS["総裁選"]:
            if wrong_year in r_lower:
                return True, "総裁選の誤り", correct_answer

    return False, None, None


def create_dpo_pair(question: str, correct_answer: str, wrong_answer: str) -> Dict:
    """DPO形式のペアを作成"""
    return {
        "prompt": question,
        "chosen": correct_answer,
        "rejected": wrong_answer
    }


# ============================================================================
# メイン処理
# ============================================================================

def main():
    print("=" * 80)
    print("Week 5 タスク1: データセット拡充（方法A + 方法B）")
    print("=" * 80)
    print(f"入力: {TRAINING_DATA_PATH}")
    print(f"出力: {OUTPUT_PATH}")
    print(f"温度パラメータ: {GENERATION_CONFIG['temperature']}")
    print("=" * 80 + "\n")

    # ディレクトリ作成
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # モデル読み込み
    model, tokenizer = load_model()
    print("✅ モデル読み込み完了\n")

    # トレーニングデータ読み込み
    print("トレーニングデータ読み込み中...")
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    print(f"✅ {len(training_data)}サンプル読み込み完了\n")

    # DPOペア収集
    dpo_pairs = []
    error_stats = {}

    print("モデル応答生成中（温度=1.2で多様性確保）...\n")

    for item in tqdm(training_data, desc="処理中"):
        question = item.get("question", "")
        correct_answer = item.get("answer", "")

        if not question or not correct_answer:
            continue

        # モデル応答生成
        model_response = generate_response(model, tokenizer, question)

        # 誤り検出
        is_error, error_type, better_response = detect_error(
            question, model_response, correct_answer
        )

        if is_error:
            # DPOペア作成
            dpo_pair = create_dpo_pair(question, better_response, model_response)
            dpo_pairs.append(dpo_pair)

            # 統計更新
            if error_type not in error_stats:
                error_stats[error_type] = 0
            error_stats[error_type] += 1

    # 結果保存
    print(f"\n誤り応答検出完了: {len(dpo_pairs)}ペア")
    print("\n誤りの内訳:")
    for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {error_type}: {count}件")

    # JSONL形式で保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\n✅ DPOデータセット保存完了: {OUTPUT_PATH}")
    print(f"   総ペア数: {len(dpo_pairs)}")

    # サンプル表示
    if dpo_pairs:
        print("\n📋 サンプル（最初の3ペア）:")
        for i, pair in enumerate(dpo_pairs[:3], 1):
            print(f"\n--- ペア {i} ---")
            print(f"質問: {pair['prompt']}")
            print(f"正解: {pair['chosen'][:100]}...")
            print(f"誤答: {pair['rejected'][:100]}...")

    print("\n" + "=" * 80)
    print("🎉 データセット拡充完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
