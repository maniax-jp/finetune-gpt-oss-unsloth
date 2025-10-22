#!/usr/bin/env python3
"""
第7次モデル基本的事実回答収集スクリプト

目的:
1. 第7次モデルに基本的事実の質問をする
2. analysisチャンネルとfinalチャンネルを分けて表示・保存
3. 正誤判定は人間が行う
"""

import torch
from unsloth import FastLanguageModel
import json
from datetime import datetime
import re
from typing import Dict, List

# ========================================
# 設定
# ========================================

SEVENTH_MODEL_PATH = "outputs/gpt-oss-20b-takaichi-v7-full-reasoning-20251022_153026/final"
COMPARISON_DATA_PATH = "data/comparison/v3_v4_comparison_20251022_101622.json"

# ========================================
# 関数
# ========================================

def extract_analysis_channel(response: str) -> str:
    """analysisチャンネルの内容を抽出"""
    pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_final_channel(response: str) -> str:
    """finalチャンネルの内容を抽出"""
    pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def load_questions() -> List[str]:
    """比較データから質問をロード"""
    with open(COMPARISON_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = [item['question'] for item in data['detailed_results'][:30]]
    return questions


# ========================================
# メイン処理
# ========================================

def main():
    print(f"\n{'='*60}")
    print("第7次モデル基本的事実回答収集")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 質問をロード
    print("[ステップ1] 質問をロード...")
    questions = load_questions()
    print(f"  ✓ 質問数: {len(questions)}\n")

    # モデルロード
    print("[ステップ2] 第7次モデルをロード...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SEVENTH_MODEL_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("  ✓ モデルロード完了\n")

    # 回答を収集
    print("[ステップ3] 回答を収集中...\n")
    print(f"{'='*60}\n")

    results = []

    for idx, question in enumerate(questions, 1):
        print(f"[質問 {idx}/{len(questions)}]")
        print(f"{question}\n")

        # プロンプト作成
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 推論実行
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # デコード
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if prompt in response:
            response = response.replace(prompt, "").strip()

        # analysisチャンネルとfinalチャンネルを抽出
        analysis_text = extract_analysis_channel(response)
        final_text = extract_final_channel(response)

        # 結果表示
        if analysis_text:
            print(f"[Analysis]")
            print(f"{analysis_text}\n")

        print(f"[Final]")
        print(f"{final_text}\n")
        print(f"{'-'*60}\n")

        # 結果記録
        results.append({
            'question': question,
            'analysis': analysis_text,
            'final': final_text,
            'raw_response': response
        })

    # JSON保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/comparison/seventh_model_responses_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": SEVENTH_MODEL_PATH,
        "total_questions": len(questions),
        "responses": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("収集完了")
    print(f"{'='*60}")
    print(f"総質問数: {len(questions)}")
    print(f"保存先: {output_path}")
    print(f"{'='*60}\n")

    # モデルをメモリから解放
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
