#!/usr/bin/env python3
"""
第6次モデル簡易固有名詞認識テスト

第6次モデルのみを検証し、固有名詞認識率を迅速に確認
"""

import torch
from unsloth import FastLanguageModel
import json
from datetime import datetime
import re

# ========================================
# 設定
# ========================================

SEVENTH_MODEL_PATH = "outputs/gpt-oss-20b-takaichi-v7-full-reasoning-20251022_153026/final"

# テスト質問（固有名詞を含む）
TEST_QUESTIONS = [
    "高市早苗さんは何党ですか？",
    "高市早苗さんの経済政策の名前は？",
    "高市早苗さんはどこの大学を卒業しましたか？",
    "高市早苗さんが務めた大臣職を教えてください",
    "高市早苗さんの出身地はどこですか？",
]

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


def check_takaichi_recognition(text: str) -> bool:
    """「高市早苗」→ "Sanae Takaichi" 認識チェック"""
    return "sanae takaichi" in text.lower()


# ========================================
# メイン処理
# ========================================

def main():
    print(f"\n{'='*60}")
    print("第7次モデル簡易固有名詞認識テスト")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # モデルロード
    print("モデルをロード中...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SEVENTH_MODEL_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("✓ モデルロード完了\n")

    # 結果を格納
    results = []
    recognized_count = 0
    total_count = len(TEST_QUESTIONS)

    # 各質問でテスト
    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{idx}/{total_count}] テスト中: {question}")

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

        # analysisチャンネルを抽出
        analysis_text = extract_analysis_channel(response)

        # 固有名詞認識チェック
        is_recognized = check_takaichi_recognition(analysis_text)
        if is_recognized:
            recognized_count += 1

        # 結果表示
        if analysis_text:
            print(f"  [Analysis抜粋] {analysis_text[:150]}...")
        print(f"  [固有名詞認識] {'✓ 認識' if is_recognized else '✗ 未認識'}\n")

        # 結果記録
        results.append({
            "question": question,
            "analysis": analysis_text[:300] if analysis_text else "(なし)",
            "recognized": is_recognized
        })

    # 認識率計算
    recognition_rate = (recognized_count / total_count * 100) if total_count > 0 else 0

    print(f"\n{'='*60}")
    print("検証結果サマリー")
    print(f"{'='*60}")
    print(f"「高市早苗」認識率: {recognized_count}/{total_count} = {recognition_rate:.1f}%")
    print(f"{'='*60}\n")

    # 結果判定
    if recognition_rate >= 95.0:
        print("✓ 目標達成: 固有名詞認識率 95% 以上")
        result_status = "success"
    elif recognition_rate >= 80.0:
        print("△ 改善あり: 固有名詞認識率 80% 以上（目標未達）")
        result_status = "partial"
    elif recognition_rate > 0:
        print("△ 一部改善: 固有名詞認識率向上（さらなる改善必要）")
        result_status = "improved"
    else:
        print("✗ 未改善: 固有名詞認識率 0%（要対策）")
        result_status = "failed"

    # JSON保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/quick_validation_seventh_{timestamp}.json"
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": SEVENTH_MODEL_PATH,
        "recognition_rate": recognition_rate,
        "recognized_count": recognized_count,
        "total_count": total_count,
        "status": result_status,
        "details": results
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n検証結果保存: {report_path}\n")

    # モデルをメモリから解放
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
