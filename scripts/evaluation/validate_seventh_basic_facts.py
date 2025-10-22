#!/usr/bin/env python3
"""
第7次モデル基本的事実検証スクリプト

目的:
1. 第7次モデルで基本的事実30問を検証
2. 第3次モデルの結果と比較
3. エラー率が10%以下であることを確認（目標: 90%以上正答率）
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

def extract_final_channel(response: str) -> str:
    """finalチャンネルの内容を抽出"""
    pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def load_questions() -> List[Dict]:
    """比較データから質問と第3次モデルのエラー情報をロード"""
    with open(COMPARISON_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    for item in data['detailed_results'][:30]:  # 最初の30問
        questions.append({
            'question': item['question'],
            'v3_response': item['v3_response'],
            'v3_errors': item['v3_errors']
        })

    return questions


def evaluate_response(question: str, response: str, v3_errors: List[str]) -> Dict:
    """
    レスポンスを評価（簡易版）

    第3次モデルで検出されたエラーと同じエラーがあるかチェック
    """
    errors = []

    # 第3次モデルのエラーパターンをチェック
    if v3_errors:
        for error in v3_errors:
            # 生年月日エラー
            if "生年月日" in error and "1960年" in error:
                if "1960" in response or "1961" in response:
                    errors.append("生年月日の誤り: 1960年代")
            elif "生年月日" in error and "1970年" in error:
                if "1970" in response:
                    errors.append("生年月日の誤り: 1970年")

            # 出身地エラー
            elif "出身地" in error:
                if "岐阜" in response or "山口" in response or "京都" in response:
                    errors.append(f"出身地の誤り: {response[:50]}...")

            # 大学エラー
            elif "大学" in error:
                if "早稲田" in response or "東京大学" in response:
                    errors.append(f"大学の誤り: {response[:50]}...")

            # 政策エラー
            elif "政策" in error or "公約" in error:
                errors.append(f"政策情報の誤り: {response[:50]}...")

    # 新しいエラーパターンもチェック（基本的な事実誤り）
    # 生年月日の範囲チェック
    if "生年月日" in question or "いつ生まれ" in question:
        # 正解: 1961年3月7日
        if "1960" in response or "1970" in response or "1965" in response:
            if not any("生年月日" in e for e in errors):
                errors.append("生年月日の誤り")

    # 出身地チェック
    if "出身地" in question:
        # 正解: 奈良県
        if "岐阜" in response or "山口" in response or "京都" in response or "大阪" in response:
            if not any("出身地" in e for e in errors):
                errors.append("出身地の誤り")

    # 大学チェック
    if "大学" in question or "学歴" in question:
        # 正解: 神戸大学
        if ("早稲田" in response or "東京大学" in response or "京都大学" in response) and "神戸大学" not in response:
            if not any("大学" in e for e in errors):
                errors.append("大学の誤り")

    return {
        'errors': errors,
        'error_count': len(errors),
        'is_correct': len(errors) == 0
    }


# ========================================
# メイン処理
# ========================================

def main():
    print(f"\n{'='*60}")
    print("第7次モデル基本的事実検証")
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

    # 評価実行
    print("[ステップ3] 基本的事実を検証中...\n")
    results = []
    total_errors = 0
    correct_count = 0

    for idx, item in enumerate(questions, 1):
        question = item['question']
        v3_errors = item['v3_errors']

        print(f"[{idx}/{len(questions)}] {question}")

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

        # finalチャンネルを抽出
        final_text = extract_final_channel(response)

        # 評価
        evaluation = evaluate_response(question, final_text, v3_errors)

        # 結果表示
        print(f"  回答: {final_text[:100]}...")
        if evaluation['errors']:
            print(f"  ❌ エラー: {', '.join(evaluation['errors'])}")
            total_errors += evaluation['error_count']
        else:
            print(f"  ✓ 正解")
            correct_count += 1
        print()

        # 結果記録
        results.append({
            'question': question,
            'response': final_text,
            'v3_errors': v3_errors,
            'v7_errors': evaluation['errors'],
            'is_correct': evaluation['is_correct']
        })

    # サマリー
    accuracy_rate = (correct_count / len(questions) * 100) if len(questions) > 0 else 0
    error_rate = (total_errors / len(questions) * 100) if len(questions) > 0 else 0

    print(f"\n{'='*60}")
    print("検証結果サマリー")
    print(f"{'='*60}")
    print(f"総質問数: {len(questions)}")
    print(f"正解数: {correct_count}")
    print(f"エラー数: {total_errors}")
    print(f"正答率: {accuracy_rate:.1f}%")
    print(f"エラー率: {error_rate:.1f}%")
    print(f"{'='*60}\n")

    # 第3次モデルとの比較
    v3_total_errors = sum(len(item['v3_errors']) for item in questions)
    print(f"【第3次モデルとの比較】")
    print(f"第3次モデルエラー数: {v3_total_errors}")
    print(f"第7次モデルエラー数: {total_errors}")
    print(f"エラー差分: {total_errors - v3_total_errors:+d}")
    print()

    # 目標達成判定
    if error_rate <= 10.0:
        print("✓ 目標達成: エラー率 10% 以下")
        result_status = "success"
    elif accuracy_rate >= 80.0:
        print("△ 改善あり: 正答率 80% 以上（目標未達）")
        result_status = "partial"
    else:
        print("✗ 要改善: 正答率 80% 未満")
        result_status = "failed"
    print()

    # JSON保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/basic_facts_validation_v7_{timestamp}.json"
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": SEVENTH_MODEL_PATH,
        "total_questions": len(questions),
        "correct_count": correct_count,
        "total_errors": total_errors,
        "accuracy_rate": accuracy_rate,
        "error_rate": error_rate,
        "v3_total_errors": v3_total_errors,
        "error_change": total_errors - v3_total_errors,
        "status": result_status,
        "details": results
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"検証結果保存: {report_path}\n")

    # モデルをメモリから解放
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
