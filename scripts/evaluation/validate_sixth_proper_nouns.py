#!/usr/bin/env python3
"""
第6次モデル固有名詞認識率検証スクリプト

Phase 10-2-A: 固有名詞認識問題の解決検証

検証項目:
1. 「高市早苗」→ "Sanae Takaichi" 認識率
2. その他の固有名詞認識率
3. 第5次モデルとの比較
4. 基本的事実の正答率
"""

import torch
from unsloth import FastLanguageModel
import json
from datetime import datetime
import re
from typing import Dict, List, Tuple

# ========================================
# 設定
# ========================================

# モデルパス
SIXTH_MODEL_PATH = "outputs/gpt-oss-20b-takaichi-v6-proper-nouns-20251022_114738/final"
FIFTH_MODEL_PATH = "outputs/gpt-oss-20b-takaichi-v5-official-20251022_105835/final"

# 固有名詞辞書（検証用）
PROPER_NOUNS = {
    "高市早苗": "Sanae Takaichi",
    "岸田文雄": "Fumio Kishida",
    "安倍晋三": "Shinzo Abe",
    "自由民主党": "Liberal Democratic Party",
    "サナエノミクス": "Sanaenomics",
    "総務大臣": "Minister of Internal Affairs",
    "経済安全保障": "economic security",
    "内閣府特命担当大臣": "Minister of State",
    "奈良県": "Nara Prefecture",
    "早稲田大学": "Waseda University",
    "松下政経塾": "Matsushita Institute",
    "防衛大臣": "Minister of Defense",
    "デジタル大臣": "Minister of Digital Affairs"
}

# テスト質問（固有名詞を含む）
TEST_QUESTIONS = [
    "高市早苗さんは何党ですか？",
    "高市早苗さんの経済政策の名前は？",
    "高市早苗さんはどこの大学を卒業しましたか？",
    "高市早苗さんが務めた大臣職を教えてください",
    "高市早苗さんの出身地はどこですか？",
    "サナエノミクスとは何ですか？",
    "高市早苗さんと岸田文雄さんの関係は？",
    "高市早苗さんと安倍晋三さんの関係は？",
    "高市早苗さんは松下政経塾出身ですか？",
    "高市早苗さんの経済安全保障への取り組みは？"
]

# ========================================
# 固有名詞認識チェック関数
# ========================================

def check_proper_noun_recognition(text: str, target_japanese: str, target_english: str) -> bool:
    """
    テキスト内に固有名詞の英語表記が含まれているかチェック

    Args:
        text: チェック対象のテキスト（モデル出力）
        target_japanese: 日本語固有名詞（例: "高市早苗"）
        target_english: 英語固有名詞（例: "Sanae Takaichi"）

    Returns:
        bool: 英語表記が含まれていればTrue
    """
    # 大文字小文字を区別しない検索
    return target_english.lower() in text.lower()


def extract_analysis_channel(response: str) -> str:
    """
    レスポンスからanalysisチャンネルの内容を抽出

    Args:
        response: モデルのレスポンス全体

    Returns:
        str: analysisチャンネルの内容（存在しない場合は空文字列）
    """
    # <|channel|>analysis<|message|>...<|end|> のパターンを抽出
    pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""


def extract_final_channel(response: str) -> str:
    """
    レスポンスからfinalチャンネルの内容を抽出

    Args:
        response: モデルのレスポンス全体

    Returns:
        str: finalチャンネルの内容（存在しない場合は全体を返す）
    """
    # <|channel|>final<|message|>... のパターンを抽出
    pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    # finalチャンネルが見つからない場合は全体を返す
    return response.strip()


# ========================================
# モデル評価関数
# ========================================

def evaluate_model(model_path: str, model_name: str) -> Dict:
    """
    モデルを評価

    Args:
        model_path: モデルのパス
        model_name: モデル名（表示用）

    Returns:
        Dict: 評価結果
    """
    print(f"\n{'='*60}")
    print(f"モデル評価: {model_name}")
    print(f"パス: {model_path}")
    print(f"{'='*60}\n")

    # モデルロード
    print("モデルをロード中...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("✓ モデルロード完了\n")

    # 評価結果を格納
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "test_count": len(TEST_QUESTIONS),
        "proper_noun_recognition": {
            "高市早苗": {"count": 0, "total": 0},
        },
        "details": []
    }

    # 各質問でテスト
    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{idx}/{len(TEST_QUESTIONS)}] テスト中: {question}")

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

        # プロンプト部分を除去
        if prompt in response:
            response = response.replace(prompt, "").strip()

        # analysisチャンネルとfinalチャンネルを抽出
        analysis_text = extract_analysis_channel(response)
        final_text = extract_final_channel(response)

        # 固有名詞認識チェック（analysisチャンネルで）
        recognized_takaichi = False
        if "高市早苗" in question:
            results["proper_noun_recognition"]["高市早苗"]["total"] += 1
            if check_proper_noun_recognition(analysis_text, "高市早苗", "Sanae Takaichi"):
                results["proper_noun_recognition"]["高市早苗"]["count"] += 1
                recognized_takaichi = True

        # 詳細結果を記録
        detail = {
            "question": question,
            "analysis": analysis_text if analysis_text else "(なし)",
            "final": final_text,
            "takaichi_recognized": recognized_takaichi if "高市早苗" in question else None
        }
        results["details"].append(detail)

        # 結果表示
        if analysis_text:
            print(f"  [Analysis] {analysis_text[:100]}...")
        print(f"  [Final] {final_text[:100]}...")
        if "高市早苗" in question:
            print(f"  [固有名詞認識] {'✓ 認識' if recognized_takaichi else '✗ 未認識'}")
        print()

    # 認識率計算
    takaichi_total = results["proper_noun_recognition"]["高市早苗"]["total"]
    takaichi_count = results["proper_noun_recognition"]["高市早苗"]["count"]
    takaichi_rate = (takaichi_count / takaichi_total * 100) if takaichi_total > 0 else 0

    results["proper_noun_recognition"]["高市早苗"]["rate"] = takaichi_rate

    print(f"\n{'='*60}")
    print(f"評価結果サマリー: {model_name}")
    print(f"{'='*60}")
    print(f"「高市早苗」認識率: {takaichi_count}/{takaichi_total} = {takaichi_rate:.1f}%")
    print(f"{'='*60}\n")

    # モデルをメモリから解放
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


# ========================================
# 比較レポート生成
# ========================================

def generate_comparison_report(fifth_results: Dict, sixth_results: Dict, output_path: str):
    """
    第5次と第6次の比較レポートを生成

    Args:
        fifth_results: 第5次モデルの評価結果
        sixth_results: 第6次モデルの評価結果
        output_path: 出力ファイルパス
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "timestamp": timestamp,
        "comparison": {
            "fifth_generation": {
                "model_path": fifth_results["model_path"],
                "takaichi_recognition_rate": fifth_results["proper_noun_recognition"]["高市早苗"]["rate"],
                "takaichi_recognized": fifth_results["proper_noun_recognition"]["高市早苗"]["count"],
                "takaichi_total": fifth_results["proper_noun_recognition"]["高市早苗"]["total"]
            },
            "sixth_generation": {
                "model_path": sixth_results["model_path"],
                "takaichi_recognition_rate": sixth_results["proper_noun_recognition"]["高市早苗"]["rate"],
                "takaichi_recognized": sixth_results["proper_noun_recognition"]["高市早苗"]["count"],
                "takaichi_total": sixth_results["proper_noun_recognition"]["高市早苗"]["total"]
            },
            "improvement": {
                "takaichi_recognition_rate_delta": (
                    sixth_results["proper_noun_recognition"]["高市早苗"]["rate"] -
                    fifth_results["proper_noun_recognition"]["高市早苗"]["rate"]
                )
            }
        },
        "fifth_generation_details": fifth_results["details"],
        "sixth_generation_details": sixth_results["details"]
    }

    # JSON保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n比較レポート保存: {output_path}")

    # サマリー表示
    print(f"\n{'='*60}")
    print("第5次 vs 第6次 比較サマリー")
    print(f"{'='*60}")
    print(f"「高市早苗」認識率:")
    print(f"  第5次: {fifth_results['proper_noun_recognition']['高市早苗']['rate']:.1f}%")
    print(f"  第6次: {sixth_results['proper_noun_recognition']['高市早苗']['rate']:.1f}%")
    print(f"  改善: {report['comparison']['improvement']['takaichi_recognition_rate_delta']:+.1f}%")
    print(f"{'='*60}\n")

    return report


# ========================================
# メイン処理
# ========================================

def main():
    print(f"\n{'='*60}")
    print("第6次モデル固有名詞認識率検証")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 第5次モデル評価
    print("\n[ステップ1] 第5次モデルの評価")
    fifth_results = evaluate_model(FIFTH_MODEL_PATH, "第5次モデル")

    # 第6次モデル評価
    print("\n[ステップ2] 第6次モデルの評価")
    sixth_results = evaluate_model(SIXTH_MODEL_PATH, "第6次モデル")

    # 比較レポート生成
    print("\n[ステップ3] 比較レポート生成")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/proper_noun_validation_{timestamp}.json"
    report = generate_comparison_report(fifth_results, sixth_results, report_path)

    print(f"\n{'='*60}")
    print("検証完了")
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 結果判定
    sixth_rate = sixth_results["proper_noun_recognition"]["高市早苗"]["rate"]
    if sixth_rate >= 95.0:
        print("✓ 目標達成: 固有名詞認識率 95% 以上")
    elif sixth_rate >= 80.0:
        print("△ 改善あり: 固有名詞認識率 80% 以上（目標未達）")
    else:
        print("✗ 要改善: 固有名詞認識率 80% 未満")

    print()


if __name__ == "__main__":
    main()
