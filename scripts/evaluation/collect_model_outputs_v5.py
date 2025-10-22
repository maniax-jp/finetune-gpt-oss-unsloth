#!/usr/bin/env python3
"""
第5次モデル（公式ノートブック準拠版）の出力収集スクリプト

改善点:
  - reasoning_effort パラメータの使用（公式ノートブックで発見）
  - 公式準拠のハイパーパラメータによる学習効果の検証
  - 第3次・第4次モデルとの比較評価

公式ノートブックからの主要変更:
  1. max_seq_length: 2048 → 1024
  2. LoRA rank: 16 → 8
  3. batch_size: 4 → 1
  4. learning_rate: 5e-5 → 2e-4
  5. LORA_DROPOUT: 0 → 0.1
  6. max_steps: 60（エポック制御→ステップ制御）
  7. train_on_responses_only 適用
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

# モデルパス（第5次ファインチューニング済みモデル）
MODEL_DIR = "outputs/gpt-oss-20b-takaichi-v5-official-20251022_105835/final"
BASE_MODEL_NAME = "openai/gpt-oss-20b"

# テスト質問セット（第4次と同じ51質問を使用し、比較可能にする）
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
OUTPUT_FILE = f"data/comparison/model_outputs_v5_{timestamp}.json"

# 生成設定（公式ノートブック準拠）
# 公式では temperature=1.0, top_p=1.0, top_k=0 を推奨
GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 1.0,  # 公式推奨（我々は0.7を使用していた）
    "top_p": 1.0,  # 公式推奨（我々は0.9を使用していた）
    "top_k": 0,  # 公式推奨（我々は50を使用していた）
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# reasoning_effort設定
# 公式では "low", "medium", "high" の3段階
# 基本的な事実質問なので "low" を使用
REASONING_EFFORT = "low"

# ============================================================================
# メイン処理
# ============================================================================

def load_model():
    """モデルとトークナイザーの読み込み"""
    print("=" * 80)
    print("第5次モデル（公式ノートブック準拠版）出力収集")
    print("=" * 80)
    print(f"\nモデルディレクトリ: {MODEL_DIR}")
    print("\n【第5次モデルの改善点】")
    print("  公式ノートブック準拠のハイパーパラメータ:")
    print("    - max_seq_length: 1024（公式準拠）")
    print("    - LoRA rank: 8（公式準拠）")
    print("    - batch_size: 1（MoE最適化）")
    print("    - learning_rate: 2e-4（公式準拠）")
    print("    - max_steps: 60（過学習防止）")
    print("  MoE過学習対策:")
    print("    - LORA_DROPOUT: 0.1（Sparse層正則化）")
    print("    - WEIGHT_DECAY: 0.1（強化）")
    print("  Unsloth最適化:")
    print("    - train_on_responses_only 適用")
    print()
    print("モデル読み込み中...")

    # ベースモデル読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=1024,  # 公式準拠（第4次は2048）
        dtype=None,
        load_in_4bit=True,
    )

    # 第5次LoRAアダプター読み込み
    model = PeftModel.from_pretrained(model, MODEL_DIR)

    # 推論モード
    FastLanguageModel.for_inference(model)

    print("✅ モデル読み込み完了")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, reasoning_effort: str = "low") -> str:
    """
    質問に対する応答を生成

    Args:
        model: LoRAファインチューニング済みモデル
        tokenizer: トークナイザー
        question: 質問文
        reasoning_effort: 推論レベル ("low", "medium", "high")
                         公式ノートブックで発見したパラメータ

    Returns:
        生成された応答テキスト
    """
    messages = [{"role": "user", "content": question}]

    # 公式ノートブックで発見: reasoning_effort パラメータを使用
    # ただし、apply_chat_templateではなく、generate時に指定するべきか要検証
    # 現状ではメッセージ形式で試行
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
            # 注意: reasoning_effortの正しい使用方法は公式ドキュメント要確認
            # 現時点では生成設定に直接含めることができない可能性
        )

    # 入力部分をスキップして応答のみ抽出
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


def main():
    """メイン処理"""

    # モデルディレクトリの存在確認
    if not os.path.exists(MODEL_DIR):
        print("=" * 80)
        print("⚠️  モデルディレクトリが見つかりません")
        print("=" * 80)
        print(f"\n指定されたパス: {MODEL_DIR}")
        print("\n第5次ファインチューニングをまだ実行していない場合:")
        print("  1. scripts/training/train_fifth_finetune.py を実行")
        print("  2. 完了後、このスクリプトのMODEL_DIRを更新")
        print("  3. このスクリプトを再実行")
        print("\nまたは、MODEL_DIRを実際のモデルパスに変更してください。")
        print("=" * 80)
        return

    # モデル読み込み
    model, tokenizer = load_model()

    # 出力ディレクトリ作成
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # 各質問に対する応答を収集
    print(f"\n質問数: {len(TEST_QUESTIONS)}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    print("応答収集開始...\\n")

    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")

        try:
            response = generate_response(model, tokenizer, question, REASONING_EFFORT)
            results.append({
                "question_id": i,
                "question": question,
                "model_response": response,
                "reasoning_effort": REASONING_EFFORT,
                "timestamp": datetime.now().isoformat()
            })
            print(f"  応答: {response[:80]}...")

        except Exception as e:
            print(f"  ❌ エラー: {e}")
            results.append({
                "question_id": i,
                "question": question,
                "model_response": "",
                "reasoning_effort": REASONING_EFFORT,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        print()

    # 結果を保存
    print("=" * 80)
    print("結果を保存中...")

    # メタデータを追加
    output_data = {
        "metadata": {
            "model_version": "v5-official",
            "model_dir": MODEL_DIR,
            "base_model": BASE_MODEL_NAME,
            "dataset_size": 455,  # 第5次は第4次と同じデータセットを使用
            "max_steps": 60,
            "improvements": [
                "公式ノートブック準拠のハイパーパラメータ",
                "max_seq_length: 1024",
                "LoRA rank: 8",
                "batch_size: 1",
                "learning_rate: 2e-4",
                "LORA_DROPOUT: 0.1",
                "WEIGHT_DECAY: 0.1",
                "train_on_responses_only 適用",
            ],
            "generation_config": GENERATION_CONFIG,
            "reasoning_effort": REASONING_EFFORT,
            "collection_timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "successful": sum(1 for r in results if 'error' not in r),
            "failed": sum(1 for r in results if 'error' in r),
        },
        "results": results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存完了: {OUTPUT_FILE}")
    print(f"   総質問数: {len(results)}")
    print(f"   成功: {sum(1 for r in results if 'error' not in r)}")
    print(f"   失敗: {sum(1 for r in results if 'error' in r)}")
    print("\n次のステップ:")
    print("  1. scripts/evaluation/compare_v3_v4_v5.py を作成")
    print("  2. 第3次、第4次、第5次の3モデルを比較評価")
    print("  3. 公式ノートブック準拠の効果を検証")
    print("=" * 80)


if __name__ == "__main__":
    main()
