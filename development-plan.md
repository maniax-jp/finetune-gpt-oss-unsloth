# GPT-OSS 20B ファインチューニング開発計画

## プロジェクト概要

UnslothフレームワークとRTX 5090 (VRAM 32GB)を使用して、GPT-OSS 20BモデルをQLoRAでファインチューニングし、Ollama形式でエクスポートする。

## 環境仕様

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **アーキテクチャ**: Blackwell (SM 12.0)
- **モデル**: GPT-OSS 20B (MXFP4形式)
- **フレームワーク**: Unsloth
- **量子化手法**: QLoRA (4-bit)
- **データ形式**: Harmony format

## 主要な技術的ポイント

### 1. MXFP4について
- GPT-OSSモデルはMoEレイヤーにMXFP4精度を使用
- MXFP4ネイティブでは学習の逆伝播が未実装
- UnslothはBitsandBytes NF4量子化でMXFP4を模倣
- RTX 5090のBlackwellアーキテクチャはFP4をネイティブサポート

### 2. QLoRAについて
- Unslothは唯一GPT-OSSのQLoRA 4-bit学習をサポート
- VRAMを4倍以上削減（GPT-OSS 20Bで14GB要求）
- LoRAアダプターを使用してメモリ効率的な学習を実現

### 3. Harmonyフォーマットについて
- OpenAI公式のGPT-OSS専用会話形式
- GPT-OSSモデルはHarmony形式でのみ正しく動作
- 重要な設定項目：
  - `reasoning_effort`: low/medium/high（推論バジェット制御）
  - `developer_instructions`: システムプロンプト相当
  - `model_identity`: 基本的にデフォルト使用

### 4. Ollama対応について
- GGUF形式への変換が必要
- Modelfileの自動生成（chat template含む）
- 推奨量子化: Q8_0またはq4_k_m

## 開発ステップ

### Phase 1: 環境構築

#### 1.1 システム要件確認
- [ ] NVIDIA Driver確認（RTX 5090対応版）
- [ ] CUDA Toolkit インストール確認
- [ ] Python 3.12のインストール
- [ ] WSL環境の場合、メモリ制限の設定

#### 1.2 Unslothインストール（RTX 5090対応）

**オプションA: Docker使用（推奨）**
```bash
docker pull unsloth/unsloth
```

**オプションB: ローカルインストール**
```bash
# 仮想環境作成
python3.12 -m venv venv
source venv/bin/activate

# Blackwell対応でUnslothインストール
export TORCH_CUDA_ARCH_LIST="12.0"
pip install unsloth

# Triton更新
pip install -U triton>=3.3.1

# Xformers（オプション、推奨）
pip install xformers --no-build-isolation
```

#### 1.3 必要なパッケージインストール
```bash
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install datasets transformers accelerate bitsandbytes
pip install ollama  # Ollama Python SDK
```

### Phase 2: データセット準備

#### 2.1 データセット要件
- [ ] Harmony形式対応のデータセット作成
- [ ] 推論例を75%以上含める（GPT-OSS推奨）
- [ ] 会話形式のJSON/JSONLデータ

#### 2.2 Harmonyフォーマット変換
```python
from unsloth import encode_conversations_with_harmony

# データセット例
dataset = [
    {
        "messages": [
            {"role": "user", "content": "質問内容"},
            {"role": "assistant", "content": "回答内容"}
        ]
    }
]

# Harmony形式でエンコード
encoded_dataset = encode_conversations_with_harmony(
    dataset,
    reasoning_effort="medium"  # low/medium/high
)
```

#### 2.3 データセット検証
- [ ] Harmony形式の正確性確認
- [ ] トークン長の確認（コンテキスト制限内）
- [ ] データ品質チェック

### Phase 3: モデル読み込みと設定

#### 3.1 ベースモデル読み込み
```python
from unsloth import FastLanguageModel

# モデルパラメータ
max_seq_length = 4096  # コンテキスト長（必要に応じて調整）
dtype = None  # 自動検出
load_in_4bit = True  # QLoRA 4-bit量子化

# モデル読み込み（MXFP4対応）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-BF16",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

#### 3.2 LoRAアダプター設定
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

### Phase 4: ファインチューニング実行

#### 4.1 トレーニング設定
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,  # 必要に応じて調整
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
```

#### 4.2 トレーニング実行
```python
trainer_stats = trainer.train()
```

#### 4.3 モニタリング
- [ ] VRAMメモリ使用量監視（nvidia-smi）
- [ ] 損失関数の推移確認
- [ ] 学習速度の確認

### Phase 5: モデルエクスポート

#### 5.1 LoRAアダプターマージ（オプション）
```python
# BF16形式でマージ
model.save_pretrained_merged(
    "gpt-oss-20b-finetuned",
    tokenizer,
    save_method="merged_16bit"
)
```

#### 5.2 GGUF形式変換（Ollama用）
```python
# Q8_0量子化でGGUFエクスポート
model.save_pretrained_gguf(
    "gpt-oss-20b-finetuned",
    tokenizer,
    quantization_method="q8_0"  # または "q4_k_m"
)
```

#### 5.3 Ollama Modelfile生成
Unslothが自動生成するModelfileを確認：
```
FROM ./gpt-oss-20b-finetuned-Q8_0.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

#### 5.4 Ollamaへのインポート
```bash
# Ollamaモデル作成
ollama create gpt-oss-20b-finetuned -f ./Modelfile

# モデル動作確認
ollama run gpt-oss-20b-finetuned "こんにちは"
```

### Phase 6: 検証とテスト

#### 6.1 推論テスト
```python
# Unsloth環境でのテスト
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    "テストプロンプト",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    use_cache=True
)

print(tokenizer.decode(outputs[0]))
```

#### 6.2 Ollama環境でのテスト
```bash
# Ollamaでの推論テスト
ollama run gpt-oss-20b-finetuned "複雑な推論が必要な質問"
```

#### 6.3 品質評価
- [ ] ベースラインモデルとの比較
- [ ] Harmony形式の動作確認
- [ ] 推論品質の評価
- [ ] メモリ使用量の確認

### Phase 7: ドキュメント化

#### 7.1 作成すべきドキュメント
- [ ] 学習パラメータの記録
- [ ] データセット仕様書
- [ ] モデル性能評価レポート
- [ ] Ollama使用方法のREADME

#### 7.2 モデル管理
- [ ] チェックポイントの保存
- [ ] バージョン管理
- [ ] Hugging Facehubへのアップロード（オプション）

## トラブルシューティング

### 一般的な問題と解決策

#### VRAM不足
- batch sizeを削減（例: 2 → 1）
- gradient_accumulation_stepsを増加
- max_seq_lengthを削減
- LoRA rankを削減（16 → 8）

#### Ollama推論で文字化けや無限生成
- **原因**: chat templateの不一致
- **解決**: Modelfile内のTEMPLATEを学習時と同じ形式に修正

#### RTX 5090特有の問題
- TORCH_CUDA_ARCH_LIST="12.0"の設定確認
- Tritonバージョン確認（>=3.3.1）
- Docker imageの使用を検討

#### Harmonyフォーマットエラー
- reasoning_effort設定の確認
- OpenAIのHarmonyライブラリドキュメント参照
- データセット形式の再確認

## 参考リソース

### 公式ドキュメント
- [Unsloth GPT-OSS Documentation](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [Unsloth Blackwell/RTX 50 Series Guide](https://docs.unsloth.ai/basics/training-llms-with-blackwell-rtx-50-series-and-unsloth)
- [Saving to Ollama Guide](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-ollama)
- [Saving to GGUF Guide](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf)

### Colabノートブック
- [GPT-OSS 20B Fine-tuning Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)

### GitHub
- [Unsloth Repository](https://github.com/unslothai/unsloth)

### Hugging Face
- [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16)
- [unsloth/gpt-oss-20b-GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

## 予想される成果物

1. **ファインチューニング済みモデル**
   - LoRAアダプター（safetensors形式）
   - マージ済みモデル（BF16形式）

2. **Ollama対応モデル**
   - GGUF形式モデルファイル（Q8_0またはq4_k_m）
   - Modelfile

3. **ドキュメント**
   - トレーニングログ
   - 性能評価レポート
   - 使用方法ガイド

4. **スクリプト**
   - データ前処理スクリプト
   - トレーニングスクリプト
   - エクスポートスクリプト

## プロジェクトタイムライン（推定）

- **Phase 1**: 環境構築（1-2日）
- **Phase 2**: データセット準備（2-3日）
- **Phase 3**: モデル設定（1日）
- **Phase 4**: ファインチューニング（データセットサイズにより変動）
- **Phase 5**: エクスポート（0.5-1日）
- **Phase 6**: 検証（1-2日）
- **Phase 7**: ドキュメント化（1日）

**合計**: 約1-2週間（データセット準備とトレーニング時間による）

## 次のアクションアイテム

1. RTX 5090用のNVIDIAドライバーとCUDA環境の確認
2. Unsloth環境のセットアップ（DockerまたはPython venv）
3. ファインチューニング用データセットの準備開始
4. 小規模データセットでのテストラン実施
5. 本番データセットでの完全なファインチューニング実行

---

## Phase 8: 性能改善計画（第2次ファインチューニング）

### 8.1 第1次ファインチューニングの結果分析

**実施内容:**
- データセット: 101サンプル（高市早苗QA）
- トレーニング: 20 epochs
- LoRA設定: rank=64, lr=1e-4
- 損失: 12.73 → 0.355 (97.2%改善)

**問題点:**
1. **知識の注入不足**: モデルが空の応答を返す、または無関係な内容を生成
2. **訓練データ量の不足**: 101サンプル × 20エポック = 2,020学習ステップでは不十分
3. **RLHF未実施**: 人間のフィードバックによる調整が欠如

### 8.2 改善戦略

#### 戦略1: データセット拡充

**目標:** 101サンプル → 300-500サンプル（3-5倍増）

**アプローチA: 既存データの拡張**
```python
# 既存QAペアから派生パターンを生成
expansion_strategies = {
    "paraphrase": "質問を別の言い方で表現",
    "detail_levels": "詳細度を変えた質問（簡潔版/詳細版）",
    "context_variation": "文脈を変えた質問",
    "follow_up": "フォローアップ質問を追加",
    "multi_turn": "単発QAを複数ターンの会話に変換"
}
```

**具体例:**
```json
// 元データ
{"Q": "高市早苗さんは何党ですか？", "A": "自由民主党（自民党）です"}

// 拡張パターン
[
  {"Q": "高市早苗氏の所属政党は？", "A": "自由民主党（自民党）です"},
  {"Q": "高市早苗さんってどこの政党？", "A": "自由民主党（自民党）に所属しています"},
  {"Q": "高市早苗議員の政党について教えて", "A": "自由民主党（自民党）の所属議員です"},
  // マルチターン会話
  {
    "messages": [
      {"role": "user", "content": "高市早苗さんについて教えて"},
      {"role": "assistant", "content": "高市早苗氏は自由民主党所属の政治家です。"},
      {"role": "user", "content": "何党ですか？"},
      {"role": "assistant", "content": "自由民主党（自民党）です"}
    ]
  }
]
```

**アプローチB: 新規データ収集**
- Wikipedia、公式サイト、ニュース記事から追加情報を抽出
- 経歴、政策、発言、実績などのカテゴリー別に体系的に収集
- ファクトチェック済みの信頼できる情報源のみ使用

**データ品質基準:**
1. 事実の正確性（公式情報源で検証可能）
2. 多様性（カテゴリー、質問パターン、回答スタイル）
3. バランス（各トピックをほぼ均等にカバー）
4. 文脈の一貫性（会話として自然な流れ）

**実装スクリプト例:**
```python
# scripts/augment_dataset.py
import json
from typing import List, Dict

def paraphrase_question(q: str, a: str) -> List[Dict]:
    """質問のパラフレーズを生成"""
    variations = []
    # 疑問詞の変換
    patterns = [
        (q, a),
        (q.replace("何", "どの"), a),
        (q.replace("ですか", "でしょうか"), a),
        (q + "について教えて", a),
    ]
    return [{"Q": q_var, "A": a_var} for q_var, a_var in patterns]

def create_multi_turn(qa_pairs: List[Dict]) -> Dict:
    """複数のQAペアをマルチターン会話に変換"""
    messages = []
    for qa in qa_pairs:
        messages.append({"role": "user", "content": qa["Q"]})
        messages.append({"role": "assistant", "content": qa["A"]})
    return {"messages": messages}

# 使用例
original_dataset = load_json("data/takaichi_qa.json")
augmented_dataset = []

for item in original_dataset:
    # 元データ追加
    augmented_dataset.append(item)

    # パラフレーズ追加
    augmented_dataset.extend(paraphrase_question(item["Q"], item["A"]))

# 目標: 101 × 3 = 303サンプル（最低）
```

#### 戦略2: RLHF (Reinforcement Learning from Human Feedback) 実装

**目標:** 人間のフィードバックに基づくモデルの調整

**Phase 8.3: Reward Modelの構築**

```python
# scripts/build_reward_model.py
from unsloth import FastLanguageModel
from trl import RewardTrainer
from transformers import TrainingArguments

# ステップ1: 比較データセット作成
comparison_dataset = [
    {
        "prompt": "高市早苗さんは何党ですか？",
        "chosen": "自由民主党（自民党）です。",  # 良い回答
        "rejected": "区に位置するこの施設は..."  # 悪い回答（実際の出力）
    },
    # 50-100サンプル収集
]

# ステップ2: Reward Modelトレーニング
reward_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/gpt-oss-20b-takaichi-20251009_200301/final",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

reward_model = FastLanguageModel.get_peft_model(
    reward_model,
    r=32,  # Reward modelは高いrankを使用
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
)

reward_trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    train_dataset=comparison_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        output_dir="outputs/reward_model",
    ),
)

reward_trainer.train()
```

**Phase 8.4: PPO (Proximal Policy Optimization) ファインチューニング**

```python
# scripts/ppo_training.py
from trl import PPOTrainer, PPOConfig
from unsloth import FastLanguageModel

# ステップ1: モデルロード
policy_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/gpt-oss-20b-takaichi-20251009_200301/final",
    max_seq_length=2048,
    load_in_4bit=True,
)

# ステップ2: PPO設定
ppo_config = PPOConfig(
    model_name="gpt-oss-takaichi-ppo",
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
    ppo_epochs=4,
    remove_unused_columns=False,
)

# ステップ3: PPOトレーナー
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    tokenizer=tokenizer,
    reward_model=reward_model,  # 前ステップで訓練したモデル
)

# ステップ4: トレーニング
prompts = [item["prompt"] for item in comparison_dataset]
for epoch in range(10):
    for prompt in prompts:
        # 生成
        response = ppo_trainer.generate(prompt)
        # 報酬計算
        reward = reward_model(prompt, response)
        # PPO更新
        ppo_trainer.step([prompt], [response], [reward])
```

**Phase 8.5: DPO (Direct Preference Optimization) 実装（代替アプローチ）**

PPOより簡単で安定したアプローチ:

```python
# scripts/dpo_training.py
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig

# DPO用データセット（比較データ）
dpo_dataset = [
    {
        "prompt": "高市早苗さんは何党ですか？",
        "chosen": "自由民主党（自民党）です。",
        "rejected": "区に位置するこの施設は、近隣の住民に..."
    },
    # 100-200サンプル
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/gpt-oss-20b-takaichi-20251009_200301/final",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # DPOは参照モデルを自動作成
    args=DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-6,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs/dpo",
    ),
    beta=0.1,  # DPO温度パラメータ
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    max_length=2048,
    max_prompt_length=1024,
)

dpo_trainer.train()
```

#### 戦略3: ハイブリッドアプローチ（推奨）

**3段階トレーニングパイプライン:**

1. **SFT (Supervised Fine-Tuning)**: 拡張データセット300-500サンプルで基礎学習
2. **DPO**: 100-200の比較データで好みの調整
3. **評価と反復**: テスト → フィードバック → 再訓練

```python
# scripts/hybrid_training_pipeline.py

# ステージ1: SFT（20-30 epochs）
sft_trainer = SFTTrainer(
    model=base_model,
    train_dataset=augmented_dataset,  # 300-500サンプル
    max_steps=None,
    num_train_epochs=25,
    # ... その他の設定
)
sft_model = sft_trainer.train()

# ステージ2: DPO（3-5 epochs）
dpo_trainer = DPOTrainer(
    model=sft_model,
    train_dataset=preference_dataset,  # 100-200比較サンプル
    num_train_epochs=3,
    # ... その他の設定
)
final_model = dpo_trainer.train()

# ステージ3: 評価
test_prompts = [...]
for prompt in test_prompts:
    response = generate(final_model, prompt)
    score = human_evaluate(response)
    # スコアが低い場合は再訓練
```

### 8.3 実装ロードマップ

**Week 1-2: データセット拡充**
- [ ] 既存101サンプルの分析とカテゴリー分類
- [ ] パラフレーズツールの実装
- [ ] 新規データ収集（Wikipedia、公式サイト）
- [ ] データ品質検証とクリーニング
- [ ] 目標300サンプル達成

**Week 3: 第2次SFTトレーニング**
- [ ] 拡張データセットでSFT実施（25-30 epochs）
- [ ] 学習曲線のモニタリング
- [ ] 中間チェックポイントの評価

**Week 4: 比較データセット作成**
- [ ] モデル出力の収集（50質問）
- [ ] 良い回答/悪い回答のペア作成（100-200ペア）
- [ ] 人間評価者によるレビュー

**Week 5: DPO/PPOトレーニング**
- [ ] DPO実装とトレーニング（3-5 epochs）
- [ ] または PPO実装（より高度）
- [ ] ハイパーパラメータチューニング

**Week 6: 評価と反復**
- [ ] 包括的なテストスイート実行
- [ ] 人間評価（複数評価者）
- [ ] 必要に応じて再訓練
- [ ] 最終モデルのOllamaエクスポート

### 8.4 評価指標

**自動評価:**
- Perplexity（困惑度）
- BLEU/ROUGE scores（参照回答との類似度）
- 応答生成率（空応答の割合）

**人間評価:**
- 事実正確性（1-5点）
- 関連性（1-5点）
- 流暢さ（1-5点）
- 有用性（1-5点）

```python
# scripts/evaluation.py
evaluation_criteria = {
    "factual_accuracy": "回答は事実に基づいているか",
    "relevance": "質問に適切に答えているか",
    "fluency": "自然な日本語か",
    "helpfulness": "ユーザーにとって有用か"
}

test_set = [
    {"question": "高市早苗さんは何党ですか？",
     "expected": "自由民主党（自民党）です"},
    # 50-100テストケース
]

def evaluate_model(model, test_set):
    scores = {"accuracy": [], "relevance": [], "fluency": [], "helpfulness": []}

    for item in test_set:
        response = generate(model, item["question"])

        # 人間評価（またはGPT-4評価）
        scores["accuracy"].append(rate_accuracy(response, item["expected"]))
        scores["relevance"].append(rate_relevance(response, item["question"]))
        scores["fluency"].append(rate_fluency(response))
        scores["helpfulness"].append(rate_helpfulness(response))

    return {k: sum(v)/len(v) for k, v in scores.items()}
```

### 8.5 期待される改善効果

**改善前（第1次ファインチューニング）:**
- データ: 101サンプル
- 応答品質: 空応答 or 無関係な内容
- 事実正確性: 0%

**改善後（第2次ファインチューニング + RLHF）:**
- データ: 300-500サンプル + 100-200比較ペア
- 応答品質: 一貫した高品質な回答
- 事実正確性: 80-90%（目標）
- 応答生成率: 100%（空応答なし）

### 8.6 リスクと対策

**リスク1: データ収集の労力**
- 対策: 半自動化ツールの活用、段階的拡張（まず200サンプル）

**リスク2: VRAM不足（RLHF時）**
- 対策: DPOを優先（PPOより軽量）、バッチサイズ調整

**リスク3: 過学習**
- 対策: validation setの設定、early stopping

**リスク4: データ品質の低下**
- 対策: 厳密なレビュープロセス、ファクトチェック

### 8.7 次のステップ

1. **即座に実行可能:**
   - 既存101サンプルのパラフレーズによる拡張（→303サンプル）
   - 第2次SFTトレーニング実施

2. **短期（1-2週間）:**
   - 新規データ収集で300-500サンプルへ拡大
   - 比較データセット作成開始

3. **中期（3-4週間）:**
   - DPOトレーニング実施
   - 包括的評価と反復改善

4. **長期（オプション）:**
   - RAG (Retrieval-Augmented Generation) との統合検討
   - より大規模なデータセット（1000+サンプル）への拡張

---

## Phase 9: Week 5 - DPO/PPOトレーニング（第3次ファインチューニング）

### 9.1 Week 4の成果と課題

**Week 4完了内容:**
- ✅ モデル出力収集: 51質問に対する応答収集完了
- ✅ 自動評価実施: 良い応答84.3%、平均評価4.43/5.0
- ✅ 比較データセット作成: 7ペアのDPOデータセット作成完了

**Week 4で判明した問題点:**

1. **基本的事実の誤り（8件検出）:**
   - 生年月日: 1961年3月7日 → 誤答「1960年11月23日」
   - 出身地: 奈良県 → 誤答「愛知県名古屋市」
   - 学歴: 神戸大学経営学部 → 誤答「東京大学法学部」
   - 選挙区: 奈良2区 → 誤答「千葉県第2選挙区」
   - 大臣経験: 総務大臣、経済安全保障担当大臣 → 誤答「経済産業大臣、外務大臣」
   - 初当選: 1993年 → 誤答「2003年」
   - 総裁選出馬: 2021年、2024年 → 誤答「2018年」
   - 応答品質: 短すぎる応答（例: 「保守派です。」のみ）

2. **モデルの弱点分析:**
   - 基本的な固有名詞（日付、地名、大学名）の記憶が不正確
   - 他の政治家の情報と混同している可能性
   - 幻覚（Hallucination）による存在しない情報の生成

3. **モデルの強み（維持すべき点）:**
   - ✅ 政策・主張に関する応答は適切
   - ✅ 複雑な質問への推論能力
   - ✅ 比較質問への対応力

### 9.2 Week 5の目標

**主目的:** 事実の正確性を向上させる

**目標指標:**
- 基本的事実の正確性: 現在15.7%誤り → 目標5%以下
- 平均評価: 現在4.43/5.0 → 目標4.7/5.0以上
- 空応答/短すぎる応答: 0件

### 9.3 戦略: DPOトレーニング

**DPO (Direct Preference Optimization) を選択する理由:**
1. PPOより実装が簡単で安定している
2. 報酬モデルの訓練が不要（PPOは必要）
3. メモリ効率が良い（RTX 5090の32GB VRAMで十分）
4. 基本的事実の修正に効果的

### 9.4 Week 5: タスク詳細

#### タスク1: データセット拡充（優先度: 最高）

**現状:** 7ペア（不十分）
**目標:** 100-200ペア

**拡充方法:**

**方法A: 既存301サンプルからの誤り抽出**
```python
# scripts/evaluation/extract_errors_from_training_data.py
# 第2次トレーニングデータ（301サンプル）に対してモデルを実行し、
# 誤った応答を自動的に抽出
```

**方法B: 温度パラメータを上げた多様性生成**
```python
# 意図的に temperature=1.0-1.5 で生成して多様な誤り応答を収集
generation_config = {
    "temperature": 1.2,  # 高温で多様性確保
    "top_p": 0.95,
    "do_sample": True,
}
```

**方法C: 手動でのchosen/rejectedペア作成**
```python
# カテゴリー別にバランスよく作成
target_distribution = {
    "基本情報（生年月日、出身地、学歴など）": 30,
    "政治活動（選挙区、大臣経験など）": 30,
    "政策・主張": 20,
    "実績・業績": 20,
    "その他": 20,
}
# 合計: 120ペア
```

**データセット品質基準:**
- 各ペアは明確な「良い応答」と「悪い応答」の対比があること
- 悪い応答は実際にモデルが生成したもの、または生成しうるもの
- 良い応答は事実に基づき、簡潔で明確なもの

**実装スクリプト:**
```bash
# 追加データ収集
python scripts/evaluation/collect_additional_errors.py \
  --model outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final \
  --num_samples 100 \
  --temperature 1.2

# 手動レビューと修正
# → data/comparison/dpo_dataset_expanded.jsonl

# マージ
python scripts/evaluation/merge_dpo_datasets.py \
  --input data/comparison/dpo_dataset_20251020_151239.jsonl \
  --input data/comparison/dpo_dataset_expanded.jsonl \
  --output data/comparison/dpo_dataset_final.jsonl
```

#### タスク2: DPOトレーニングスクリプト作成

**スクリプトパス:** `scripts/training/dpo_training.py`

**実装内容:**
```python
#!/usr/bin/env python3
"""
Week 5: DPOトレーニングスクリプト

目的: 第2次ファインチューニング済みモデルに対して、
     DPO (Direct Preference Optimization) を実施し、
     事実の正確性を向上させる
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import PeftModel
import torch

# ============================================================================
# 設定
# ============================================================================

# ベースモデル（第2次ファインチューニング済み）
BASE_MODEL_NAME = "openai/gpt-oss-20b"
ADAPTER_PATH = "outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final"

# DPOデータセット
DPO_DATASET_PATH = "data/comparison/dpo_dataset_final.jsonl"

# ハイパーパラメータ
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "num_train_epochs": 3,  # 過学習防止のため少なめ
    "learning_rate": 5e-6,  # SFTより低い学習率
    "max_steps": -1,  # エポック数で制御
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.0,
    "lr_scheduler_type": "linear",
    "seed": 42,
    "output_dir": "outputs/gpt-oss-20b-takaichi-v3-dpo",
}

# DPO固有パラメータ
DPO_BETA = 0.1  # KLペナルティの強度（0.1-0.5が一般的）

# ============================================================================
# メイン処理
# ============================================================================

def main():
    print("=" * 80)
    print("Week 5: DPOトレーニング開始")
    print("=" * 80)

    # ステップ1: ベースモデル読み込み
    print("\n[1/5] ベースモデル読み込み...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # LoRAアダプター読み込み
    print(f"[2/5] LoRAアダプター読み込み: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # 新しいLoRAレイヤー追加（DPO用）
    print("[3/5] DPO用LoRAアダプター追加...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # SFTより小さいrank（微調整のため）
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ステップ2: データセット読み込み
    print(f"[4/5] DPOデータセット読み込み: {DPO_DATASET_PATH}")
    dataset = load_dataset("json", data_files=DPO_DATASET_PATH, split="train")
    print(f"  総サンプル数: {len(dataset)}")

    # ステップ3: DPOトレーナー設定
    print("[5/5] DPOトレーナー設定...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOは参照モデルを自動作成
        args=DPOConfig(**TRAINING_CONFIG),
        beta=DPO_BETA,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=2048,
        max_prompt_length=1024,
    )

    # ステップ4: トレーニング実行
    print("\n" + "=" * 80)
    print("DPOトレーニング開始")
    print("=" * 80)
    print(f"  エポック数: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"  学習率: {TRAINING_CONFIG['learning_rate']}")
    print(f"  バッチサイズ: {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  Beta (KLペナルティ): {DPO_BETA}")
    print("=" * 80 + "\n")

    trainer_stats = dpo_trainer.train()

    # ステップ5: モデル保存
    print("\n" + "=" * 80)
    print("モデル保存")
    print("=" * 80)

    final_output_dir = f"{TRAINING_CONFIG['output_dir']}/final"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"✅ モデル保存完了: {final_output_dir}")

    # 統計情報
    print("\n📊 トレーニング統計:")
    print(f"  最終Loss: {trainer_stats.training_loss:.4f}")
    print(f"  総ステップ数: {trainer_stats.global_step}")

    print("\n" + "=" * 80)
    print("🎉 Week 5: DPOトレーニング完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

**使用方法:**
```bash
source activate.sh
python scripts/training/dpo_training.py
```

#### タスク3: トレーニング実行とモニタリング

**実行手順:**
1. データセット拡充完了の確認（100ペア以上）
2. DPOスクリプト実行
3. VRAMとGPU使用率の監視
4. 損失関数の推移確認（TensorBoard推奨）

**期待される学習曲線:**
- エポック1: 急激な損失低下
- エポック2-3: 緩やかな改善
- 過学習の兆候（validation lossが上昇）があれば早期停止

#### タスク4: 評価とベースライン比較

**評価スクリプト:** `scripts/evaluation/compare_models.py`

**評価内容:**
```python
# 第2次モデル vs 第3次モデル（DPO後）の比較
evaluation_set = [
    # 基本的事実（8問）: Week 4で誤りが見つかった質問
    "高市早苗さんの生年月日は？",
    "高市早苗さんの出身地はどこですか？",
    "高市早苗さんの学歴について教えてください",
    "高市早苗さんはどの選挙区から立候補していますか？",
    "高市早苗さんが務めた大臣のポストは？",
    "高市早苗さんの初当選はいつですか？",
    "高市早苗さんは総裁選に出馬したことがありますか？",
    "高市早苗さんの政治的立場は？",

    # 政策・主張（5問）: 強みを維持できているか確認
    "高市早苗さんの経済政策について教えてください",
    "高市早苗さんの安全保障政策は？",
    # ...
]

# 評価指標
metrics = {
    "基本的事実の正確性": 0,  # 8問中何問正解か
    "応答品質": 0,  # 1-5点の平均
    "応答生成率": 0,  # 空応答の割合
}
```

**比較レポート生成:**
```bash
python scripts/evaluation/compare_models.py \
  --model_v2 outputs/gpt-oss-20b-takaichi-v2-optimized-20251020_114843/final \
  --model_v3 outputs/gpt-oss-20b-takaichi-v3-dpo/final \
  --output docs/week5_comparison_report.md
```

### 9.5 期待される改善効果

**改善前（第2次モデル）:**
- 基本的事実の誤り: 8/51 (15.7%)
- 平均評価: 4.43/5.0
- 問題: 固有名詞の不正確さ、幻覚

**改善後（第3次モデル - DPO適用後）:**
- 基本的事実の誤り: 目標 2/51以下 (4%以下)
- 平均評価: 目標 4.7/5.0以上
- 改善点:
  - ✅ 生年月日、出身地、学歴などの基本情報が正確に
  - ✅ 大臣経験、選挙歴などの政治活動情報が正確に
  - ✅ 短すぎる応答の削減
  - ✅ 幻覚の減少

### 9.6 Week 5: 実装スケジュール

**Day 1-3: データセット拡充**
- [ ] 追加の誤り応答収集（目標: 93ペア追加）
- [ ] 手動レビューと品質チェック
- [ ] カテゴリー別バランス確認
- [ ] 最終データセット作成（100-120ペア）

**Day 4-5: DPOトレーニング**
- [ ] `scripts/training/dpo_training.py` 実装
- [ ] トレーニング実行（3エポック、推定4-6時間）
- [ ] 学習曲線のモニタリング
- [ ] 中間チェックポイントの評価

**Day 6-7: 評価と検証**
- [ ] 第2次 vs 第3次モデルの比較評価
- [ ] 51質問すべてに対する再テスト
- [ ] 人間評価（可能であれば）
- [ ] Week 5完了レポート作成

**Day 8-9: Ollamaエクスポート（Phase 5の再実施）**
- [ ] VRAM不足問題の解決
- [ ] GGUF形式エクスポート（Q8_0推奨）
- [ ] Ollama Modelfile生成
- [ ] Ollamaでの推論テスト

**Day 10: 最終評価とドキュメント化**
- [ ] プロジェクト全体の振り返り
- [ ] 第1次 → 第2次 → 第3次の改善推移グラフ作成
- [ ] 今後の改善提案（Phase 10以降）

### 9.7 成功基準

**必須条件（Must Have）:**
1. ✅ 100ペア以上のDPOデータセット作成完了
2. ✅ DPOトレーニング完了（3エポック以上）
3. ✅ 基本的事実の誤り率が10%以下に改善

**期待条件（Should Have）:**
4. 🎯 基本的事実の誤り率が5%以下
5. 🎯 平均評価4.7/5.0以上
6. 🎯 Ollamaエクスポート完了

**理想条件（Nice to Have）:**
7. ⭐ 基本的事実の誤り率0%
8. ⭐ 複数人による人間評価の実施
9. ⭐ 第4次トレーニング（追加データ）の計画策定

### 9.8 リスクと対策

**リスク1: DPOデータセット拡充の労力**
- **影響度:** 高
- **対策:**
  - 自動収集ツールの活用（温度パラメータを上げた生成）
  - 段階的拡張（まず50ペア、次に100ペア）
  - 優先度付け（基本的事実を優先）

**リスク2: DPOトレーニングでの過学習**
- **影響度:** 中
- **対策:**
  - エポック数を少なめに設定（3エポック）
  - 学習率を低めに設定（5e-6）
  - 中間チェックポイントで評価

**リスク3: 改善効果が不十分**
- **影響度:** 中
- **対策:**
  - Beta値（KLペナルティ）の調整（0.1 → 0.3）
  - 追加のDPOトレーニング（エポック追加）
  - データセットの再レビューと品質向上

**リスク4: Ollamaエクスポートでのリソース不足**
- **影響度:** 低
- **対策:**
  - 段階的エクスポート（LoRAアダプターのみ → フルマージ）
  - 量子化レベルの調整（Q8_0 → q4_k_m）

### 9.9 次のフェーズへの準備（Phase 10以降）

**短期的改善（Phase 10候補）:**
1. さらなるデータセット拡充（300 → 500サンプル）
2. 第4次ファインチューニング（追加SFT）
3. マルチターン会話の改善

**中長期的拡張（将来）:**
1. RAG (Retrieval-Augmented Generation) 統合
2. 他の政治家データへの拡張
3. リアルタイム情報更新の仕組み

### 9.10 参考資料

**DPO関連:**
- [DPO: Direct Preference Optimization (論文)](https://arxiv.org/abs/2305.18290)
- [TRL (Transformer Reinforcement Learning) ドキュメント](https://huggingface.co/docs/trl)
- [Unsloth DPO Example](https://github.com/unslothai/unsloth)

**Week 4成果物（参照用）:**
- `data/comparison/model_outputs_20251020_150646.json` - モデル出力
- `data/comparison/model_outputs_20251020_150646_evaluated.json` - 評価済みデータ
- `data/comparison/dpo_dataset_20251020_151239.jsonl` - 初期DPOデータセット（7ペア）
- `docs/week4_completion_report.md` - Week 4完了レポート
