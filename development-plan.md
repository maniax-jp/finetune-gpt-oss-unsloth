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
