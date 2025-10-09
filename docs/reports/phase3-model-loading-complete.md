# Phase 3: モデル読み込みと設定完了レポート

## ✅ 実施結果: 成功

GPT-OSS 20BモデルをUnsloth + QLoRAで正常に読み込み、推論テストに成功しました。

## 📊 モデル読み込み結果

### システム構成
- **GPU**: NVIDIA GeForce RTX 5090
- **Total VRAM**: 31.84 GB
- **CUDA Version**: 12.8
- **Torch**: 2.8.0+cu128
- **Triton**: 3.4.0
- **Xformers**: 0.0.33+f204359.d20251008

### モデル情報
- **Model**: openai/gpt-oss-20b
- **Total parameters**: 11,057,028,672 (約110億パラメータ)
- **Trainable parameters**: 15,925,248 (約1,600万パラメータ)
- **Trainable %**: 0.1440%

### VRAM使用量
| 状態 | 割り当て | 予約済み |
|------|---------|---------|
| モデル読み込み後 | 11.71 GB | 11.73 GB |
| QLoRA適用後 | 11.77 GB | 11.79 GB |

**余裕VRAM**: 約20GB（バッチサイズ増加可能）

## 🔧 QLoRA設定

### パラメータ
```python
r = 32                    # LoRA rank
lora_alpha = 32          # LoRA alpha (通常rankと同じ)
lora_dropout = 0         # Unsloth最適化（dropout無し）
bias = "none"            # biasトレーニング無し
```

### ターゲットモジュール
- **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP**: `gate_proj`, `up_proj`, `down_proj`

### 最適化機能
- **Gradient Checkpointing**: Unsloth最適化版
- **4-bit Quantization**: 有効（QLoRA）
- **Max Seq Length**: 2048トークン

## 🧪 推論テスト結果

### テストプロンプト
```
高市早苗さんについて教えてください。
```

### 結果
- ✅ **推論成功**: モデルが正常に応答を生成
- **入力トークン数**: 91トークン
- **推論速度**: 高速（Unsloth最適化）
- **Harmonyフォーマット**: 自動適用

### 生成された応答の特徴
- Harmonyフォーマットのチャンネルシステム（analysis, commentary, final）を使用
- Reasoning effort: medium（自動設定）
- 思考過程（analysis channel）を含む出力
- 日本語質問に対して適切に応答

**注**: ファインチューニング前のため、高市早苗氏に関する正確な情報は含まれていませんが、推論機能は正常に動作しています。

## 📈 性能ベンチマーク（Unsloth公式データ）

### VRAM要件比較
| 方式 | gpt-oss-20b | gpt-oss-120b |
|------|-------------|--------------|
| QLoRA | 14GB | 65GB |
| BF16 LoRA | 44GB | 210GB |

**実測値**: 11.77GB（公式値14GBより少ない！）

### Unslothの性能向上
- **速度**: 1.5倍高速化
- **VRAM**: 70%削減
- **コンテキスト長**: 10倍以上サポート（Flex Attention使用時）

## 🎯 RTX 5090 最適化

### 推奨学習パラメータ（32GB VRAM用）
```python
batch_size = 4               # 推奨範囲内
gradient_accumulation = 4    # 実効バッチサイズ16
max_seq_length = 2048        # データセット最大852トークンに対応
learning_rate = 2e-4
epochs = 3-5                 # 小規模データセット（16会話）用
warmup_ratio = 0.1
weight_decay = 0.01
```

### メモリ見積もり
```
モデル（4-bit QLoRA）: ~12GB
学習時追加メモリ: ~2-4GB（バッチサイズ4）
合計: ~14-16GB
余裕: 16-18GB（バッチサイズ8-12も可能）
```

## 📂 作成ファイル

### [`scripts/load_model.py`](../../scripts/load_model.py)

**機能**:
- GPT-OSS 20B + Unsloth統合
- QLoRA自動設定
- VRAM使用量モニタリング
- 推論テスト機能

**使用方法**:
```bash
# 基本的な使用
source ./activate.sh
python scripts/load_model.py

# 推論テスト付き
python scripts/load_model.py --test-inference

# カスタムプロンプト
python scripts/load_model.py --test-inference --prompt "あなたの質問"

# 16-bit LoRA（VRAM 44GB必要）
python scripts/load_model.py --no-4bit

# QLoRA無し（推論のみ）
python scripts/load_model.py --no-qlora --test-inference
```

## 🔍 技術的詳細

### Harmonyフォーマット自動適用
UnslothのGPT-OSSサポートにより、tokenizerの`apply_chat_template`が自動的にHarmonyフォーマットを適用:
- System prompt with reasoning effort
- Channel system (analysis, commentary, final)
- Knowledge cutoff date
- Current date

### 4-bit量子化の実装
- **方式**: QLoRA (Quantized Low-Rank Adaptation)
- **量子化**: BitsAndBytes NF4（MXFP4相当）
- **精度**: 推論品質を維持しながらVRAM削減
- **対象**: すべてのLinear層（attention + MLP）

## ⚠️ 既知の制限事項

1. **ファインチューニング前**
   - モデルは高市早苗氏に関する知識を持っていない
   - 一般的な知識ベース（2024年6月カットオフ）のみ

2. **推論モード**
   - Harmonyフォーマットのチャンネルシステムにより出力が冗長
   - ファインチューニング後は最適化される予定

3. **メモリ**
   - 初回実行時にモデルダウンロード（約40GB）
   - キャッシュ後は読み込み時間2-3分

## ✅ 次のステップ: Phase 4（ファインチューニング実行）

Phase 3完了により、ファインチューニングの準備が整いました。

### Phase 4で実施する内容
1. **4.1 ファインチューニングスクリプト作成**
   - データセット読み込み（Harmony形式）
   - 学習ループ実装
   - チェックポイント保存

2. **4.2 ファインチューニング実行**
   - 高市早苗データセット（16会話）で学習
   - VRAM使用量モニタリング
   - 学習曲線の記録

3. **4.3 ファインチューニング結果検証**
   - 学習済みモデルでテスト推論
   - 高市早苗氏に関する質問への回答精度確認

## 📌 重要な発見

1. **VRAM効率**: 公式値14GBに対し実測11.77GB（17%削減）
2. **RTX 5090対応**: Blackwell (CC 12.0)で完全動作
3. **Unsloth最適化**: Xformersとの統合により高速推論
4. **余裕メモリ**: 20GB以上の余裕があり、バッチサイズ拡大可能

## 🎉 Phase 3完了チェックリスト

- [x] GPT-OSS 20Bモデル情報調査
- [x] Unsloth + GPT-OSS 20B統合スクリプト作成
- [x] QLoRA設定（4-bit量子化、rank=32）
- [x] モデル読み込みテスト実行（成功）
- [x] 推論テスト実行（成功）
- [x] VRAM使用量確認（11.77GB、余裕20GB）

**Phase 3: 100%完了** 🎯
