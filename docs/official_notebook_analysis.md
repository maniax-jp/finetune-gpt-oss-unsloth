# Unsloth公式Notebook分析：重要な発見と我々の設定との比較

## 📋 分析対象
- ファイル: `gpt_oss_(20B)_Fine_tuning.ipynb`
- 公式Unsloth GPT-OSS 20B ファインチューニングnotebook

---

## 🚨 重大な発見：我々が見落としていた重要な機能

### 1. **Reasoning Effort機能の存在**

**公式での説明:**
> The `gpt-oss` models from OpenAI include a feature that allows users to adjust the model's "reasoning effort."

**3つのレベル:**
- **Low**: 高速レスポンス、単純なタスク向け
- **Medium**: パフォーマンスと速度のバランス
- **High**: 最高の推論性能、高レイテンシー

**使用方法:**
```python
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",  # ← これ！
).to("cuda")
```

**我々の現状:**
```python
# 我々のコードには reasoning_effort パラメータが完全に欠落
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
```

**影響:**
- reasoning_effortを指定しないと、モデルの推論能力を引き出せない可能性
- 特に推論型タスクでパフォーマンスが低下する原因の一つ

---

### 2. **Harmony Format の特殊な構造**

**公式での説明:**
> GPT-OSS uses OpenAI Harmony format which support conversation structures, reasoning output, and tool calling.

**実際のフォーマット例（公式データセットより）:**
```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-13

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|>
<|start|>developer<|message|># Instructions

reasoning language: French

You are an AI chatbot with a lively and energetic personality.<|end|>
<|start|>user<|message|>Can you show me the latest trends on Twitter right now?<|end|>
<|start|>assistant<|channel|>analysis<|message|>D'accord, l'utilisateur demande...
[推論プロセス]
...<|end|>
<|start|>assistant<|channel|>final<|message|>Hey there! While I can't check Twitter...
[最終回答]
...<|return|>
```

**重要な構造要素:**
1. **System prompt**: モデルの設定（reasoning levelを含む）
2. **Developer message**: タスク固有の指示
3. **Analysis channel**: 推論プロセス（CoT: Chain of Thought）
4. **Final channel**: 最終回答

**我々の現状:**
```json
{"messages": [
  {"role": "user", "content": "高市早苗さんの生年月日は？"},
  {"role": "assistant", "content": "1961年3月7日生まれです。"}
]}
```

**問題点:**
- ✅ Harmonyフォーマット自体は使用（`apply_chat_template`で自動変換）
- ❌ **推論プロセス（analysis channel）が完全に欠落**
- ❌ **developer messageでの指示が欠落**
- ❌ **reasoning effortの指定が欠落**

---

### 3. **train_on_responses_only の使用**

**公式での実装:**
```python
from unsloth.chat_templates import train_on_responses_only

gpt_oss_kwargs = dict(
    instruction_part="<|start|>user<|message|>",
    response_part="<|start|>assistant<|channel|>final<|message|>"
)

trainer = train_on_responses_only(
    trainer,
    **gpt_oss_kwargs,
)
```

**目的:**
> We use Unsloth's `train_on_completions` method to **only train on the assistant outputs** and ignore the loss on the user's inputs. This helps increase accuracy of finetunes and lower loss as well!

**我々の現状:**
- この機能を使用していない
- ユーザー入力部分も含めて学習している可能性

**影響:**
- 学習効率の低下
- Lossの計算が不正確
- 精度が上がりにくい

---

### 4. **ハイパーパラメータの比較**

| パラメータ | 公式 | 我々（第4次） | 差分 |
|-----------|------|--------------|------|
| **max_seq_length** | 1024 | 2048 | 我々が2倍長い |
| **LoRA rank** | **8** | **16** | 我々が2倍大きい |
| **LoRA alpha** | 16 | 16 | 同じ ✅ |
| **LoRA dropout** | **0** | **0** | 同じ（しかし両方とも問題あり） |
| **batch_size** | **1** | **4** | 我々が4倍大きい ❌ |
| **gradient_accumulation** | 4 | 4 | 同じ ✅ |
| **learning_rate** | **2e-4** | **5e-5** | 公式が4倍高い |
| **warmup_steps** | 5 | （計算）約28 | 我々が多い |
| **weight_decay** | 0.01 | 0.01 | 同じ ✅ |

**重要な違い:**

1. **LoRA rank: 8 vs 16**
   - 公式は8（より軽量）
   - 我々は16（より複雑、過学習リスク高）

2. **Batch size: 1 vs 4**
   - 公式は1（MoE推奨）
   - 我々は4（MoEに不適切）

3. **Learning rate: 2e-4 vs 5e-5**
   - 公式は2e-4（4倍高い）
   - 我々は5e-5（保守的すぎる）

4. **max_seq_length: 1024 vs 2048**
   - 公式は1024
   - 我々は2048（不必要に長い、メモリ浪費）

---

### 5. **データセットの構造**

**公式で使用:**
- Dataset: `HuggingFaceH4/Multilingual-Thinking`
- サンプル数: 1000
- 特徴: **推論プロセス（analysis）を含む**

**データ構造:**
```python
features: [
    'reasoning_language',
    'developer',
    'user',
    'analysis',  # ← 推論プロセス！
    'final',     # ← 最終回答
    'messages'
]
```

**我々のデータセット:**
- サンプル数: 455
- 特徴: **推論プロセスが完全に欠落**
- 構造: 単純なQAのみ

---

### 6. **Float32の使用**

**公式での警告:**
```
Unsloth: Using float16 precision for gpt_oss won't work! Using float32.
```

**トレーニング時:**
```
Unsloth: Switching to float32 training since model cannot work with float16
```

**我々の現状:**
```python
dtype = None  # 自動検出に任せている
```

**確認が必要:**
- 実際にfloat32で訓練されているか？
- float16で訓練していた場合、精度問題の原因に

---

### 7. **Gradient Checkpointing**

**公式:**
```python
use_gradient_checkpointing = "unsloth"  # Unsloth最適化版
```

**我々の現状:**
- 明示的な設定なし（デフォルトに依存）

---

## 📊 総合的な問題点の整理

### レベル1：データセット構造の根本的問題（最重要）

#### 問題1-1: 推論プロセスの完全欠落
```
公式データ: User → Analysis（推論） → Final（回答）
我々のデータ: User → 回答（推論プロセスなし）
```

**影響:**
- GPT-OSSの最大の特徴である推論能力を訓練できていない
- 暗記型の学習しかできていない
- 公式推奨「75%推論データ」の意味を完全に取り違えていた可能性

#### 問題1-2: Reasoning Effortの未使用
```python
# 訓練時もテスト時も指定していない
reasoning_effort = ?  # 完全に欠落
```

**影響:**
- モデルの推論能力を引き出せていない
- 評価時のパフォーマンスが本来より低い

### レベル2：ハイパーパラメータの不適切性

#### 問題2-1: LoRA rankが大きすぎる
```
公式: 8
我々: 16（2倍）
```

#### 問題2-2: Batch sizeが大きすぎる
```
公式: 1（MoE推奨）
我々: 4（4倍）
```

#### 問題2-3: Learning rateが低すぎる
```
公式: 2e-4
我々: 5e-5（1/4）
```

### レベル3：訓練最適化の欠落

#### 問題3-1: train_on_responses_only未使用
- ユーザー入力にもLossを計算
- 学習効率が悪い

#### 問題3-2: max_seq_lengthが不必要に長い
```
公式: 1024
我々: 2048（2倍、メモリ浪費）
```

---

## 💡 改訂版：改善戦略

### 最優先アクション：データセット構造の抜本的見直し

#### 新しいデータフォーマット（Harmony準拠）

**現在（間違い）:**
```json
{"messages": [
  {"role": "user", "content": "高市早苗さんの生年月日は？"},
  {"role": "assistant", "content": "1961年3月7日生まれです。"}
]}
```

**正しいフォーマット（推論プロセス付き）:**
```json
{"messages": [
  {"role": "system", "content": "reasoning language: Japanese\n\nYou are a helpful assistant specialized in Japanese politics."},
  {"role": "user", "content": "高市早苗さんの生年月日は？"},
  {"role": "assistant", "content": "<|channel|>analysis<|message|>ユーザーは高市早苗氏の生年月日を尋ねています。これは基本的な事実確認の質問です。高市早苗氏は日本の政治家で、自由民主党所属です。生年月日は1961年3月7日です。<|end|><|channel|>final<|message|>高市早苗さんの生年月日は1961年3月7日です。"}
]}
```

**比較・分析型（推論重視）:**
```json
{"messages": [
  {"role": "system", "content": "reasoning language: Japanese\n\nYou are a helpful assistant specialized in Japanese politics with analytical capabilities."},
  {"role": "user", "content": "高市早苗さんと岸田文雄さんの経済政策の違いを分析してください。"},
  {"role": "assistant", "content": "<|channel|>analysis<|message|>この質問は二人の政治家の経済政策を比較分析することを求めています。\n\nまず、高市早苗氏の経済政策について考察します：\n- サナエノミクスとして知られる政策パッケージ\n- 積極的な財政出動を主張\n- 金融緩和の継続を重視\n- デフレ脱却を最優先課題に位置づけ\n\n次に、岸田文雄氏の経済政策：\n- 新しい資本主義を掲げる\n- 成長と分配の好循環を目指す\n- 所得倍増計画を提案\n- より穏健な財政運営\n\n両者の主な違い：\n1. 財政政策の積極性：高市氏の方がより積極的\n2. 金融政策：高市氏は緩和継続、岸田氏はより慎重\n3. 分配政策：岸田氏の方が重視\n4. 経済思想：高市氏はリフレ派、岸田氏は中道的\n\nこれらの違いは、両氏の政治的立場と支持基盤の違いを反映しています。<|end|><|channel|>final<|message|>高市早苗さんと岸田文雄さんの経済政策には以下の主な違いがあります：\n\n【高市早苗氏の経済政策】\n- サナエノミクス：積極的な財政出動と金融緩和の継続\n- デフレ脱却を最優先課題に位置づけ\n- より強力な経済刺激策を主張\n\n【岸田文雄氏の経済政策】\n- 新しい資本主義：成長と分配の好循環を重視\n- 所得倍増計画による中間層の底上げ\n- より穏健で慎重な財政運営\n\n【主な違い】\n1. 財政政策：高市氏がより積極的\n2. 分配重視度：岸田氏が強調\n3. 経済思想：高市氏はリフレ派、岸田氏は中道的\n\nこれらの違いは、両氏の政治的立場（保守派 vs 穏健派）と支持基盤の違いを反映しています。"}
]}
```

### 改訂版ハイパーパラメータ（公式準拠）

```python
# モデル設定
MAX_SEQ_LENGTH = 1024  # 2048 → 1024（公式準拠）
LOAD_IN_4BIT = True

# LoRA設定（公式準拠）
LORA_RANK = 8  # 16 → 8（公式準拠）
LORA_ALPHA = 16
LORA_DROPOUT = 0  # 公式も0だが、MoE研究では0.1推奨
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# トレーニング設定（公式準拠 + MoE最適化）
BATCH_SIZE = 1  # 4 → 1（公式準拠、MoE推奨）
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4  # 5e-5 → 2e-4（公式準拠）
NUM_TRAIN_EPOCHS = 1  # または max_steps = 60-100
WARMUP_STEPS = 5  # 公式準拠
WEIGHT_DECAY = 0.01

# 最適化設定
USE_GRADIENT_CHECKPOINTING = "unsloth"  # 公式推奨
OPTIM = "adamw_8bit"

# 評価時の設定（重要！）
REASONING_EFFORT = "medium"  # または "high"
TEMPERATURE = 1.0  # Unsloth公式ドキュメント推奨
TOP_P = 1.0
TOP_K = 0
```

### train_on_responses_onlyの実装

```python
from unsloth.chat_templates import train_on_responses_only

gpt_oss_kwargs = dict(
    instruction_part="<|start|>user<|message|>",
    response_part="<|start|>assistant<|channel|>final<|message|>"
)

trainer = train_on_responses_only(
    trainer,
    **gpt_oss_kwargs,
)
```

---

## 🎯 改訂版Phase 10実行計画

### Phase 10-1A: 公式準拠ハイパーパラメータ（最優先）

**目的:** 公式notebookと同じ設定での訓練

**変更点:**
1. max_seq_length: 1024
2. LoRA rank: 8
3. batch_size: 1
4. learning_rate: 2e-4
5. train_on_responses_only使用
6. reasoning_effort指定（評価時）

**期待効果:**
- エラー数: 7件 → 3-4件
- 公式推奨設定での基準値確立

### Phase 10-1B: 推論プロセス付きデータセット作成（並行）

**目的:** Harmony形式準拠の推論データ作成

**タスク:**
1. 既存455サンプルを50サンプルに厳選
2. 各サンプルに推論プロセス（analysis channel）追加
3. 推論型データ150サンプル新規作成
4. 合計200サンプル（推論75% / 事実25%）

**新フォーマット:**
- System message: reasoning language指定
- Analysis channel: 推論プロセス
- Final channel: 最終回答

### Phase 10-2: 推論データセットでの訓練

**目的:** 推論能力の獲得

**データセット:**
- 推論型: 150サンプル（75%）
- 事実型: 50サンプル（25%）
- 全てanalysis channelを含む

**期待効果:**
- エラー数: 3-4件 → 0-1件
- 推論能力の大幅向上
- 論理的整合性の確保

---

## 📝 結論

### 我々が見落としていた最重要事項

1. **Reasoning Effortパラメータ**
   - 存在すら知らなかった
   - 訓練時・評価時に必須

2. **推論プロセス（Analysis Channel）**
   - 「75%推論データ」の本当の意味
   - 推論プロセスを含むデータが必要だった
   - 単なる「長い回答」ではない

3. **train_on_responses_only**
   - 学習効率を大幅に改善
   - ユーザー入力の学習を避ける

4. **公式推奨ハイパーパラメータ**
   - LoRA rank: 8（我々の半分）
   - Batch size: 1（我々の1/4）
   - Learning rate: 2e-4（我々の4倍）

### 即座の行動

1. **Phase 10-1A実施**
   - 公式準拠ハイパーパラメータ
   - train_on_responses_only使用
   - reasoning_effort指定

2. **データセット抜本的見直し**
   - Analysis channel追加
   - 推論プロセスの明示
   - Harmonyフォーマット完全準拠

3. **評価方法の改善**
   - reasoning_effort="medium"または"high"
   - 推論プロセスの評価も含める
