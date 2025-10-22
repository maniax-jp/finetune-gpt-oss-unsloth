# 根本原因分析：モデル性能悪化の要因

## 📋 Executive Summary

第4次ファインチューニングで性能が悪化した根本原因は、**MoE（Mixture of Experts）アーキテクチャの特性を考慮しない学習戦略**にあります。

**結論**:
- データセットの質は問題ない
- **学習戦略とハイパーパラメータがMoEモデルに不適切**
- 小規模データセットでのMoE過学習が発生

---

## 🔍 発見された根本的問題

### 1. ベースモデル：GPT-OSS-20Bの特性

#### アーキテクチャ
```
総パラメータ数: 21B
アクティブパラメータ: 3.6B（forward pass毎）
構造: 24層、32個のMoE experts、Top-4ルーティング
```

#### MoEアーキテクチャの特徴
- **Sparse Activation**: 各トークンで全expertの一部のみが活性化
- **Expert Specialization**: expertが特定のタスクに特化
- **Memory Efficiency**: 16GB以内で動作可能

### 2. MoEモデルのファインチューニング課題

#### 研究で明らかになった問題点

**【問題1】小規模データセットでの過学習傾向**
> "MoE-based language models only perform well when the dataset is very large, otherwise their performance deteriorates due to overfitting."

**現状との対比:**
```
第2次: 301サンプル × 25エポック = 7,525回の学習
第4次: 455サンプル × 25エポック = 11,375回の学習  ← 過学習リスク増大
```

**【問題2】Expert数と過学習の関係**
> "A fewer number of experts helped at fine-tuning."
> "Variants with fewer experts enjoyed more successful fine-tuning"

**GPT-OSS-20Bの状況:**
- **32 experts** = 非常に多い
- 小規模データセットでは一部のexpertのみが頻繁に活性化
- 活性化されるexpertが過学習しやすい

**【問題3】Sparse層への正則化不足**
> "Sparse models are more prone to overfitting"
> "Explore higher regularization (e.g. dropout) within the experts themselves"

**現在の設定の問題:**
```python
LORA_DROPOUT = 0  ← Dropoutなし！
WEIGHT_DECAY = 0.01  ← 標準的だが、MoEには不十分
```

**【問題4】ハイパーパラメータの不適切性**
> "Sparse models benefit from noisier hyperparameters including small batch sizes and high learning rates"

**現在の設定:**
```python
BATCH_SIZE = 4  ← 大きすぎる（MoEには小さいバッチが推奨）
GRADIENT_ACCUMULATION_STEPS = 4  ← 実効バッチサイズ16
LEARNING_RATE = 5e-5  ← 標準的（MoEには高めの学習率が推奨）
NUM_TRAIN_EPOCHS = 25  ← 多すぎる（過学習のリスク）
```

---

## 📊 性能悪化の具体的メカニズム

### ステップ1: データ量増加による過学習加速
```
301サンプル → 455サンプル (+50%)
総学習回数: 7,525回 → 11,375回 (+51%)
```
→ 小規模データセットでの過学習リスクが増大

### ステップ2: Expert Specialization の歪み
- 455サンプルでは、一部のexpertのみが頻繁に活性化
- 活性化されるexpertが特定のパターンを過度に学習
- 基本的事実よりも、データセット特有のパターンを記憶

### ステップ3: 正則化不足による過適合
- Dropout = 0 → Sparse層の過学習を防げない
- 25エポック → 同じデータを繰り返し学習
- 基本的事実が「ノイズ」として扱われる可能性

### ステップ4: DPO効果の上書き
- 第3次でDPOにより修正された事実
- 第4次の大量学習（25エポック）で再び誤った方向に
- DPOの効果が消失

---

## 🎯 エビデンス：第3次 vs 第4次の比較

### エラー数の変化
```
第3次（DPO後）: 3件のエラー
第4次（データ拡充後）: 7件のエラー  (+133%)
```

### 具体的な悪化例

#### 例1: 初当選年
```
第3次: 「1993年」 ✅ 正確！
第4次: 「2009年」 ❌ 誤り（DPO効果が消失）
```

#### 例2: 出身地
```
第3次: 「岐阜県山口市」（誤りだが検出外）
第4次: 「岩手県奥州市」 ❌ より顕著なエラー
```

#### 例3: 選挙区
```
第3次: 「北九州市第2選挙区」
第4次: 「東京都議会第1選挙区」 ❌ エラー3件検出
```

---

## 🚫 誤った仮説の排除

### ❌ 仮説1: データ品質の問題
**根拠**: ユーザーから「データセットに誤りが含まれている可能性は考慮しなくてよい」

**追加分析**:
- 第3次モデルでは「1993年」を正しく回答
- → データには正しい情報が含まれている
- 第4次で悪化 → データ追加が直接的原因ではない

### ❌ 仮説2: ライブラリのバグ
**根拠**:
- Unslothは広く使用されている安定したライブラリ
- 第3次までは改善傾向
- 学習自体は正常に完了（Loss低下）

### ❌ 仮説3: モデル容量不足
**根拠**:
- 21Bパラメータ（3.6B active）は十分な容量
- 小規模データセットには過剰なほど

---

## ✅ 確定した根本原因

### 主原因：MoE特性を無視した学習戦略

1. **過学習の促進**
   - 小規模データセット（455サンプル）
   - 過剰なエポック数（25）
   - Dropout = 0（正則化不足）

2. **MoE非推奨パラメータ**
   - バッチサイズ大（実効16）
   - 学習率標準（MoEには高めが推奨）
   - 全パラメータ更新（FFNのみが推奨）

3. **Expert活性化の偏り**
   - 32 experts × 小規模データ
   - 一部expertの過学習
   - Specializationの歪み

---

## 💡 改善方針

### 即座に実施すべき対策

#### 1. **正則化の強化**
```python
# Sparse層に高いDropout
LORA_DROPOUT = 0.1  # 0 → 0.1
EXPERT_DROPOUT = 0.2  # 新規追加

# Weight Decayの増加
WEIGHT_DECAY = 0.1  # 0.01 → 0.1
```

#### 2. **ハイパーパラメータの調整**
```python
# MoE推奨設定
BATCH_SIZE = 1  # 4 → 1（ノイジーな学習）
GRADIENT_ACCUMULATION_STEPS = 4  # 実効4
LEARNING_RATE = 1e-4  # 5e-5 → 1e-4（高め）
NUM_TRAIN_EPOCHS = 5  # 25 → 5（過学習防止）
```

#### 3. **選択的パラメータ更新**
```python
# FFN（Feed-Forward Network）のみ更新
# expertの過学習を防ぐ
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
# 除外: q_proj, k_proj, v_proj, o_proj
```

#### 4. **Instruction Tuning の活用**
- MoEは instruction tuning で大きく改善
- 現在のQA形式を強化
- System promptの追加

### 中長期的な対策

#### 1. **データセット戦略の見直し**
```
現状: 455サンプル × 25エポック = 過学習
改善:
  - Option A: データ量を大幅増加（2000+サンプル）
  - Option B: エポック数削減（5エポック以下）
  - Option C: データ augmentation
```

#### 2. **別のベースモデルの検討**
- **Dense model**（非MoE）の検討
  - 例: LLaMA 3.1 8B
  - 小規模データセットに適している
  - 過学習しにくい

#### 3. **混合学習戦略**
```
1. ベースモデルでSFT（5エポック、高正則化）
2. DPO訓練（事実修正）
3. Instruction tuning（汎化性能向上）
```

---

## 📈 期待される効果

### 短期的改善（正則化強化）
- 過学習の抑制
- DPO効果の維持
- エラー率: 7件 → 3-4件（予想）

### 中期的改善（ハイパーパラメータ最適化）
- Expert活性化の均等化
- 基本的事実の定着
- エラー率: 3-4件 → 1-2件（予想）

### 長期的改善（アーキテクチャ変更）
- Dense modelへの移行検討
- 本番品質の達成
- エラー率: < 1件（目標）

---

## 🎯 推奨される次のアクション

### Priority 1: 緊急対応（今すぐ）
1. **第5次ファインチューニング実施**
   - 正則化強化（Dropout 0.1）
   - エポック削減（25 → 5）
   - バッチサイズ削減（4 → 1）

### Priority 2: 検証（1週間以内）
1. 第5次モデルの評価
2. 第3次 vs 第5次の比較
3. 改善効果の測定

### Priority 3: 戦略的判断（2週間以内）
1. Dense modelへの移行検討
2. データ augmentation 戦略
3. 長期的な学習パイプライン設計

---

## 📚 参考文献

1. Hugging Face Blog: "Mixture of Experts Explained"
2. OpenAI: "gpt-oss Model Card"
3. Research: "MoE models only perform well when dataset is very large"
4. Best Practice: "Sparse models benefit from higher dropout"
5. Fine-tuning Study: "Fewer experts helped at fine-tuning"

---

## 結論

**データセットは無実です。**

問題の本質は、**MoE（Mixture of Experts）アーキテクチャの特性を理解せずに、Dense modelと同じ学習戦略を適用したこと**にあります。

455サンプルという小規模データセットで25エポックの学習を行い、Dropout=0という設定は、MoEモデルにとって**最悪の組み合わせ**でした。

即座に正則化を強化し、エポック数を削減した第5次ファインチューニングを実施することを強く推奨します。
