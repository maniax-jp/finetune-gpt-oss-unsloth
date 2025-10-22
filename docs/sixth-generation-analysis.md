# 第6次ファインチューニング問題分析レポート

**日時**: 2025-10-22 15:21
**対象**: 第6次モデル (`gpt-oss-20b-takaichi-v6-proper-nouns-20251022_114738`)
**検証者**: Claude Code

---

## 1. 検証結果サマリー

### 固有名詞認識率

| モデル | 「高市早苗」認識率 | 改善幅 |
|--------|-------------------|--------|
| 第5次 | 0.0% (0/9) | - |
| 第6次 | **0.0% (0/5)** | **+0.0%** |

**結論**: 第6次モデルは固有名詞認識率において第5次モデルから**改善なし**。

---

## 2. 問題の根本原因

### 2.1 訓練データの構成

第6次モデルの訓練データ ([takaichi_sanae_mixed_v6.jsonl](../dataset/takaichi_sanae_mixed_v6.jsonl)):

| データ種別 | サンプル数 | 比率 | analysisチャンネル |
|-----------|-----------|------|-------------------|
| 推論データ | 50 | 9.9% | ✓ あり |
| 非推論データ | 455 | 90.1% | ✗ なし |
| **合計** | **505** | **100%** | - |

### 2.2 訓練データの詳細

#### 推論データ（1-50行目）

```json
{
  "messages": [
    {"role": "user", "content": "高市早苗さんは何党ですか？"},
    {"role": "assistant", "content": "<|channel|>analysis<|message|>Key entity: 高市早苗 (Sanae Takaichi) - Japanese politician, LDP member.\n\nUser asks in Japanese: \"What party does Sanae Takaichi belong to?\"\n\nSanae Takaichi is a member of the Liberal Democratic Party...<|end|><|channel|>final<|message|>高市早苗さんは自由民主党（LDP）に所属しています。"}
  ]
}
```

✓ **analysisチャンネルあり**
✓ **固有名詞の日英対応明記**: `高市早苗 (Sanae Takaichi)`

#### 非推論データ（51-505行目）

```json
{
  "messages": [
    {"role": "user", "content": "高市早苗さんの出身大学は？"},
    {"role": "assistant", "content": "神戸大学経営学部を卒業しています。"}
  ]
}
```

✗ **analysisチャンネルなし**
✗ **固有名詞の日英対応なし**

---

## 3. 問題点の特定

### 3.1 データ比率の不均衡

- **推論データ**: 9.9% (50/505)
- **非推論データ**: 90.1% (455/505)

**問題**: 非推論データが圧倒的に多いため、モデルは「analysisチャンネルを生成しない」パターンを優先的に学習。

### 3.2 学習アルゴリズムの影響

**使用された設定**:
- `max_steps=60`（早期停止）
- `batch_size=1`, `gradient_accumulation_steps=4`（実効バッチサイズ=4）
- 推論データの有効ステップ数: `50 / 4 = 12.5ステップ`

**結果**: 60ステップ中、推論データが学習されるのは約12-13ステップのみ（約20%）。

### 3.3 train_on_responses_only の影響

```python
dataset = trainer.train(
    ...,
    train_on_responses_only=True,  # アシスタントの出力のみでloss計算
)
```

- 非推論データ（455サンプル）: アシスタントの出力は短い（平均50-100トークン）
- 推論データ（50サンプル）: アシスタントの出力は長い（平均200-400トークン）

**問題**: 非推論データは短いため、loss計算に占める割合が小さく、モデルは非推論データのパターンを優先的に学習。

---

## 4. モデルの実際の挙動

### 4.1 第6次モデルの出力例

**質問**: 「高市早苗さんは何党ですか？」

**モデル出力**:

```
<|channel|>analysis<|message|>The user asks in Japanese: "高市早苗さんは何党ですか？" They want to know which party Naoto Takahashi? Actually "高市早苗" is a Japanese politician, Naoko Takahashi? Wait, "高市早苗" is a Japanese politician, a member of the Liberal Democratic Party? Let's recall: Takahashi Sai...
```

**分析**:
- ✓ analysisチャンネルを生成している
- ✗ **「高市早苗」を "Naoto Takahashi", "Naoko Takahashi", "Takahashi Sai" と誤認識**
- ✗ **"Sanae Takaichi" という正しい英語名を出力していない**

### 4.2 第5次モデルとの比較

| モデル | analysisチャンネル生成 | 「高市早苗」の英語名 |
|--------|----------------------|---------------------|
| 第5次 | ✓ 生成する | "Takashi S", "Takahashi Sanae" (誤り) |
| 第6次 | ✓ 生成する | "Naoto Takahashi", "Naoko Takahashi" (誤り) |

**共通点**: 両モデルとも「高市早苗」の正しい英語名 "Sanae Takaichi" を学習していない。

---

## 5. 根本原因のまとめ

第6次モデルが固有名詞認識に失敗した理由:

1. **データ比率の不均衡**: 推論データ 9.9% vs 非推論データ 90.1%
2. **学習ステップ数の不足**: 推論データが学習されるのは60ステップ中12-13ステップのみ
3. **実効的な学習量の不足**: `train_on_responses_only=True` により、短い非推論データが優先的に学習される
4. **データの混合方法**: 推論データを先頭50行に配置したが、ランダムシャッフルされていない可能性

---

## 6. 改善策の提案

### Phase 10-2-B: 推論データ比率の増加

#### 案1: 推論データを50%に増加

- 推論データ: 250サンプル（50% of 500）
- 非推論データ: 250サンプル（50% of 500）
- 合計: 500サンプル

**メリット**:
- 推論データの学習ステップ数が増加（12 → 30ステップ）
- 固有名詞の日英対応が学習されやすい

**デメリット**:
- 推論データの作成に時間がかかる（200サンプル追加作成）

#### 案2: 非推論データにもanalysisチャンネルを追加

- 既存の455サンプルに簡易的なanalysisチャンネルを自動付与
- 固有名詞の日英対応を明記

**メリット**:
- 全データに固有名詞の日英対応が含まれる
- データ作成の手間が少ない

**デメリット**:
- 自動生成されたanalysisの品質が不明

#### 案3: 学習ステップ数の増加

- `max_steps=60` → `max_steps=150`
- 推論データの学習ステップ数が増加

**メリット**:
- データを変更せずに学習量を増やせる

**デメリット**:
- 過学習のリスクが増加
- 学習時間が2.5倍に増加

---

## 7. 次のアクション

### 推奨アクション: **案2（非推論データにanalysisチャンネル自動付与）**

**理由**:
1. データ作成の手間が最小（自動生成）
2. 全データに固有名詞の日英対応を含められる
3. 学習時間は変わらない

**実装手順**:
1. 非推論データ（455サンプル）に簡易analysisチャンネルを自動付与するスクリプトを作成
2. 固有名詞辞書を使って日英対応を自動挿入
3. 第7次ファインチューニングを実行
4. 固有名詞認識率を検証

---

## 8. 検証データ

- [quick_validation_20251022_152053.json](../logs/quick_validation_20251022_152053.json)
- [takaichi_sanae_mixed_v6.jsonl](../dataset/takaichi_sanae_mixed_v6.jsonl)
- [create_reasoning_data_v2.py](../scripts/data_collection/create_reasoning_data_v2.py)
- [merge_reasoning_nonreasoning.py](../scripts/data_collection/merge_reasoning_nonreasoning.py)

---

## 9. 結論

第6次モデルは、訓練データに固有名詞の日英対応を含めたにもかかわらず、**データ比率の不均衡**と**学習ステップ数の不足**により、固有名詞認識率が0%のまま改善しなかった。

次のフェーズ（Phase 10-2-B）では、**全訓練データにanalysisチャンネルと固有名詞の日英対応を含める**ことで、この問題を解決する必要がある。

---

**作成日時**: 2025-10-22 15:21
**次回更新予定**: Phase 10-2-B実施後
