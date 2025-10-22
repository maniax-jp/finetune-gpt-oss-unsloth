# 固有名詞認識問題の分析

## 問題の発見

第5次モデルの評価中に、重大な問題が発見されました：

**モデルが「高市早苗」を正しく認識できていない**

### 実際の出力例

```
[1/49] 高市早苗さんは何党ですか？
応答: analysisUser asks in Japanese: "高市早苗さんは何党ですか？"
      presumably asking which party San...

[3/49] 高市早苗さんの出身地はどこですか？
応答: analysisUser asks in Japanese: "高町早苗さんの出身地はどこですか？"
      They likely meant 高橋早苗 (Takashi Saori?)...

[16/49] 高市早苗さんの外交方針は？
応答: analysisThe user asks in Japanese: "高市早苗さんの外交方針は？"
      meaning "What is Kazunobu Takaichi...

[19/49] 高市早苗さんは保守派ですか？
応答: analysisThe user asks in Japanese: "High city Naozora San?
      Are they conservative...

[21/49] 高市早苗さんのデジタル政策について教えてください
応答: analysisUser asks in Japanese: "Tell me about Hiromi Takashi's digital policy?" ...
```

### 問題の深刻度

| 質問番号 | 誤った認識 | 正解 |
|---------|-----------|------|
| 1 | "San..." | Sanae Takaichi |
| 3 | "高橋早苗 (Takashi Saori)" | 高市早苗 (Sanae Takaichi) |
| 16 | "Kazunobu Takaichi" | Sanae Takaichi |
| 19 | "High city Naozora San" | Sanae Takaichi |
| 21 | "Hiromi Takashi" | Sanae Takaichi |

**エラー率: ほぼ100%** - 1つも正しく認識できていない

## 根本原因

### 原因1: Analysis Channelでの英語思考

公式ノートブックの推論プロセスは英語で思考します：

```json
{
  "role": "assistant",
  "content": "<|channel|>analysis<|message|>User asks about Sanae Takaichi...<|end|>"
             "<|channel|>final<|message|>高市早苗さんは自由民主党です。"
}
```

しかし、現在のデータセットには：
- ❌ 英語での固有名詞表記がない
- ❌ 「高市早苗 = Sanae Takaichi」という対応関係がない
- ❌ 推論プロセス自体が存在しない（100%非推論データ）

### 原因2: データセットの構造

現在のデータセット（455サンプル）:
```json
{
  "messages": [
    {"role": "user", "content": "高市早苗さんは何党ですか？"},
    {"role": "assistant", "content": "自由民主党です。"}
  ]
}
```

公式推奨のデータセット:
```json
{
  "messages": [
    {"role": "user", "content": "高市早苗さんは何党ですか？"},
    {"role": "assistant", "content":
      "<|channel|>analysis<|message|>"
      "User asks about Sanae Takaichi's political party. "
      "She is a member of the Liberal Democratic Party (LDP)."
      "<|end|>"
      "<|channel|>final<|message|>"
      "高市早苗さんは自由民主党（LDP）に所属しています。"
    }
  ]
}
```

### 原因3: トークナイザーの問題

日本語の漢字「高市早苗」は、トークナイザーにとって：
- 複数のサブワードトークンに分割される
- 英語表記との対応関係が学習されていない
- ベースモデルでも正しく認識できていない可能性

## 影響範囲

### 直接的影響

1. **質問理解の失敗**
   - モデルが質問の主語を理解できない
   - 誤った人物について回答する可能性

2. **回答品質の低下**
   - 固有名詞を認識できないため、関連情報を引き出せない
   - データセットに「高市早苗」で学習した内容と紐付かない

3. **推論能力の欠如**
   - Analysis channelで正しく思考できない
   - 英語と日本語の橋渡しができない

### 副次的影響

1. **他の固有名詞への波及**
   - 「岸田文雄」→ "Fumio Kishida"
   - 「安倍晋三」→ "Shinzo Abe"
   - 「自由民主党」→ "Liberal Democratic Party"
   - これらも同様の問題を抱える可能性

2. **専門用語の認識**
   - 「サナエノミクス」→ "Sanaenomics"
   - 「経済安全保障」→ "Economic Security"
   - 「デジタル田園都市国家構想」→ "Digital Garden City Nation"

## 解決策

### 短期的対策（Phase 10-1A）

**1. 固有名詞認識用のプリアンブル追加**

すべての推論データの冒頭に、固有名詞の対応表を追加：

```json
{
  "role": "assistant",
  "content":
    "<|channel|>analysis<|message|>"
    "Key entities: 高市早苗 (Sanae Takaichi), Japanese politician, LDP member. "
    "User asks about Sanae Takaichi's political party affiliation. "
    "She is a member of the Liberal Democratic Party (自由民主党, LDP)."
    "<|end|>"
    "<|channel|>final<|message|>"
    "高市早苗さんは自由民主党（LDP）に所属しています。"
}
```

**2. 固有名詞辞書の作成**

主要な固有名詞の日英対応表：
- 人名: 高市早苗、岸田文雄、安倍晋三、等
- 組織名: 自由民主党、経済安全保障推進室、等
- 政策名: サナエノミクス、デジタル田園都市国家構想、等
- 地名: 奈良県第2区、等

**3. システムプロンプトでの明示**

```json
{
  "role": "system",
  "content":
    "You are an expert on Japanese politician Sanae Takaichi (高市早苗). "
    "When analyzing questions, always recognize:\n"
    "- 高市早苗 = Sanae Takaichi (politician)\n"
    "- 自由民主党 = Liberal Democratic Party (LDP)\n"
    "- サナエノミクス = Sanaenomics (economic policy)"
}
```

### 中期的対策（Phase 10-2）

**1. 推論データの段階的追加**

ステップ1: 固有名詞認識特化データ（50サンプル）
- 「高市早苗とは誰ですか？」
- 「Sanae Takaichiについて教えてください」
- 「高市早苗さんの英語表記は？」

ステップ2: 基本情報の推論データ（100サンプル）
- Analysis channelで固有名詞を明示
- 英語と日本語の両方で説明

ステップ3: 複雑な推論データ（150サンプル）
- 比較分析、時系列分析、政策分析
- 複数の固有名詞を含む質問

**2. データセット比率の段階的変更**

| フェーズ | 推論データ | 非推論データ | 合計 |
|---------|-----------|-------------|------|
| 現在 | 0 (0%) | 455 (100%) | 455 |
| Phase 10-2-1 | 50 (10%) | 455 (90%) | 505 |
| Phase 10-2-2 | 150 (25%) | 455 (75%) | 605 |
| Phase 10-2-3 | 300 (50%) | 455 (50%) | 755 |
| 最終目標 | 1050 (75%) | 350 (25%) | 1400 |

### 長期的対策（Phase 10-3以降）

**1. 多言語対応の強化**

- 日本語↔英語の固有名詞対応を完全学習
- 中国語・韓国語の表記も含める（東アジア政治の文脈）

**2. 固有名詞エンティティの体系化**

- 人名データベース（政治家100名）
- 組織名データベース（政党、省庁、委員会）
- 政策名データベース（アベノミクス、サナエノミクス等）
- 地名データベース（選挙区、出身地）

**3. DPO（Direct Preference Optimization）での修正**

誤った固有名詞認識のペアを作成：

```json
{
  "prompt": "高市早苗さんは何党ですか？",
  "chosen": "<|channel|>analysis<|message|>User asks about Sanae Takaichi's party...",
  "rejected": "<|channel|>analysis<|message|>User asks about High City Shizuka..."
}
```

## 検証方法

### テスト項目

1. **固有名詞認識テスト**
   - 「高市早苗」を含む質問10問
   - 正しく "Sanae Takaichi" と認識できるか

2. **他の固有名詞テスト**
   - 「岸田文雄」「安倍晋三」等の認識率
   - 組織名・政策名の認識率

3. **推論品質テスト**
   - Analysis channelで正しく思考できるか
   - 英語と日本語の整合性

### 成功基準

| 項目 | 現状 | 目標（Phase 10-2） |
|-----|------|-------------------|
| 固有名詞認識率 | 0% | 95%以上 |
| 推論プロセスの正確性 | N/A | 80%以上 |
| 最終回答の正確性 | 85.7% (v3) | 95%以上 |

## まとめ

### 重要な発見

1. **公式ノートブックの推論プロセスは英語で思考する**
   - これは日本語QAには想定外の課題
   - 固有名詞の日英対応が必須

2. **現在のデータセットは推論学習に不十分**
   - 100%非推論データ
   - 固有名詞の英語表記がゼロ

3. **第5次モデルの高いLoss (7.79)の原因**
   - 単なる過学習抑制だけでなく
   - データ形式のミスマッチも影響

### 次のアクション

**優先度1: 固有名詞認識データの作成**
1. 高市早苗の基本情報50サンプル
2. Analysis channelに固有名詞明示
3. 日英対応を明確に学習

**優先度2: 第5次モデルの再評価**
1. 固有名詞認識テストの実施
2. 誤認識パターンの詳細分析
3. 改善策の効果検証

**優先度3: Phase 10-2への移行判断**
1. Phase 10-1（ハイパーパラメータ最適化）の効果検証
2. Phase 10-2（推論データ追加）の詳細計画
3. 固有名詞認識を含めた学習戦略の策定

---

**作成日**: 2025-10-22
**関連ドキュメント**:
- [docs/root_cause_analysis.md](root_cause_analysis.md)
- [docs/official_notebook_analysis.md](official_notebook_analysis.md)
- [docs/unsloth_official_findings.md](unsloth_official_findings.md)
