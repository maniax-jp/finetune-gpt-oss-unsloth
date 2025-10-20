# Claude Code プロジェクト設定

## 最重要事項

**応答言語: 必ず日本語で応答してください**

すべての会話、説明、コメント、ドキュメントは日本語で行うこと。

## プロジェクト概要

このプロジェクトは、GPT-OSS 20Bモデルのファインチューニングプロジェクトです。
高市早苗氏に関するQAデータセットを収集し、チャットボット向けにモデルを最適化します。

## 現在のフェーズ

Phase 2: データ収集（本格収集）
- 目標: 300-500サンプル

## ディレクトリ構造

```
finetune-gpt-oss-unsloth/
├── scripts/
│   ├── data_collection/     # データ収集スクリプト
│   ├── training/            # 学習スクリプト
│   └── evaluation/          # 評価スクリプト
├── data/
│   ├── raw/                 # 元データ
│   ├── processed/           # 処理済みデータ
│   └── metadata/            # メタデータ
├── exported_models/         # エクスポート済みモデル
├── docs/                    # ドキュメント
└── logs/                    # ログファイル
```

## 重要ファイル

- `development-plan.md`: 開発計画書
- `docs/data-collection-plan.md`: データ収集計画書
- `data/processed/merged_collection.json`: 収集済みQAデータ

## 注意事項

- Python仮想環境: `.venv` を使用
- すべてのPythonスクリプトは `source activate.sh` 後に実行
- Git操作は慎重に（main branchで作業中）
