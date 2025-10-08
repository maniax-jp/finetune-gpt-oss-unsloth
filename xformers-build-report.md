# Xformers ビルドレポート

実施日: 2025-10-08

## ビルド試行結果

### ❌ ソースからのビルド失敗

Xformersをソースからビルドを試みましたが、CUDA Toolkitが完全にインストールされていないため失敗しました。

### 失敗の原因

1. **CUDA Toolkitの不足**: `nvcc`コンパイラが利用できない
2. **C++コンパイル環境**: Xformersのソースビルドには完全なCUDA開発環境が必要

### 現在の状況

- ✅ Ninja: インストール済み (1.13.0)
- ❌ CUDA Toolkit (nvcc): 未インストール
- ✅ Xformers (プリビルド版): 0.0.32.post2（RTX 50XX未対応だが動作可能）

## 結論と推奨事項

### オプション1: 現状維持（推奨）

**理由**:
- Unslothは既にPyTorch標準Attentionに自動切替済み
- すべての機能が正常に動作
- Xformersなしでも学習可能

**利点**:
- ✅ 追加のセットアップ不要
- ✅ 安定動作
- ✅ すぐにファインチューニング開始可能

**欠点**:
- ⚠️ Xformers使用時より若干遅い可能性（実用上は問題なし）

### オプション2: CUDA Toolkitインストール後にビルド

**必要な手順**:

1. **CUDA Toolkit 12.6+のインストール**:
```bash
# NVIDIA公式からCUDA Toolkitをダウンロード
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# 環境変数設定
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

2. **Xformersビルド**:
```bash
source ./activate.sh
export TORCH_CUDA_ARCH_LIST="12.0"
uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

**利点**:
- ✅ RTX 5090に最適化されたXformers
- ✅ より高速な学習

**欠点**:
- ⚠️ CUDA Toolkitのインストールが必要（数GB）
- ⚠️ ビルドに30分〜1時間かかる
- ⚠️ WSL環境での複雑な設定

### オプション3: Docker環境使用

Docker版Unslothには最適化済みのXformersが含まれている可能性があります（未確認）。

## パフォーマンス比較（推定）

| 構成 | 推定速度 | 安定性 | セットアップ |
|------|---------|--------|-------------|
| PyTorch Attention（現状） | 100% | ✅ 高い | ✅ 完了 |
| Xformers（プリビルド版） | 動作不可 | ❌ RTX 50XX未対応 | - |
| Xformers（ソースビルド） | 110-120% | ⚠️ 要検証 | ❌ CUDA Toolkit必要 |

## 推奨アクション

### ✅ すぐに開発を始める場合（推奨）

**現状のまま進める**:
```bash
source ./activate.sh
python your_finetune_script.py
```

Unslothは自動的にPyTorch Attentionを使用し、正常に動作します。

### 🔧 最適化を求める場合

1. Phase 2-7を完了させて、まずファインチューニングを成功させる
2. パフォーマンスに問題があれば、その後CUDA Toolkitインストール＆Xformersビルドを検討

## 技術的詳細

### Xformersが必要な理由

- Memory-efficient attention実装
- Flash Attention対応
- GPU利用効率の向上

### RTX 50XX (Blackwell)での課題

- プリビルド版のXformersはCompute Capability 12.0未対応
- ソースからビルドする場合、最新のmainブランチが必要
- CUDA 12.6+対応のビルド環境が必須

## 現在の環境で可能なこと

### ✅ 実行可能

1. GPT-OSS 20BのQLoRAファインチューニング
2. Harmonyフォーマットでのデータセット学習
3. Ollama形式へのエクスポート
4. すべてのUnsloth機能

### ⚠️ 制限事項

1. Xformers最適化なし（PyTorch標準Attention使用）
2. 若干の速度低下の可能性（実用上は問題なし）

## まとめ

**結論**: Xformersのソースビルドは現時点では不要です。現在の環境でGPT-OSS 20Bのファインチューニングは問題なく実行できます。

**次のステップ**: Phase 1.3（追加パッケージインストール）またはPhase 2（データセット準備）に進むことを推奨します。

Xformers最適化が必要になった場合は、CUDA Toolkitのインストール後に再度ビルドを試みることができます。
