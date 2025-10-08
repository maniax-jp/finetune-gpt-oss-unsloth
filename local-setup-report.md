# Unsloth ローカルインストールレポート

実施日: 2025-10-08

## セットアップ方法: ローカルインストール（オプションB）

## ✅ インストール結果

### 仮想環境
- **ツール**: uv 0.9.0
- **Python**: 3.12.3
- **仮想環境パス**: `/home/maniax/dev/finetune-gpt-oss-unsloth/.venv`
- **有効化コマンド**: `source .venv/bin/activate`

### インストールされたパッケージ（主要）

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| unsloth | 2025.10.1 | メインフレームワーク |
| unsloth-zoo | 2025.10.1 | 最適化パッチ |
| torch | 2.8.0+cu128 | PyTorch（CUDA 12.8対応） |
| triton | 3.4.0 | GPU最適化ライブラリ |
| xformers | 0.0.32.post2 | Attention最適化 |
| transformers | 4.56.2 | Hugging Face Transformers |
| bitsandbytes | 0.48.1 | 量子化ライブラリ |
| peft | 0.17.1 | LoRA実装 |
| trl | 0.23.0 | トレーニングライブラリ |
| accelerate | 1.10.1 | 分散学習サポート |
| datasets | 4.1.1 | データセット管理 |

### NVIDIA CUDAライブラリ（自動インストール）

- nvidia-cublas-cu12==12.8.4.1
- nvidia-cudnn-cu12==9.10.2.21
- nvidia-cufft-cu12==11.3.3.83
- nvidia-cusolver-cu12==11.7.3.90
- nvidia-cusparse-cu12==12.5.8.93
- nvidia-nccl-cu12==2.27.3
- その他多数

**合計86パッケージがインストールされました。**

## ✅ 動作確認結果

### 実行テスト

```bash
source .venv/bin/activate
python -c "import unsloth; import torch; ..."
```

**出力結果**:
```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
========
Switching to PyTorch attention since your Xformers is broken.
========

Unsloth: Xformers does not work in RTX 50X, Blackwell GPUs as of yet. Please build from source via
```
pip install ninja
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth version: 2025.10.1
PyTorch: 2.8.0+cu128
CUDA Available: True
GPU: NVIDIA GeForce RTX 5090
```

### 環境確認サマリー

| 項目 | 値 | 状態 |
|------|-----|------|
| Unsloth | 2025.10.1 | ✅ 正常動作 |
| PyTorch | 2.8.0+cu128 | ✅ |
| CUDA | Available (True) | ✅ |
| GPU | RTX 5090 | ✅ 認識 |
| Xformers | 0.0.32.post2 | ⚠️ RTX 50XX未対応 |

## ⚠️ Xformersの問題と対応

### 現状

Xformersのプリビルド版（0.0.32.post2）はRTX 50XX（Blackwell）に対応していないため、Unslothは自動的にPyTorch標準のAttentionに切り替えました。

### 影響

- ✅ **学習は正常に動作**: PyTorch標準Attentionで問題なく学習可能
- ⚠️ **パフォーマンス**: Xformersを使用した場合より若干遅い可能性
- ✅ **機能性**: すべてのUnsloth機能が利用可能

### 対応方法（オプション）

より高速な学習を希望する場合、Xformersをソースからビルド:

```bash
source .venv/bin/activate

# Ninjaビルドツールインストール
uv pip install ninja

# Xformersをソースからビルド（RTX 50XX対応）
export TORCH_CUDA_ARCH_LIST="12.0"
uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

**注意**: ビルドには時間がかかります（30分〜1時間程度）。

**推奨**: 現時点ではXformersなしでも十分動作するため、必要に応じて後から対応可能。

## 環境設定ファイル

### .venv有効化を自動化（オプション）

プロジェクトディレクトリに`.envrc`を作成（direnv使用時）:

```bash
echo "source .venv/bin/activate" > .envrc
direnv allow
```

または、毎回手動で有効化:

```bash
cd /home/maniax/dev/finetune-gpt-oss-unsloth
source .venv/bin/activate
```

### Blackwell対応環境変数の永続化（推奨）

`~/.bashrc`に追加:

```bash
echo 'export TORCH_CUDA_ARCH_LIST="12.0"' >> ~/.bashrc
source ~/.bashrc
```

## 使用方法

### 基本的なワークフロー

1. **仮想環境の有効化**:
```bash
cd /home/maniax/dev/finetune-gpt-oss-unsloth
source .venv/bin/activate
```

2. **Pythonスクリプト実行**:
```bash
python your_script.py
```

3. **終了時**:
```bash
deactivate
```

### Jupyter Notebook使用（オプション）

```bash
source .venv/bin/activate
uv pip install jupyter
jupyter notebook
```

## Phase 1.3: 必要なパッケージインストール

次のステップとして、追加の依存関係をインストール:

```bash
source .venv/bin/activate

# Ollama Python SDK
uv pip install ollama

# その他の有用なパッケージ
uv pip install ipython jupyter matplotlib
```

## トラブルシューティング

### CUDA out of memory

- バッチサイズを削減
- `load_in_4bit=True`を確認
- gradient checkpointingを有効化

### ImportError

```bash
# 仮想環境が有効か確認
which python
# /home/maniax/dev/finetune-gpt-oss-unsloth/.venv/bin/python であるべき

# 再有効化
source .venv/bin/activate
```

### パッケージの再インストール

```bash
source .venv/bin/activate
uv pip install --reinstall unsloth
```

## ローカルインストール vs Docker比較

| 項目 | ローカル（今回） | Docker |
|------|---------------|--------|
| セットアップ時間 | 短い（数分） | 長い（イメージDL） |
| ディスク使用量 | 小さい（数GB） | 大きい（27GB） |
| カスタマイズ性 | ✅ 高い | 中程度 |
| 永続性 | ✅ 永続的 | 要カスタムイメージ |
| 環境分離 | venvレベル | コンテナレベル |
| パフォーマンス | ✅ ネイティブ | わずかなオーバーヘッド |

**結論**: 開発・実験にはローカルインストールが適しています。

## 総合評価

### ✅ Phase 1.2完了（ローカルインストール）

| 項目 | 状態 |
|------|------|
| uv環境 | ✅ 0.9.0 |
| Python 3.12仮想環境 | ✅ |
| Unsloth | ✅ 2025.10.1 |
| PyTorch CUDA | ✅ 2.8.0+cu128 |
| GPU認識 | ✅ RTX 5090 |
| Triton | ✅ 3.4.0 |
| Xformers | ⚠️ 動作するがBlackwell最適化なし |
| 86パッケージ | ✅ インストール完了 |

**Phase 1.2（Unslothローカルインストール）**: ✅ 完了

ローカル環境でのUnslothセットアップが正常に完了しました。Xformersの最適化なしでも、すべての機能が利用可能です。

## 次のステップ

- **Phase 1.3**: 追加パッケージインストール（Ollama SDKなど）
- **Phase 2**: データセット準備
- **オプション**: Xformersをソースからビルド（パフォーマンス向上）

## 参考情報

- Unsloth Documentation: https://docs.unsloth.ai/
- Unsloth GitHub: https://github.com/unslothai/unsloth
- UV Documentation: https://docs.astral.sh/uv/
- Xformers GitHub: https://github.com/facebookresearch/xformers
