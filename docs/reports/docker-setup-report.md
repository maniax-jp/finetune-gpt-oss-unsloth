# Unsloth Dockerセットアップレポート

実施日: 2025-10-08

## セットアップ方法: Docker使用（オプションA）

## ✅ インストール結果

### Docker環境
- **Dockerバージョン**: 28.4.0
- **Docker動作状態**: ✅ 正常

### NVIDIA Container Toolkit
- **状態**: ✅ 正常動作
- **GPU認識**: RTX 5090が正しく認識されている
- **テストコマンド**: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` ✅ 成功

### Unsloth Dockerイメージ
- **イメージ名**: `unsloth/unsloth:latest`
- **イメージID**: 63362bdb6533
- **サイズ**: 27.2GB
- **作成日**: 5 days ago
- **ダウンロード状態**: ✅ 完了

## ✅ 動作確認結果

### Unslothインポートテスト

コマンド実行:
```bash
docker run --rm --gpus all --entrypoint python unsloth/unsloth -c "import unsloth; ..."
```

**出力結果**:
```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
INFO 10-08 08:45:44 [__init__.py:216] Automatically detected platform cuda.
WARNING 10-08 08:45:45 [interface.py:391] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth imported successfully
PyTorch: 2.8.0+cu128
CUDA Available: True
GPU: NVIDIA GeForce RTX 5090
```

### 確認された環境詳細

| 項目 | 値 | 状態 |
|------|-----|------|
| Unsloth | インポート成功 | ✅ |
| PyTorch | 2.8.0+cu128 | ✅ |
| CUDA | Available (True) | ✅ |
| GPU検出 | RTX 5090 | ✅ |

### 注意事項

⚠️ **WSL検出による警告**:
```
WARNING: Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
```

- WSL環境では`pin_memory=False`が自動設定される
- パフォーマンスへの影響は軽微（学習自体は問題なく実行可能）
- WSLの制約によるもので、回避不可

## Docker使用方法

### 基本コマンド

#### 1. インタラクティブシェル起動
```bash
docker run --rm -it --gpus all \
  -v /home/maniax/dev/finetune-gpt-oss-unsloth:/workspace \
  --entrypoint /bin/bash \
  unsloth/unsloth
```

#### 2. Pythonスクリプト実行
```bash
docker run --rm --gpus all \
  -v /home/maniax/dev/finetune-gpt-oss-unsloth:/workspace \
  --entrypoint python \
  unsloth/unsloth /workspace/your_script.py
```

#### 3. Jupyter Notebook起動（デフォルト）
```bash
docker run --rm -it --gpus all \
  -p 8888:8888 \
  -v /home/maniax/dev/finetune-gpt-oss-unsloth:/workspace \
  unsloth/unsloth
```

その後、ブラウザで `http://localhost:8888` にアクセス

### おすすめの使用方法

プロジェクトディレクトリをマウントして作業:

```bash
# プロジェクトディレクトリに移動
cd /home/maniax/dev/finetune-gpt-oss-unsloth

# Dockerコンテナでインタラクティブシェル起動
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  unsloth/unsloth
```

コンテナ内で:
```bash
# Unslothが利用可能
python -c "import unsloth; print('Ready!')"

# ファインチューニングスクリプト実行
python finetune_gpt_oss.py
```

## docker-compose設定（オプション）

より便利に使うため、`docker-compose.yml`を作成することを推奨:

```yaml
version: '3.8'

services:
  unsloth:
    image: unsloth/unsloth:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - .:/workspace
    working_dir: /workspace
    stdin_open: true
    tty: true
    entrypoint: /bin/bash
```

使用方法:
```bash
docker-compose run --rm unsloth
```

## Blackwell (RTX 5090) 対応状況

### ✅ 確認済み対応項目

1. **CUDA 12.8**: Dockerイメージが`cu128`対応PyTorchを含む
2. **GPU認識**: RTX 5090が正しく認識される
3. **Unsloth Zoo**: 自動的に最適化パッチを適用

### 追加設定不要

Docker版では以下の設定が不要（イメージ内で完結）:
- ❌ `TORCH_CUDA_ARCH_LIST="12.0"`設定
- ❌ Tritonの手動アップデート
- ❌ Xformersのソースビルド

すべてDockerイメージに含まれているため、即座に使用可能。

## トラブルシューティング

### イメージが大きすぎる場合

27.2GBのイメージサイズが問題になる場合は、オプションB（ローカルインストール）を検討してください。

### GPUが認識されない場合

```bash
# NVIDIA Container Toolkit再インストール
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### WSL環境での注意点

- `pin_memory=False`の警告は正常（無視可能）
- メモリ不足の場合は`~/.wslconfig`でWSLメモリを増やす

## 総合評価

### ✅ Phase 1.2完了

| 項目 | 状態 |
|------|------|
| Docker環境 | ✅ |
| NVIDIA Container Toolkit | ✅ |
| Unslothイメージ | ✅ ダウンロード済み |
| GPU認識 | ✅ RTX 5090認識 |
| Unsloth動作確認 | ✅ インポート成功 |
| PyTorch CUDA | ✅ 利用可能 |

**Phase 1.2（Unslothインストール）**: ✅ 完了

Docker環境でのUnslothセットアップが正常に完了しました。

## 次のステップ

Phase 1.3またはPhase 2に進む準備が整いました:

- **Phase 1.3**: 必要なパッケージのインストール（Ollama SDKなど）
- **Phase 2**: データセット準備

Docker環境が整ったので、いつでもファインチューニング作業を開始できます。

## 参考情報

- Unsloth Docker Hub: https://hub.docker.com/r/unsloth/unsloth
- Unsloth Documentation: https://docs.unsloth.ai/
- Docker + GPU Guide: https://docs.docker.com/config/containers/resource_constraints/#gpu
