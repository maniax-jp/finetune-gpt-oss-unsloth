# CUDA Toolkit 12.8 インストールガイド

## 推奨バージョン: CUDA Toolkit 12.8

### 理由
- PyTorchがCUDA 12.8でビルドされている
- NVIDIAドライバー 576.88がCUDA 12.9対応（12.8は互換）
- RTX 5090（Blackwell）完全サポート

## インストール方法

### オプションA: ランチャーインストーラー（推奨）

```bash
# 一時ディレクトリに移動
cd /tmp

# CUDA 12.8ランチャーダウンロード
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_565.57.01_linux.run

# インストーラー実行（ドライバーなし、Toolkitのみ）
sudo sh cuda_12.8.0_565.57.01_linux.run --toolkit --silent --override

# クリーンアップ
rm cuda_12.8.0_565.57.01_linux.run
```

**注意**: `--toolkit`オプションでドライバーをスキップ（既にインストール済みのため）

### オプションB: パッケージマネージャー（Ubuntu/Debian）

```bash
# NVIDIA公式リポジトリ追加
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# CUDA Toolkit 12.8インストール（ドライバーなし）
sudo apt-get install -y cuda-toolkit-12-8

# クリーンアップ
rm cuda-keyring_1.1-1_all.deb
```

### オプションC: Condaパッケージ（仮想環境内）

```bash
source ./activate.sh

# CUDA Toolkitをconda-forgeからインストール
uv pip install cudatoolkit==12.8
```

**注意**: このオプションはnvccが含まれない場合があります

## 環境変数の設定

インストール後、環境変数を設定:

```bash
# CUDA Toolkitのパスを追加
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# .bashrcに永続化
cat << 'EOF' >> ~/.bashrc

# CUDA Toolkit 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

source ~/.bashrc
```

## インストール確認

```bash
# nvccバージョン確認
nvcc --version

# 期待される出力:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.8, V12.8.X
```

## Xformersビルド実行

CUDA Toolkit インストール後:

```bash
cd /home/maniax/dev/finetune-gpt-oss-unsloth
source ./activate.sh

# Xformersをソースからビルド
export TORCH_CUDA_ARCH_LIST="12.0"
uv pip install ninja
uv pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

ビルドには30分〜1時間かかる可能性があります。

## ディスク容量要件

- **CUDA Toolkit 12.8**: 約 4-6 GB
- **ビルド時の一時ファイル**: 約 2-3 GB
- **合計**: 約 6-9 GB

事前に空き容量を確認:
```bash
df -h /usr/local
df -h /tmp
```

## トラブルシューティング

### nvccが見つからない

```bash
# CUDA_HOMEを確認
echo $CUDA_HOME

# パスを再設定
export PATH=/usr/local/cuda-12.8/bin:$PATH

# nvccの場所を探す
find /usr/local -name nvcc 2>/dev/null
```

### 複数のCUDAバージョンがインストールされている

```bash
# シンボリックリンクを12.8に変更
sudo ln -sf /usr/local/cuda-12.8 /usr/local/cuda

# 確認
ls -la /usr/local/cuda
```

### WSL環境での注意点

- WSLではNVIDIAドライバーをWindows側で管理
- Linux側ではドライバーをインストールしない（--toolkitオプション必須）
- `/usr/lib/wsl/lib`にあるWSL用CUDAライブラリと競合しないよう注意

## 代替案: CUDA Toolkitなしで進める

Xformersの最適化は必須ではありません。以下の場合はCUDA Toolkitインストール不要:

- ✅ 学習速度が許容範囲内
- ✅ すぐにファインチューニングを開始したい
- ✅ ディスク容量を節約したい

現在の環境（PyTorch標準Attention）でも十分に動作します。

## 次のステップ

### CUDA Toolkitインストール後:
1. nvccの動作確認
2. Xformersソースビルド
3. Unsloth動作確認

### CUDA Toolkitなしで進める場合:
1. Phase 1.3: 追加パッケージインストール
2. Phase 2: データセット準備
