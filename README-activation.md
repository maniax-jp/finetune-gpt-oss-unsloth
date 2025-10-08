# 仮想環境の自動有効化

## ✅ 設定完了

このプロジェクトでは仮想環境の有効化を簡単にするため、以下のファイルが作成されました：

### 1. `activate.sh` - 推奨方法

プロジェクトディレクトリで以下を実行：

```bash
source ./activate.sh
```

これにより自動的に：
- ✅ Python 3.12仮想環境が有効化
- ✅ `TORCH_CUDA_ARCH_LIST="12.0"` が設定（Blackwell対応）
- ✅ 環境情報が表示

**出力例**:
```
✅ Virtual environment activated
Python: Python 3.12.3
Location: /home/maniax/dev/finetune-gpt-oss-unsloth/.venv/bin/python
TORCH_CUDA_ARCH_LIST: 12.0
```

### 2. `.envrc` - direnv使用時（オプション）

direnvがインストールされている場合：

```bash
# direnvを有効化
direnv allow

# ディレクトリに入ると自動的に有効化される
cd /home/maniax/dev/finetune-gpt-oss-unsloth
# → 自動的に .venv が有効化
```

direnvのインストール（必要な場合）：
```bash
sudo apt install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

## 使用方法

### 方法A: activate.sh を使用（推奨）

```bash
cd /home/maniax/dev/finetune-gpt-oss-unsloth
source ./activate.sh
python your_script.py
```

### 方法B: 手動で有効化

```bash
cd /home/maniax/dev/finetune-gpt-oss-unsloth
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"
python your_script.py
```

### 方法C: エイリアス作成（便利）

`~/.bashrc` に追加：

```bash
alias finetune='cd /home/maniax/dev/finetune-gpt-oss-unsloth && source ./activate.sh'
```

その後：
```bash
source ~/.bashrc
finetune  # → プロジェクトディレクトリに移動＆環境有効化
```

## 環境の確認

仮想環境が正しく有効化されているか確認：

```bash
which python
# 出力: /home/maniax/dev/finetune-gpt-oss-unsloth/.venv/bin/python

echo $TORCH_CUDA_ARCH_LIST
# 出力: 12.0

python -c "import unsloth; print('Unsloth OK')"
# 出力: Unsloth OK
```

## 環境の無効化

作業終了時：

```bash
deactivate
```

## トラブルシューティング

### activate.sh が動かない

権限を確認：
```bash
chmod +x ./activate.sh
```

### Python が見つからない

.venv が存在するか確認：
```bash
ls -la .venv/bin/python
```

存在しない場合は再作成：
```bash
uv venv --python 3.12
```

### 環境変数が設定されない

手動で設定：
```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

または `.bashrc` に追加して永続化：
```bash
echo 'export TORCH_CUDA_ARCH_LIST="12.0"' >> ~/.bashrc
source ~/.bashrc
```
