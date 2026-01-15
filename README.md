## 预训练模型

由于GitHub文件大小限制，预训练模型的大文件未包含在仓库中。

### 自动下载

首次运行时，模型会自动从HuggingFace下载：

```python
from transformers import BartModel
model = BartModel.from_pretrained('facebook/bart-base', cache_dir='./pretrain/bart-base')
```

### 手动下载

如果自动下载失败，请手动下载以下文件到 `pretrain/bart-base/` 目录：

- [config.json](https://huggingface.co/facebook/bart-base/resolve/main/config.json)
- [vocab.json](https://huggingface.co/facebook/bart-base/resolve/main/vocab.json)
- [merges.txt](https://huggingface.co/facebook/bart-base/resolve/main/merges.txt)

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/13flix/diffu2kg.git
cd diffu2kg

# 2. 下载预训练模型（自动）
python -c "from transformers import BartModel; BartModel.from_pretrained('facebook/bart-base', cache_dir='./pretrain/bart-base')"

# 3. 运行训练或推理
python main.py
```
