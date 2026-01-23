# Diffu2KG

Knowledge Graph Generation using Diffusion Models

## 项目简介

Diffu2KG是一个基于扩散模型的知识图谱生成项目，用于从训练数据中生成知识图谱三元组。

## 项目结构

```
diffu2kg/
├── config.py              # 统一配置管理
├── evaluation.py          # 评估工具
├── main.py               # 训练脚本
├── inference_main.py      # 推理脚本
├── kg_dataloader.py      # 知识图谱数据加载器
├── model_utils.py        # 模型创建工具
├── trainer.py           # 训练器
├── transformer_model.py  # Transformer模型
├── modeling_bart.py     # BART模型
├── diffusion/           # 扩散模型核心模块
├── utils/              # 工具函数
├── pretrain/           # 预训练模型
├── checkpoints/        # 模型检查点（已忽略）
└── result/            # 结果文件（已忽略）
```

## 安装依赖

```bash
pip install torch transformers numpy
```

## 使用方法

### 训练模型

```bash
python main.py
```

### 推理和评估

```bash
python inference_main.py
```

## 配置说明

所有配置都在 `config.py` 中统一管理，包括：

- `DataConfig`: 数据路径和批次大小等配置
- `ModelConfig`: 模型架构和超参数配置
- `TrainingConfig`: 训练超参数配置
- `InferenceConfig`: 推理相关配置

## 评估指标

项目使用以下评估指标：

- Hit@1: 预测结果中排名前1的准确率
- Hit@3: 预测结果中排名前3的准确率
- Hit@10: 预测结果中排名前10的准确率
- MRR: 平均倒数排名

## 优化说明

本项目已完成代码结构优化：

1. 删除了重复的目录结构（modeling/）
2. 创建了统一的配置管理模块（config.py）
3. 提取了评估逻辑为独立模块（evaluation.py）
4. 重构了主要脚本以使用统一配置
5. 清理了未使用的文件

## 许可证

MIT License
