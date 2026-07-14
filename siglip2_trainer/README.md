# siglip2_trainer 模块说明

SigLIP2 微调训练包，由主入口 `fine_tuning_siglip2_fixed_eval.py` 调用。

---

## 目录结构

```
siglip2_trainer/
├── config.py          配置
├── augmentation.py    数据增强
├── losses.py          损失函数
├── models.py          模型定义与加载
├── datasets.py        数据集
├── checkpoints.py     Checkpoint 管理与早停
├── lstm_stage.py      LSTM Stage 2 训练
├── evaluation.py      分类评估与可视化
├── visualization.py   训练曲线与日志
└── logging_utils.py   配置保存与数据集统计
```

---

## 各模块说明

### `config.py` — 训练配置

**类**：`Config`

所有超参数和路径的集中配置，修改此文件即可控制训练行为，无需改动其他代码。

主要配置项：
- `EPOCHS / BATCH_SIZE / LEARNING_RATE / WEIGHT_DECAY`：基础训练参数
- `SCHEDULER_TYPE`：学习率调度器（`warmup_cosine` / `cosine` / `step` / `exponential`）
- `FINETUNE_STRATEGY`：微调策略（`last_n_blocks` / `proj_only` / `all`）
- `IMAGE_ROOT / TEXT_ROOT`：训练数据路径
- `MODEL_DIR`：输出目录（checkpoint、graph_info、可视化图片）
- `RESUME_FROM_CHECKPOINT`：断点续训路径（留空则从头训练）
- `ENABLE_EVAL`：是否在训练中途开启评估
- **`USE_LSTM`**：是否启用 LSTM Stage 2（`True` / `False`，在此修改即可）
- `LSTM_VIDEO_DIR / LSTM_LABEL_DIR`：LSTM 训练用的视频和标注目录

---

### `augmentation.py` — 数据增强

**类**：`SigLIPAugmentation`

对训练图片做随机增强，包括水平翻转、旋转、随机裁剪、颜色抖动、透视变换。增强参数通过 `Config.AUGMENTATION_CONFIG` 传入。仅在训练时生效（`is_training=True`）。

---

### `losses.py` — 损失函数

**类**：`SigLipLoss`, `SupConLoss`

- **`SigLipLoss`**：SigLIP 对比损失，基于图像-文本 logit 矩阵的 sigmoid 损失。支持 label smoothing 和多类别 batch（同类正样本不作为负样本）。
- **`SupConLoss`**：有监督对比损失（Khosla et al., NeurIPS 2020）。直接在图像特征空间上拉近同类、推远不同类，与推理阶段的最近中心匹配目标一致。

两个 loss 加权叠加：`total = SigLipLoss + SUPCON_WEIGHT × SupConLoss`。

---

### `models.py` — 模型定义与加载

**类**：`LSTMTemporalHead`
**函数**：`setup_model`, `load_checkpoint`

- **`LSTMTemporalHead`**：轻量 LSTM 头，接收 SigLIP2 特征序列 `[B, T, 768]`，输出聚合 embedding `[B, 768]` 和分类 logits `[B, C]`。Stage 2 专用。
- **`setup_model`**：加载 SigLIP2 模型（AutoModel），按策略冻结/解冻参数，创建 Adam 优化器和学习率调度器。
- **`load_checkpoint`**：从 `.pt` 文件恢复模型权重、优化器状态、调度器状态，用于断点续训。

---

### `datasets.py` — 数据集

**类**：`SimpleDataset`, `GroupedDataset`, `VideoSequenceDataset`
**函数**：`load_label_file`, `make_collate_fn`

- **`SimpleDataset`**：主训练数据集。每个样本返回一张图片 + 对应文本 + 类别标签。支持按比例划分 train / val / test 三份。
- **`GroupedDataset`**：旧版数据集（每次随机从每个类别各取一张），保留兼容。
- **`VideoSequenceDataset`**：Stage 2 专用。从预提取的特征缓存（`.pt`）中构建滑动窗口序列，供 LSTM 训练。
- **`load_label_file`**：解析 `frame_idx class_id` 格式的帧级标注文件（1-based → 0-based）。
- **`make_collate_fn`**：为 `GroupedDataset` 创建 collate 函数，调用完整 processor 一次生成 `spatial_shapes`（naflex 模型必需）。

---

### `checkpoints.py` — Checkpoint 管理与早停

**类**：`CheckpointManager`, `EarlyStopping`

- **`CheckpointManager`**：定期保存 checkpoint（`_epoch_N.pt`），维护最佳 loss（`_best_loss.pt`）和最佳评估准确率（`_best_eval.pt`）的模型；自动清理超出 `MAX_CHECKPOINTS` 的旧文件。
- **`EarlyStopping`**：监控评估准确率，连续 `PATIENCE` 次无提升时触发停止。

---

### `lstm_stage.py` — LSTM Stage 2 训练

**函数**：`precompute_video_features`, `train_lstm_stage`, `compute_and_save_class_centers_lstm`

Stage 2 两阶段逻辑：

**为什么 LSTM 使用视频数据而非静态图片？**
SigLIP2 的静态图片训练集中每张图片是独立的，没有时序关系。LSTM 的作用是学习相邻帧之间的状态转移规律（例如礼盒包装的连续动作），必须使用时序连续的帧序列才有意义。**LSTM 仍然建立在 SigLIP2 的训练结果之上**：`precompute_video_features` 调用的就是微调后的 SigLIP2 从视频帧中提取特征，LSTM 接收这些特征序列进行训练。

- **`precompute_video_features`**：用冻结的 SigLIP2 对所有训练视频逐帧提取特征，缓存到 `siglip2_features_cache.pt`，避免每个 LSTM epoch 重复提取。
- **`train_lstm_stage`**：冻结 SigLIP2，加载特征缓存，用滑动窗口序列训练 LSTM，Loss = CE（最后帧预测）。保存最佳和最终权重。
- **`compute_and_save_class_centers_lstm`**：用 SigLIP2 + LSTM 计算类别中心，写入 `graph_info_lstm_final.json` 的 `center_feature_siglip2_lstm` 字段。

---

### `evaluation.py` — 分类评估与可视化

**函数**：`compute_and_save_class_centers`, `evaluate_with_class_centers`, `plot_confusion_matrix`, `plot_classification_results`

- **`compute_and_save_class_centers`**：用当前 SigLIP2 权重对训练集所有图片提取特征，取每类均值作为类别中心，更新并保存到 `classification_viz/graph_info_{epoch}.json`。
- **`evaluate_with_class_centers`**：在测试集上计算每张图的特征与各类中心的余弦相似度，输出总体和各类准确率，保存结果 JSON 和可视化图片。
- **`plot_confusion_matrix`**：绘制归一化混淆矩阵热力图（seaborn）。
- **`plot_classification_results`**：绘制测试样本的图片 + 各类相似度柱状图。

---

### `visualization.py` — 训练曲线与日志

**函数**：`plot_loss_curve`, `save_training_summary`

- **`plot_loss_curve`**：每个 epoch 结束后更新 `loss_curve.png`、`lr_curve.png`、`accuracy_curve.png`，并追加写入 `training_log.txt`。
- **`save_training_summary`**：训练结束后保存完整的 loss / lr / accuracy 历史到文本文件。

---

### `logging_utils.py` — 配置保存与数据集统计

**类**：`NumpyEncoder`
**函数**：`save_training_config`, `collect_dataset_info`, `collect_model_info`

- **`NumpyEncoder`**：自定义 JSON 编码器，自动将 `numpy` 类型转换为 Python 原生类型，供所有模块序列化时使用。
- **`save_training_config`**：训练开始时将完整配置保存为 `training_config.json` 和 `training_config.txt`，便于复现实验。
- **`collect_dataset_info`**：统计数据集类别数、每类样本数等，兼容 `SimpleDataset` 和 `GroupedDataset`。
- **`collect_model_info`**：统计模型总参数量、可训练参数量及比例，可训练层名称列表。

---

## 训练流程

```
Stage 1: SigLIP2 微调
  setup_model → SimpleDataset → train_one_epoch × EPOCHS
  每 SAVE_EVERY_N_EPOCHS:
    checkpoint_manager.save + compute_and_save_class_centers
  每 EVAL_EVERY_N_EPOCHS (需 ENABLE_EVAL=True):
    evaluate_with_class_centers → early_stopping

Stage 2: LSTM 短期记忆（需 USE_LSTM=True）
  precompute_video_features（首次耗时，之后复用缓存）
  train_lstm_stage × LSTM_EPOCHS
  compute_and_save_class_centers_lstm
```

## 如何控制 LSTM

在 `config.py` 中修改：
```python
USE_LSTM = True   # 启用 Stage 2
USE_LSTM = False  # 仅 Stage 1
```
