import torch


class Config:
    # Training parameters (10个类别/每类600样本 = 总数6000，训练数据约3600)
    EPOCHS        = 100         # 减少Epoch数量，由于数据集变大，30轮已有足够多的迭代次数(约 3300 steps)
    BATCH_SIZE    = 32           # 数据量小时适当减小batch，保证每epoch有更多梯度更新步
    LEARNING_RATE = 1e-4         # V2: pooler从零初始化，需要较高LR（SigLIP完全冻结）
    WEIGHT_DECAY  = 0.05

    # Learning rate scheduler
    SCHEDULER_TYPE = "warmup_cosine"
    WARMUP_EPOCHS  = 5           # 更长 warmup 让 loss 暴涨发生在 LR 还很小的阶段
    MIN_LR         = 1e-7

    # Checkpoint saving
    SAVE_EVERY_N_EPOCHS = 5     # 每5个epoch保存一次，与评估频率一致
    MAX_CHECKPOINTS     = 10

    # Early stopping
    EARLY_STOPPING = True
    PATIENCE       = 2           # 评估频率增高后，如果连续5次评估（即10个Epoch）不提升，即可停止
    MIN_DELTA      = 1e-4

    # 数据集划分
    VAL_RATIO = 0.1              # 验证集比例（用于评估和 early stopping）
    EVAL_EVERY_N_EPOCHS = 1     # 每5个epoch评估一次
    EVAL_SKIP_FIRST_N  = 1    # 前N个epoch跳过评估和特征中心计算，加速训练初期
    ENABLE_EVAL = True           # 开启评估，用于 early stopping 和准确率监控

    # Label smoothing for SigLip Loss
    LABEL_SMOOTHING = 0.1        # 将 ±1 软化为 ±0.9，防止模型过度追求训练集确定性
    SIGLIP_WEIGHT = 0.0          # SigLipLoss 权重，测试目标为图像特征匹配时可设 0 让 SupCon 主导

    # Supervised Contrastive Loss
    USE_SUPCON_LOSS = True      # 是否叠加 SupCon Loss（直接优化图像特征判别性，与推理目标一致）
    SUPCON_WEIGHT = 1.0         # 推理用图像中心匹配，SupCon 直接优化这个目标
    SUPCON_TEMPERATURE = 0.1     # SupCon 温度系数，0.07 梯度过大易震荡，0.1 更稳定

    # 特征中心计算触发策略（基于 loss 停滞）
    LOSS_PATIENCE_FOR_EVAL = 5   # 连续N个epoch无更低loss时，计算特征中心向量并保存到新graph_info

    SIGLIP_MODEL    = "/home/kewei/Data/CLIPNEW/models/siglip2-so400m-patch14-224"
    MAX_TEXT_LENGTH = 64

    # Fine-tuning strategy
    FINETUNE_STRATEGY      = "last_n_blocks"
    NUM_LAST_VISUAL_BLOCKS = 1
    NUM_LAST_TEXT_BLOCKS   = 0
    TRAIN_VISUAL_PROJ      = True
    TRAIN_TEXT_PROJ        = True

    # === Attention Pooler (V2: patch-level tokens) ===
    NUM_QUERY_TOKENS   = 8       # Learnable query tokens for cross-attention
    POOLER_NUM_LAYERS  = 2       # Cross-attention layers
    POOLER_NUM_HEADS   = 8       # Attention heads
    POOLER_DROPOUT     = 0.3

    # Data paths
    IMAGE_ROOT = "/home/kewei/spatial_encoder/clip/data/train_data/0625_grasp"


    # Save paths
    MODEL_DIR = "/home/kewei/spatial_encoder/clip/train/siglip2_output/grasp_m2_0625"
    MODEL_NAME = "model_siglip2"

    # Resume training from checkpoint
    RESUME_FROM_CHECKPOINT = ""

    SEED = 42
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Background mask: 遮盖图像顶部不相关背景（相机位置固定）
    BACKGROUND_MASK_RATIO = 0 # 顶部20%打黑色mask，0表示不启用

    # AMP (Automatic Mixed Precision)
    USE_AMP = False

    # Image augmentation
    USE_AUGMENTATION   = True
    AUGMENTATION_CONFIG = {
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 5,
        'crop_scale': (0.8, 1.0),
        'crop_ratio': (0.95, 1.05),
        'brightness': 0.35,
        'contrast': 0.3,
        'saturation': 0.25,
        'hue': 0.08,
        'grayscale_prob': 0.15,
        'use_blur': True,
        'blur_kernel': 3,
        'blur_sigma': (0.1, 1.5),
        'perspective_scale': 0.1,
        'perspective_prob': 0.3,
        'random_erasing_prob': 0.15,
        'gaussian_noise_std': 0.02,
    }

    # ===== 时序头（Stage 2，两阶段训练）=====
    USE_TEMPORAL_HEAD    = False        # 是否启用时序头（SigLIP2 Stage 1 训练完成后启用）
    TEMPORAL_VIDEO_DIR   = "/home/kewei/spatial_encoder/clip/data/train_data/gift_ego_video/gift_ego_video"
    TEMPORAL_LABEL_DIR   = "/home/kewei/spatial_encoder/clip/data/train_data/gift_ego_video/gift_ego_label"
    SEQ_LEN              = 12            # 序列帧数（滑动窗口大小）
    SEQ_STRIDE           = 2             # 滑动窗口步长
    LSTM_HIDDEN_SIZE     = 256           # LSTM 隐层维度
    LSTM_NUM_LAYERS      = 1             # LSTM 层数
    LSTM_DROPOUT         = 0.1
    TEMPORAL_LR          = 1e-4          # 时序头学习率
    TEMPORAL_EPOCHS      = 30            # 时序头训练轮数
    TEMPORAL_BATCH_SIZE  = 32            # 时序头批大小（以序列为单位）
    TEMPORAL_NUM_CLASSES = 2             # 类别数（与训练数据一致）
    TEMPORAL_SAVE_NAME   = "temporal_head"  # 时序头 checkpoint 文件名前缀
    TEMPORAL_SUPCON_WEIGHT = 0.3         # SupCon Loss 权重（作用于 embedding，使 proj/ln 被训练）
    RESIDUAL_ALPHA       = 0.3           # 残差融合权重：α*siglip + (1-α)*时序嵌入（1.0=纯SigLIP，0.0=纯时序）

    # 时序头类型切换（向后兼容，默认 lstm）
    TEMPORAL_HEAD_TYPE      = "gru"   # "lstm" | "gru" | "transformer" | "tcn"

    # CausalTransformer 专用参数
    TRANSFORMER_NUM_HEADS   = 8        # 768 / 8 = 96 维/头
    TRANSFORMER_FF_DIM      = 1024
    TRANSFORMER_NUM_LAYERS  = 1
    TRANSFORMER_DROPOUT     = 0.1

    # TCN 专用参数
    TCN_MODEL_DIM           = 256
    TCN_KERNEL_SIZE         = 3
    TCN_NUM_BLOCKS          = 3        # 感受野覆盖 T=12
    TCN_DROPOUT             = 0.1

    # GRU 专用参数
    GRU_HIDDEN_DIM          = 512
    GRU_NUM_LAYERS          = 1
    GRU_DROPOUT             = 0.1

    # ===== ArcFace Vision-Only 训练 =====
    ARCFACE_SCALE         = 30.0       # 缩放因子 s（控制 softmax 锐度）
    ARCFACE_MARGIN        = 0.5        # 角度间距 m（弧度，0.5 ≈ 28.6°）
    NUM_CLASSES           = 2          # 类别数（需与数据集一致）
    ARCFACE_HEAD_LR_MULT  = 10.0       # ArcFace 分类头的学习率倍率（随机初始化，需更高 LR）

    # ArcFace 模式输出路径（与 SigLIP 训练分开）
    ARCFACE_MODEL_DIR     = "/home/kewei/spatial_encoder/clip/train/siglip2_output/gift_ego_arcface"
    ARCFACE_MODEL_NAME    = "model_arcface"

    def to_dict(self):
        """将配置转换为字典"""
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(self, attr)
                if not callable(value):
                    config_dict[attr] = value
        return config_dict
