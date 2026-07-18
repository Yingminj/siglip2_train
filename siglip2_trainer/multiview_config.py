import torch


class MultiViewConfig:
    # === Training parameters ===
    EPOCHS        = 50
    BATCH_SIZE    = 64           # SigLIP frozen -> no grad graph -> less VRAM
    LEARNING_RATE = 1e-4         # Single LR for pooler (only trainable module)
    WEIGHT_DECAY  = 0.05
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32*2 = 64

    # === Scheduler ===
    SCHEDULER_TYPE = "warmup_cosine"
    WARMUP_EPOCHS  = 5
    MIN_LR         = 1e-7

    # === Checkpoint ===
    SAVE_EVERY_N_EPOCHS = 5
    MAX_CHECKPOINTS     = 10

    # === Early stopping ===
    EARLY_STOPPING = True
    PATIENCE       = 3
    MIN_DELTA      = 1e-3

    # === Data split ===
    VAL_RATIO           = 0.2
    EVAL_EVERY_N_EPOCHS = 5
    EVAL_SKIP_FIRST_N   = 1
    ENABLE_EVAL         = True
    TRAIN_EVAL_MAX_SAMPLES = 512  # 每次评估时从训练集均匀抽样的最大样本数，用于判断过拟合

    # === Loss (SupCon only, no text) ===
    USE_SUPCON_LOSS    = True
    SUPCON_WEIGHT      = 1.0
    SUPCON_TEMPERATURE = 0.1

    # === Model ===
    SIGLIP_MODEL = "/home/liuqian/Aqcy/siglip2_train/siglip2-so400m-patch14-224"

    # === Multi-View specific ===
    NUM_VIEWS = 3
    VIEW_NAMES = ['center', 'left', 'right']
    # 每个视角的注意力先验权重(logit 空间, 与 VIEW_NAMES 顺序一致, index 0=center).
    # 可学习, 仅作初始化先验. center=+1.0 表示中间视角初始注意力质量约为侧视角的 e^1≈2.7 倍.
    # 设为 None 则关闭先验(初始化为全 0, 仍可学习).
    VIEW_BIAS_INIT = None

    # === Cross-View Attention Pooler ===
    NUM_QUERY_TOKENS   = 8       # Learnable query tokens for cross-attention
    POOLER_NUM_LAYERS  = 2       # Cross-attention layers
    POOLER_NUM_HEADS   = 8       # Attention heads
    POOLER_DROPOUT     = 0.1

    # === Data paths ===
    # "image": image class folders.
    # "video": MP4 files with frame-level TXT labels.
    # "video_class_folders": predefined train/val/test/M1...M6 directories.
    # "video_class_folders_transition": class_folders + transition-dense video data.
    DATA_MODE = "video_class_folders_transition"
    IMAGE_ROOT = "/home/liuqian/Aqcy/gift_m6_picture_train40_mult_0630"
    VIDEO_DATA_ROOT = "/home/liuqian/Aqcy/datagift_0716_rsb"
    VIDEO_FRAMES_PER_VIDEO = 24
    GRAPH_INFO_PATH = "/home/liuqian/Aqcy/gift_m6_picture_train40_mult_0630/graph_info.json"
    VIDEO_ROOT = "/home/liuqian/Aqcy/cube_data/cubedata_multiviewer_0713"
    VIDEO_LABEL_ROOT = "/home/liuqian/Aqcy/cube_data/cubedata_multiviewer_0713_label"
    VIDEO_FRAME_STRIDE = 5       # 1=每帧；5=每 5 帧采一帧
    VIDEO_SAMPLES_PER_CLASS = 1000  # 每状态总数；按 VAL_RATIO 分为 800/200

    # ---- 边界效应优化 ----
    # 类文件夹模式下，跳过视频首尾的比例（0.0=保留全部帧，包括边界帧）
    EDGE_FRACTION = 0.0
    # 边界强调: 0.0=均匀采样, 0.3=30%样本集中在视频首尾20%区域
    BOUNDARY_EMPHASIS = 0.25
    # 过渡帧密集采样: 标签切换点前后各 N 帧
    TRANSITION_WINDOW = 30
    # 过渡窗口内的采样步长 (1=每帧都采)
    TRANSITION_DENSE_STRIDE = 2
    # 过渡帧数据集占总训练数据的比例 (0.0=不加过渡数据, 0.3=30%)
    TRANSITION_MIX_RATIO = 0.3

    # === Save paths ===
    MODEL_DIR  = "/home/liuqian/Aqcy/train_giftvedio_0718/trainresult_0718_base"
    MODEL_NAME = "baseline"

    # === Resume ===
    RESUME_FROM_CHECKPOINT = ""

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # === Augmentation (gentler for multi-view to preserve spatial semantics) ===
    USE_AUGMENTATION    = True
    AUGMENTATION_CONFIG = {
        'horizontal_flip_prob': 0.3,
        'rotation_degrees': 3,
        'crop_scale': (0.85, 1.0),
        'crop_ratio': (0.95, 1.05),
        'brightness': 0.3,
        'contrast': 0.25,
        'saturation': 0.2,
        'hue': 0.08,
        'grayscale_prob': 0.12,
        'use_blur': True,
        'blur_kernel': 3,
        'blur_sigma': (0.1, 1.2),
        'perspective_scale': 0.05,
        'perspective_prob': 0.15,
        'random_erasing_prob': 0.1,
        'gaussian_noise_std': 0.015,
    }

    # === Feature center ===
    LOSS_PATIENCE_FOR_EVAL = 5
    NUM_CLASSES            = 6
    def to_dict(self):
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(self, attr)
                if not callable(value):
                    config_dict[attr] = value
        return config_dict
