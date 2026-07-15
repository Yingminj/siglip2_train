import torch


class MultiViewConfig:
    # === Training parameters ===
    EPOCHS        = 100
    BATCH_SIZE    = 64           # SigLIP frozen -> no grad graph -> less VRAM
    # Parameter-group learning rates.  The newly initialized multi-view modules
    # can learn relatively quickly, while the pretrained SigLIP LayerNorm is
    # adapted conservatively to avoid overfitting the small in-domain dataset.
    POOLER_LEARNING_RATE      = 1e-4
    VISION_NORM_LEARNING_RATE = 1e-5
    # Kept as a compatibility alias for logs/tools that still read this field.
    LEARNING_RATE = POOLER_LEARNING_RATE
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
    PATIENCE       = 5
    MIN_DELTA      = 1e-4

    # === Data split ===
    VAL_RATIO           = 0.1
    EVAL_EVERY_N_EPOCHS = 5
    EVAL_SKIP_FIRST_N   = 1
    ENABLE_EVAL         = True

    # === Joint image-center + text-conditioned classification loss ===
    USE_SUPCON_LOSS    = True
    SUPCON_WEIGHT      = 1.0
    SUPCON_TEMPERATURE = 0.1
    TEXT_CE_WEIGHT     = 0.5

    # === Model ===
    SIGLIP_MODEL = "/home/tjzn-zwt/Aqqqcy/github/siglip2_train/siglip2-so400m-patch14-224"

    # === Multi-View specific ===
    NUM_VIEWS = 3
    VIEW_NAMES = ['head','wrist_left', 'wrist_right']
    # 每个视角的注意力先验权重(logit 空间, 与 VIEW_NAMES 顺序一致, index 0=head).
    # 可学习, 仅作初始化先验. head=+1.0 表示头部初始注意力质量约为腕部的 e^1≈2.7 倍.
    # 设为 None 则关闭先验(初始化为全 0, 仍可学习).
    VIEW_BIAS_INIT = None

    # === Cross-View Attention Pooler ===
    NUM_QUERY_TOKENS   = 8       # Learnable query tokens for cross-attention
    POOLER_NUM_LAYERS  = 2       # Cross-attention layers
    POOLER_NUM_HEADS   = 8       # Attention heads
    POOLER_DROPOUT     = 0.1

    # === Text-conditioned gated cross-attention ===
    TEXT_FUSION_NUM_HEADS = 8
    TEXT_FUSION_DROPOUT   = 0.1
    TEXT_GATE_BIAS_INIT   = 4.0  # sigmoid(4) ~= 0.982, near-identity gate
    MAX_TEXT_LENGTH       = 64

    # Every image is compared with every entry below; never insert the sample's
    # ground-truth text alone. Replace these fallbacks with concrete descriptions
    # of the visible object/action/state for best results. Values may be a string
    # or a list of prompt variants. Missing classes fall back to their folder name.
    CLASS_PROMPTS = {}

    # Validation combines center probabilities and text-conditioned probabilities.
    CENTER_SCORE_WEIGHT = 0.7
    CENTER_SCORE_TEMPERATURE = 0.1

    # === Data paths ===
    IMAGE_ROOT = "/home/yang/siglip_train/gift_m6_picture_train40_mult_0630"

    # === Save paths ===
    MODEL_DIR  = "/home/yang/siglip_train/gift_m7_0707_1"
    MODEL_NAME = "model_siglip2_multiview_v2"

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
    NUM_CLASSES            = 7
    def to_dict(self):
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(self, attr)
                if not callable(value):
                    config_dict[attr] = value
        return config_dict
