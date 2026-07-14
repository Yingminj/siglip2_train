import math
import os

import torch
from torch import nn, optim
from transformers import AutoModel, AutoProcessor

from .losses import SigLipLoss


class LSTMTemporalHead(nn.Module):
    """
    轻量 LSTM 短期记忆头，叠加在冻结的 SigLIP2 视觉特征之上。

    输入: SigLIP2 特征序列 [B, T, D=768]
    输出:
      - embedding [B, D]  — 投影回 SigLIP2 特征空间，用于类别中心匹配
      - logits    [B, C]  — 最后时间步的分类 logits，用于 CE 训练
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=1,
                 dropout=0.1, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj       = nn.Linear(hidden_dim, input_dim)
        self.ln         = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, seq_embeds):
        """
        seq_embeds: [B, T, D] — L2 归一化的 SigLIP2 特征序列
        Returns:
            embedding: [B, D]  (未 L2 归一化，调用方归一化)
            logits:    [B, C]
        """
        lstm_out, _ = self.lstm(seq_embeds)   # [B, T, H]
        last        = lstm_out[:, -1, :]      # [B, H]
        embedding   = self.ln(self.proj(last))  # [B, D]
        logits      = self.classifier(last)   # [B, C]
        return embedding, logits


class GRUTemporalHead(nn.Module):
    """
    单向 GRU 时序头（LSTM 直接替换版）。
    参数量约为 LSTM 的 75%，T=12 短序列上性能相当甚至略好。
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=1,
                 dropout=0.1, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj       = nn.Linear(hidden_dim, input_dim)
        self.ln         = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, seq_embeds):
        gru_out, _ = self.gru(seq_embeds)
        last       = gru_out[:, -1, :]
        embedding  = self.ln(self.proj(last))
        logits     = self.classifier(last)
        return embedding, logits


class CausalTransformerHead(nn.Module):
    """
    因果 Transformer 时序头（首选）。
    带因果掩码的 Transformer Encoder，取最后帧输出，直接 attend 所有过去帧。
    """
    def __init__(self, input_dim=768, num_heads=8, ff_dim=1024,
                 num_layers=1, dropout=0.1, num_classes=10):
        super().__init__()
        self.register_buffer('pos_enc', self._build_pos_enc(64, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(input_dim),
        )
        self.proj       = nn.Linear(input_dim, input_dim)
        self.ln         = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    @staticmethod
    def _build_pos_enc(max_len, d_model):
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, seq_embeds):
        B, T, D = seq_embeds.shape
        x = seq_embeds + self.pos_enc[:, :T, :]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        x         = self.transformer(x, mask=causal_mask, is_causal=True)
        last      = x[:, -1, :]
        embedding = self.ln(self.proj(last))
        logits    = self.classifier(last)
        return embedding, logits


class CausalConv1d(nn.Module):
    """因果 1D 卷积：只在左侧 padding，保证输出不依赖未来帧。"""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):  # x: [B, C, T]
        return self.conv(x)[:, :, :x.shape[2]]


class TCNBlock(nn.Module):
    """TCN 残差块：两层因果膨胀卷积 + 残差连接。"""
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.c1   = CausalConv1d(channels, kernel_size, dilation)
        self.c2   = CausalConv1d(channels, kernel_size, dilation)
        self.n1   = nn.LayerNorm(channels)
        self.n2   = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x):  # x: [B, C, T]
        r = x
        x = self.act(self.c1(self.n1(x.transpose(1, 2)).transpose(1, 2)))
        x = self.drop(x)
        x = self.act(self.c2(self.n2(x.transpose(1, 2)).transpose(1, 2)))
        return self.drop(x) + r


class TCNTemporalHead(nn.Module):
    """
    TCN 因果时序头。
    膨胀因子 [1, 2, 4] 覆盖 T=12，完全并行，无梯度消失。
    """
    def __init__(self, input_dim=768, model_dim=256, kernel_size=3,
                 num_blocks=3, dropout=0.1, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.blocks     = nn.ModuleList([
            TCNBlock(model_dim, kernel_size, 2 ** i, dropout)
            for i in range(num_blocks)
        ])
        self.proj       = nn.Linear(model_dim, input_dim)
        self.ln         = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, seq_embeds):
        x    = self.input_proj(seq_embeds).transpose(1, 2)  # [B, model_dim, T]
        for blk in self.blocks:
            x = blk(x)
        last      = x.transpose(1, 2)[:, -1, :]
        embedding = self.ln(self.proj(last))
        logits    = self.classifier(last)
        return embedding, logits


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    从checkpoint加载模型和训练状态

    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型实例
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器实例（可选）

    Returns:
        dict: 包含epoch, loss等信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")

    print(f"\n从checkpoint加载: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ 已加载模型权重")
    else:
        print(f"  ⚠️  警告: checkpoint中没有找到model_state_dict")

    # 加载optimizer状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ 已加载Optimizer状态")
    elif optimizer is not None:
        print(f"  ⚠️  警告: checkpoint中没有找到optimizer_state_dict")

    # 加载scheduler状态（如果有的话）
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  ✓ 已加载Scheduler状态")

    # 提取训练信息
    info = {
        'start_epoch': 0,
        'previous_loss': None
    }

    if 'epoch' in checkpoint:
        info['start_epoch'] = checkpoint['epoch'] + 1  # 从下一个epoch继续
        print(f"  ✓ 从Epoch {info['start_epoch']}继续训练 (checkpoint保存于Epoch {checkpoint['epoch']})")
    else:
        print(f"  ⚠️  警告: checkpoint中没有找到epoch信息")

    if 'loss' in checkpoint:
        info['previous_loss'] = checkpoint['loss']
        print(f"  ✓ Checkpoint的Loss: {checkpoint['loss']:.4f}")

    if 'lr' in checkpoint:
        print(f"  ✓ Checkpoint的学习率: {checkpoint['lr']:.2e}")

    return info


def setup_model(config):
    print(f"Loading model with AutoModel: {config.SIGLIP_MODEL}")
    print(f"Device: {config.DEVICE}")

    # 使用 AutoModel 和 AutoProcessor，自动识别模型类型
    model     = AutoModel.from_pretrained(config.SIGLIP_MODEL)
    processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
    model     = model.to(config.DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    if config.FINETUNE_STRATEGY == "proj_only":
        # SigLIP2 没有独立的 visual_projection/text_projection，
        # 投影功能内置在 vision_model.head 和 text_model.head 中
        if config.TRAIN_VISUAL_PROJ:
            if hasattr(model, 'visual_projection'):
                model.visual_projection.weight.requires_grad = True
            elif hasattr(model.vision_model, 'head'):
                for param in model.vision_model.head.parameters():
                    param.requires_grad = True
        if config.TRAIN_TEXT_PROJ:
            if hasattr(model, 'text_projection'):
                model.text_projection.weight.requires_grad = True
            elif hasattr(model.text_model, 'head'):
                for param in model.text_model.head.parameters():
                    param.requires_grad = True

    elif config.FINETUNE_STRATEGY == "last_n_blocks":
        print(f"\nFine-tuning last {config.NUM_LAST_VISUAL_BLOCKS} visual blocks")
        print(f"Fine-tuning last {config.NUM_LAST_TEXT_BLOCKS} text blocks")

        for layer in model.vision_model.encoder.layers[-config.NUM_LAST_VISUAL_BLOCKS:]:
            for param in layer.parameters():
                param.requires_grad = True

        for layer in model.text_model.encoder.layers[-config.NUM_LAST_TEXT_BLOCKS:]:
            for param in layer.parameters():
                param.requires_grad = True

        if config.TRAIN_VISUAL_PROJ:
            if hasattr(model, 'visual_projection'):
                model.visual_projection.weight.requires_grad = True
            elif hasattr(model.vision_model, 'head'):
                for param in model.vision_model.head.parameters():
                    param.requires_grad = True
        if config.TRAIN_TEXT_PROJ:
            if hasattr(model, 'text_projection'):
                model.text_projection.weight.requires_grad = True
            elif hasattr(model.text_model, 'head'):
                for param in model.text_model.head.parameters():
                    param.requires_grad = True

        # 冻结 logit_scale / logit_bias：防止 exp(logit_scale) 持续增长导致
        # SigLipLoss 梯度爆炸，压制 SupConLoss 的聚类效果
        model.logit_scale.requires_grad = False
        model.logit_bias.requires_grad  = False

    elif config.FINETUNE_STRATEGY == "all":
        print("\nFine-tuning entire model (all layers)")
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown FINETUNE_STRATEGY: {config.FINETUNE_STRATEGY}")

    print("\nTrainable parameters:")
    trainable_total = 0
    total_params    = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(f"  - {name}")
            trainable_total += param.numel()
    print(f"\nTotal trainable: {trainable_total:,} / Total: {total_params:,} "
          f"({100 * trainable_total / total_params:.1f}%)")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=config.WEIGHT_DECAY
    )

    if config.SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
        )
        print(f"\nUsing Cosine Annealing LR: {config.LEARNING_RATE} -> {config.MIN_LR} "
              f"over {config.EPOCHS} epochs")

    elif config.SCHEDULER_TYPE == "warmup_cosine":
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.EPOCHS - config.WARMUP_EPOCHS,
                    eta_min=config.MIN_LR
                )
            ],
            milestones=[config.WARMUP_EPOCHS]
        )
        print(f"\nUsing Warmup ({config.WARMUP_EPOCHS} epochs) + Cosine Annealing LR")

    elif config.SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.EPOCHS // 3, gamma=0.1
        )

    elif config.SCHEDULER_TYPE == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    else:
        scheduler = None

    loss_fn = SigLipLoss()

    return model, processor, optimizer, scheduler, loss_fn


class SingleViewSigLIPModel(nn.Module):
    """
    Single-view V2: frozen SigLIP + CrossViewAttentionPooler on patch tokens.

    Flow:
      pixel_values [B, C, H, W]
        -> SigLIP vision_model (frozen, no_grad)
        -> last_hidden_state [B, 256, D]
        -> CrossViewAttentionPooler -> [B, D]
        -> L2 normalize
    """

    def __init__(self, base_model, pooler, embed_dim=1152):
        super().__init__()
        self.base_model = base_model
        self.pooler = pooler
        self.config = base_model.config

    def encode(self, pixel_values):
        """
        Encode single-view images into embeddings via patch-level attention pooling.

        Args:
            pixel_values: [B, C, H, W]
        Returns:
            [B, D] L2-normalized embedding
        """
        with torch.no_grad():
            vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state  # [B, 256, D]

        fused = self.pooler(patch_tokens)  # [B, D]
        fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-12)
        return fused


def setup_single_view_model(config):
    """
    Build SingleViewSigLIPModel with frozen SigLIP + trainable attention pooler.

    Returns:
        model, processor, optimizer, scheduler
    """
    from .multiview_models import CrossViewAttentionPooler

    print(f"Loading SigLIP2 base model: {config.SIGLIP_MODEL}")
    print(f"Device: {config.DEVICE}")

    base_model = AutoModel.from_pretrained(config.SIGLIP_MODEL)
    processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)

    embed_dim = base_model.config.vision_config.hidden_size
    print(f"Vision embedding dim: {embed_dim}")

    pooler = CrossViewAttentionPooler(
        embed_dim=embed_dim,
        num_queries=config.NUM_QUERY_TOKENS,
        num_heads=config.POOLER_NUM_HEADS,
        num_layers=config.POOLER_NUM_LAYERS,
        dropout=config.POOLER_DROPOUT,
    )

    model = SingleViewSigLIPModel(base_model, pooler, embed_dim=embed_dim)
    model = model.to(config.DEVICE)

    # Freeze everything, then unfreeze pooler only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.pooler.parameters():
        param.requires_grad = True

    print("\nTrainable parameters:")
    trainable_total = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(f"  - {name} ({param.numel():,})")
            trainable_total += param.numel()

    print(f"\nTotal trainable: {trainable_total:,} / Total: {total_params:,} "
          f"({100 * trainable_total / total_params:.2f}%)")
    print(f"SigLIP: completely frozen (no gradient computation)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=config.WEIGHT_DECAY,
    )
    print(f"\nOptimizer: AdamW, LR={config.LEARNING_RATE:.2e}")

    scheduler = None
    if config.SCHEDULER_TYPE == "warmup_cosine":
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.EPOCHS - config.WARMUP_EPOCHS,
                    eta_min=config.MIN_LR,
                ),
            ],
            milestones=[config.WARMUP_EPOCHS],
        )
        print(f"Using Warmup ({config.WARMUP_EPOCHS} epochs) + Cosine Annealing LR")
    elif config.SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
        )

    return model, processor, optimizer, scheduler


class VisionModelWrapper:
    """
    薄包装器：让 evaluation.py 中 model.vision_model 调用正常工作。
    evaluation.py 期望 model.vision_model 来访问视觉编码器，
    此包装器将独立的 vision_model 适配为这种接口。
    """
    def __init__(self, vision_model):
        self.vision_model = vision_model

    def eval(self):
        self.vision_model.eval()

    def train(self, mode=True):
        self.vision_model.train(mode)

    def state_dict(self):
        # 加上 vision_model. 前缀，与完整模型 key 格式一致
        return {f'vision_model.{k}': v for k, v in self.vision_model.state_dict().items()}

    def load_state_dict(self, state_dict, **kwargs):
        # 兼容有/无 vision_model. 前缀的 state_dict
        sample_key = next(iter(state_dict))
        if sample_key.startswith('vision_model.'):
            state_dict = {k.replace('vision_model.', '', 1): v for k, v in state_dict.items()}
        return self.vision_model.load_state_dict(state_dict, **kwargs)

    def parameters(self):
        return self.vision_model.parameters()

    def named_parameters(self):
        return self.vision_model.named_parameters()

    def to(self, device):
        self.vision_model = self.vision_model.to(device)
        return self


def setup_vision_only_model(config):
    """
    Vision-only 模型设置（ArcFace 训练模式）。
    只加载 SigLIP2 视觉编码器，释放文本端，配合 ArcFace + SupCon loss。
    """
    from .losses import ArcFaceLoss, SupConLoss

    print(f"Loading SigLIP2 vision encoder: {config.SIGLIP_MODEL}")
    print(f"Device: {config.DEVICE}")

    full_model = AutoModel.from_pretrained(config.SIGLIP_MODEL)
    processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)

    # 提取特征维度（从模型配置读取）
    feat_dim = full_model.config.vision_config.hidden_size
    print(f"Vision feature dimension: {feat_dim}")

    # 提取视觉编码器，释放文本端内存
    vision_model = full_model.vision_model.to(config.DEVICE)
    del full_model
    torch.cuda.empty_cache()

    # 冻结全部参数
    for param in vision_model.parameters():
        param.requires_grad = False

    # 解冻最后 N 个 visual blocks
    if config.FINETUNE_STRATEGY in ("last_n_blocks", "proj_only"):
        n_blocks = config.NUM_LAST_VISUAL_BLOCKS
        print(f"\nFine-tuning last {n_blocks} visual blocks")
        for layer in vision_model.encoder.layers[-n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    # 解冻 vision head（投影层）
    if config.TRAIN_VISUAL_PROJ and hasattr(vision_model, 'head'):
        for param in vision_model.head.parameters():
            param.requires_grad = True
        print("Fine-tuning vision_model.head (projection)")

    if config.FINETUNE_STRATEGY == "all":
        print("\nFine-tuning entire vision model (all layers)")
        for param in vision_model.parameters():
            param.requires_grad = True

    # 打印可训练参数
    print("\nTrainable parameters (vision encoder):")
    trainable_total = 0
    total_params = 0
    for name, param in vision_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(f"  - {name}")
            trainable_total += param.numel()
    print(f"\nVision trainable: {trainable_total:,} / Total: {total_params:,} "
          f"({100 * trainable_total / total_params:.1f}%)")

    # 创建 ArcFace loss（含可训练分类头）
    arcface_loss = ArcFaceLoss(
        feat_dim=feat_dim,
        num_classes=config.NUM_CLASSES,
        scale=config.ARCFACE_SCALE,
        margin=config.ARCFACE_MARGIN,
    ).to(config.DEVICE)
    arcface_params = sum(p.numel() for p in arcface_loss.parameters())
    print(f"\nArcFace head parameters: {arcface_params:,} "
          f"(feat_dim={feat_dim}, num_classes={config.NUM_CLASSES})")

    # 创建 SupCon loss
    supcon_loss = SupConLoss(temperature=config.SUPCON_TEMPERATURE)

    # Optimizer：视觉 encoder 用 base LR，ArcFace 头用更高 LR
    head_lr = config.LEARNING_RATE * config.ARCFACE_HEAD_LR_MULT
    param_groups = [
        {
            'params': [p for p in vision_model.parameters() if p.requires_grad],
            'lr': config.LEARNING_RATE,
        },
        {
            'params': list(arcface_loss.parameters()),
            'lr': head_lr,
        },
    ]
    optimizer = optim.Adam(
        param_groups,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"\nOptimizer: Vision LR={config.LEARNING_RATE:.2e}, "
          f"ArcFace Head LR={head_lr:.2e}")

    # Scheduler
    scheduler = None
    if config.SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
        )
    elif config.SCHEDULER_TYPE == "warmup_cosine":
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.EPOCHS - config.WARMUP_EPOCHS,
                    eta_min=config.MIN_LR
                )
            ],
            milestones=[config.WARMUP_EPOCHS]
        )
        print(f"Using Warmup ({config.WARMUP_EPOCHS} epochs) + Cosine Annealing LR")
    elif config.SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.EPOCHS // 3, gamma=0.1
        )

    return vision_model, processor, optimizer, scheduler, arcface_loss, supcon_loss, feat_dim
