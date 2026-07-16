"""
Multi-View SigLIP2 Models (V2: Patch-Level Cross-View Attention Pooling)
========================================================================

Architecture (following PaliGemma/pi0 pattern):
  - SigLIP2 towers frozen, with trainable vision post-LayerNorm
  - Uses last_hidden_state (patch tokens) instead of pooler_output
  - CrossViewAttentionPooler: learnable queries cross-attend to all patch tokens
  - Learnable or text-initialized class queries gate image-token attention
  - View position embeddings distinguish camera origins

Modules:
  - MultiViewSimpleDataset: splits 2x2 grid images into 4 views
  - CrossViewAttentionPooler: query-based cross-attention fusion
  - MultiViewSigLIPModel: frozen SigLIP + trainable pooler
  - setup_multiview_model: model construction and optimizer
"""

import os

import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .datasets import SimpleDataset
from .augmentation import SigLIPAugmentation


# =============================================================================
# Multi-View Dataset
# =============================================================================

class MultiViewSimpleDataset(SimpleDataset):
    """
    Extends SimpleDataset to split composite images into individual views.

    Supported layouts (auto-detected by aspect ratio):
      - 1x3 horizontal (1920x480): 3 views of 640x480 [left_wrist, right_wrist, head]
      - 2x2 grid (1920x1488): 4 views of 960x744

    Text loading is skipped since text data is not used in V2.
    """

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load composite image
        with Image.open(sample['image_path']) as img_file:
            image = img_file.convert('RGB')
            image.load()

        views = split_image_to_views(image)

        # Apply augmentation to each view independently
        if self.augment_fn is not None:
            views = [self.augment_fn(v) for v in views]

        return {
            'views': views,
            'label': self.class_to_idx[sample['class']],
            'class': sample['class'],
        }


def split_image_to_views(image):
    """
    Split a composite PIL Image into view crops.
    Auto-detects layout by aspect ratio:
      - width/height > 3.0: 1x3 horizontal (1920x480 → 3 views of 640x480)
      - otherwise: 2x2 grid (1920x1488 → 4 views of 960x744)
    """
    w, h = image.size
    if w / h > 3.0:
        # 1x3 horizontal: [head | wrist_left | wrist_right]  (index 0=head)
        view_w = w // 3
        return [
            image.crop((0, 0, view_w, h)),
            image.crop((view_w, 0, 2 * view_w, h)),
            image.crop((2 * view_w, 0, w, h)),
        ]
    else:
        # 2x2 grid
        half_w, half_h = w // 2, h // 2
        return [
            image.crop((0, 0, half_w, half_h)),
            image.crop((half_w, 0, w, half_h)),
            image.crop((0, half_h, half_w, h)),
            image.crop((half_w, half_h, w, h)),
    ]


def make_multiview_collate_fn(processor, config):
    """
    Collate function for MultiViewSimpleDataset.
    Flattens 4 views per sample into a single batch of images.
    No text processing (V2: SupCon only).
    """
    def collate_fn(batch):
        # Flatten views: 4 views per sample -> 4*B images
        all_views = []
        for item in batch:
            all_views.extend(item['views'])

        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        # Process images (4*B) through processor
        image_inputs = processor(images=all_views, return_tensors="pt")

        result = {k: v for k, v in image_inputs.items()}
        result['labels'] = labels

        return result

    return collate_fn


# =============================================================================
# Cross-View Attention Pooler (replaces MultiViewFusion from V1)
# =============================================================================

class CrossViewAttentionPooler(nn.Module):
    """
    Cross-attention pooler that aggregates 1024 patch tokens into a single embedding.

    Architecture:
      - num_queries learnable query tokens [Q, D]
      - Multi-layer cross-attention: queries attend to all patch tokens (K, V)
      - Mean pool over queries -> single vector
      - LayerNorm for output stability

    Input:  [B, N_patches, D]  (e.g., 4 views x 256 patches = 1024 tokens)
    Output: [B, D]
    """

    def __init__(self, embed_dim=1152, num_queries=8, num_heads=8,
                 num_layers=2, dropout=0.1, num_views=3, view_bias_init=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_views = num_views

        # Learnable per-view attention bias (logit space).
        # Added to cross-attention logits so some views get more attention mass.
        # view_bias_init e.g. [1.0, 0.0, 0.0] -> head (index 0) starts ~e^1 stronger.
        if view_bias_init is None:
            init = torch.zeros(num_views)
        else:
            init = torch.tensor(view_bias_init, dtype=torch.float32)
            assert init.numel() == num_views, \
                f"view_bias_init length {init.numel()} != num_views {num_views}"
        self.view_logits = nn.Parameter(init)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
            }))

        # Output normalization
        self.output_ln = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer['cross_attn'].in_proj_weight)
            nn.init.xavier_uniform_(layer['ffn'][0].weight)
            nn.init.xavier_uniform_(layer['ffn'][3].weight)

    def forward(self, x):
        """
        Args:
            x: [B, N_patches, D] concatenated patch tokens from all views
        Returns:
            [B, D] fused embedding
        """
        B, S, _ = x.shape

        # Per-view attention bias: expand [num_views] -> [S] -> [Q, S] (additive mask)
        num_patches = S // self.num_views
        key_bias = self.view_logits.repeat_interleave(num_patches)        # [S]
        attn_mask = key_bias.unsqueeze(0).expand(self.num_queries, S)     # [Q, S]

        # Expand queries to batch size
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Q, D]

        # Cross-attention layers (pre-norm style)
        for layer in self.layers:
            # Cross-attention: queries attend to patch tokens
            residual = queries
            queries = layer['norm1'](queries)
            queries = layer['cross_attn'](
                query=queries,
                key=x,
                value=x,
                attn_mask=attn_mask,
            )[0] + residual

            # FFN
            residual = queries
            queries = layer['norm2'](queries)
            queries = layer['ffn'](queries) + residual

        # Mean pool over query tokens -> [B, D]
        output = queries.mean(dim=1)

        # Output LayerNorm
        output = self.output_ln(output)

        return output


class TextConditionedGatedCrossAttention(nn.Module):
    """Score every candidate class query against multi-view image tokens.

    Class embeddings are queries and image patch tokens are keys/values.
    Every image is compared with every class query, so the ground-truth label is
    never injected into the input.  The gate follows the idea used by GenLIP,
    but is kept separate from q_proj so pretrained SigLIP weights remain intact.
    """

    def __init__(self, embed_dim=1152, num_heads=8, dropout=0.1,
                 gate_bias_init=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.3025851))  # ln(10)

        self._init_weights(gate_bias_init)

    def _init_weights(self, gate_bias_init):
        # Identity Q/K/V initialization starts from dot-product matching in the
        # original SigLIP embedding space instead of a fully random projection.
        with torch.no_grad():
            eye = torch.eye(self.embed_dim)
            self.cross_attn.in_proj_weight.zero_()
            self.cross_attn.in_proj_weight[:self.embed_dim].copy_(eye)
            self.cross_attn.in_proj_weight[self.embed_dim:2 * self.embed_dim].copy_(eye)
            self.cross_attn.in_proj_weight[2 * self.embed_dim:].copy_(eye)
            self.cross_attn.in_proj_bias.zero_()
            self.cross_attn.out_proj.weight.copy_(eye)
            self.cross_attn.out_proj.bias.zero_()

        # sigmoid(4) ~= 0.982: the gate initially behaves almost like identity.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    def forward(self, image_tokens, class_text_queries, view_logits=None,
                num_views=1):
        """
        Args:
            image_tokens: [B, S, D]
            class_text_queries: [C, D], one cached query per class
            view_logits: optional learnable per-view attention bias [V]
        Returns:
            class_logits: [B, C]
            conditioned_queries: [B, C, D]
        """
        batch_size, seq_len, _ = image_tokens.shape
        num_classes = class_text_queries.shape[0]

        queries = class_text_queries.unsqueeze(0).expand(batch_size, -1, -1)
        normalized_queries = self.query_norm(queries)
        normalized_images = self.image_norm(image_tokens)

        attn_mask = None
        if view_logits is not None:
            if seq_len % num_views != 0:
                raise ValueError(
                    f"Image token count {seq_len} is not divisible by num_views={num_views}"
                )
            patches_per_view = seq_len // num_views
            key_bias = view_logits.repeat_interleave(patches_per_view)
            attn_mask = key_bias.unsqueeze(0).expand(num_classes, seq_len)

        attended, _ = self.cross_attn(
            query=normalized_queries,
            key=normalized_images,
            value=normalized_images,
            attn_mask=attn_mask,
            need_weights=False,
        )

        gate = torch.sigmoid(self.gate_proj(normalized_queries))
        gated_attended = gate * attended
        normalized_attended = self.output_norm(gated_attended)
        conditioned = queries + normalized_attended

        # Compatibility score for every image/class pair. Clamp the learned
        # scale as done in CLIP-like models to prevent unstable logits.
        class_logits = F.cosine_similarity(
            F.normalize(normalized_attended, dim=-1),
            F.normalize(queries, dim=-1),
            dim=-1,
        ) * self.logit_scale.exp().clamp(max=100.0)

        return class_logits, conditioned


# =============================================================================
# Multi-View SigLIP2 Model Wrapper (V2)
# =============================================================================

class MultiViewSigLIPModel(nn.Module):
    """
    Wraps frozen SigLIP2 + CrossViewAttentionPooler.

    Flow:
      pixel_values [4B, C, H, W]
        -> SigLIP vision_model (frozen, no_grad)
        -> last_hidden_state [4B, 256, D]
        -> reshape [B, 4, 256, D]
        -> + view_position_embedding [4, 1, D]
        -> reshape [B, 1024, D]
        -> CrossViewAttentionPooler -> [B, D]
        -> L2 normalize
    """

    def __init__(self, base_model, pooler, num_views=4, embed_dim=1152,
                 text_fusion=None, class_query_mode="text", num_classes=None):
        super().__init__()
        self.base_model = base_model
        self.pooler = pooler
        self.text_fusion = text_fusion or TextConditionedGatedCrossAttention(
            embed_dim=embed_dim,
        )
        self.num_views = num_views
        self.class_names = []
        self.class_query_mode = class_query_mode

        if class_query_mode == "learnable":
            if not num_classes or num_classes < 1:
                raise ValueError("num_classes must be positive for learnable class queries")
            # Keep the historical state-dict key for checkpoint/test compatibility.
            self.class_text_queries = nn.Parameter(
                torch.empty(num_classes, embed_dim)
            )
            nn.init.trunc_normal_(self.class_text_queries, std=0.02)
        elif class_query_mode == "text":
            # Filled once by set_class_prompts(). Persistent so checkpoints retain
            # the exact frozen text queries used during training and inference.
            self.register_buffer(
                'class_text_queries',
                torch.empty(0, embed_dim),
                persistent=True,
            )
        else:
            raise ValueError(
                f"class_query_mode must be 'learnable' or 'text', got {class_query_mode!r}"
            )

        # View position embedding: [num_views, 1, D]
        # Broadcast-added to each view's 256 patch tokens
        self.view_pos_embed = nn.Parameter(
            torch.zeros(num_views, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.view_pos_embed, std=0.02)

        # Expose config for compatibility
        self.config = base_model.config

    def get_extra_state(self):
        """Persist prompt order/text alongside tensor weights in checkpoints."""
        return {
            'class_names': self.class_names,
            'class_query_mode': self.class_query_mode,
            'resolved_class_prompts': getattr(self, 'resolved_class_prompts', {}),
        }

    def set_extra_state(self, state):
        state = state or {}
        self.class_names = list(state.get('class_names', []))
        self.loaded_class_query_mode = state.get('class_query_mode')
        self.resolved_class_prompts = dict(state.get('resolved_class_prompts', {}))

    def set_learnable_class_names(self, class_names):
        """Bind dataset label order to the trainable class-query rows."""
        if self.class_query_mode != "learnable":
            raise RuntimeError("set_learnable_class_names requires learnable query mode")
        if len(class_names) != self.class_text_queries.shape[0]:
            raise ValueError(
                f"Configured {self.class_text_queries.shape[0]} learnable queries, "
                f"but dataset has {len(class_names)} classes"
            )
        self.class_names = list(class_names)
        self.resolved_class_prompts = {}
        print(f"Initialized {len(self.class_names)} learnable class queries in order: "
              f"{self.class_names}")

    @torch.no_grad()
    def set_class_prompts(self, processor, class_names, class_prompts=None,
                          max_length=64):
        """Encode and cache candidate class prompts with frozen SigLIP text tower.

        ``class_prompts`` maps a class name to either one string or several
        prompt variants. Multiple variants are averaged into one class query.
        Missing entries fall back to the class name and emit a warning.
        """
        if self.class_query_mode != "text":
            raise RuntimeError("set_class_prompts requires text query mode")
        class_prompts = class_prompts or {}
        tokenizer = getattr(processor, 'tokenizer', processor)
        device = next(self.base_model.parameters()).device
        was_training = self.base_model.text_model.training
        self.base_model.text_model.eval()

        cached_queries = []
        resolved_prompts = {}
        for class_name in class_names:
            prompts = class_prompts.get(class_name)
            if prompts is None:
                prompts = [f"an image of {class_name}"]
                print(f"  Warning: no CLASS_PROMPTS entry for {class_name!r}; "
                      f"using {prompts[0]!r}")
            elif isinstance(prompts, str):
                prompts = [prompts]
            else:
                prompts = list(prompts)

            if not prompts or any(not str(prompt).strip() for prompt in prompts):
                raise ValueError(f"CLASS_PROMPTS[{class_name!r}] must contain non-empty text")

            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            text_outputs = self.base_model.text_model(
                input_ids=tokens['input_ids'].to(device),
                attention_mask=tokens.get('attention_mask', None).to(device)
                if tokens.get('attention_mask', None) is not None else None,
            )
            query = F.normalize(text_outputs.pooler_output.float(), dim=-1).mean(dim=0)
            cached_queries.append(F.normalize(query, dim=-1))
            resolved_prompts[class_name] = prompts

        self.class_text_queries = torch.stack(cached_queries).to(
            device=device,
            dtype=next(self.text_fusion.parameters()).dtype,
        )
        self.class_names = list(class_names)
        self.resolved_class_prompts = resolved_prompts

        if was_training:
            self.base_model.text_model.train()

        print(f"Cached {len(self.class_names)} class text queries in order: "
              f"{self.class_names}")

    def encode_views(self, pixel_values, return_patch_tokens=False):
        """
        Encode multi-view images and fuse into a single embedding.

        Args:
            pixel_values: [B*num_views, C, H, W]
        Returns:
            [B, D] L2-normalized fused embedding
        """
        # Most of SigLIP stays frozen, but post_layernorm is trainable.  Do not
        # wrap this forward pass in torch.no_grad(), otherwise that LayerNorm
        # cannot receive gradients.
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        patch_tokens = vision_outputs.last_hidden_state  # [B*V, 256, D]

        B_total = patch_tokens.shape[0]
        B = B_total // self.num_views
        num_patches = patch_tokens.shape[1]  # 256
        D = patch_tokens.shape[2]

        # Reshape: [B*V, 256, D] -> [B, V, 256, D]
        patch_tokens = patch_tokens.view(B, self.num_views, num_patches, D)

        # Add view position embedding: [V, 1, D] broadcasts to [B, V, 256, D]
        patch_tokens = patch_tokens + self.view_pos_embed.unsqueeze(0)

        # Flatten views: [B, V, 256, D] -> [B, V*256, D] = [B, 1024, D]
        patch_tokens = patch_tokens.view(B, self.num_views * num_patches, D)

        # Cross-attention pooling
        fused = self.pooler(patch_tokens)  # [B, D]

        # L2 normalize final output
        fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-12)
        if return_patch_tokens:
            return fused, patch_tokens
        return fused

    def forward(self, pixel_values):
        if self.class_text_queries.numel() == 0:
            raise RuntimeError(
                "Class queries are not initialized. Configure learnable queries "
                "or call model.set_class_prompts(...) before training/inference."
            )

        image_embeds, patch_tokens = self.encode_views(
            pixel_values,
            return_patch_tokens=True,
        )
        class_logits, _ = self.text_fusion(
            image_tokens=patch_tokens,
            class_text_queries=self.class_text_queries,
            view_logits=self.pooler.view_logits,
            num_views=self.num_views,
        )
        return image_embeds, class_logits


# =============================================================================
# Model Setup
# =============================================================================

def setup_multiview_model(config):
    """
    Build MultiViewSigLIPModel with frozen SigLIP + trainable pooler.

    Returns:
        model, processor, optimizer, scheduler
    """
    print(f"Loading SigLIP2 base model: {config.SIGLIP_MODEL}")
    print(f"Device: {config.DEVICE}")
    print(f"Multi-View: {config.NUM_VIEWS} views")

    base_model = AutoModel.from_pretrained(config.SIGLIP_MODEL)
    processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)

    # Get embedding dimension from model config
    embed_dim = base_model.config.vision_config.hidden_size
    print(f"Vision embedding dim: {embed_dim}")

    # Create cross-attention pooler
    pooler = CrossViewAttentionPooler(
        embed_dim=embed_dim,
        num_queries=config.NUM_QUERY_TOKENS,
        num_heads=config.POOLER_NUM_HEADS,
        num_layers=config.POOLER_NUM_LAYERS,
        dropout=config.POOLER_DROPOUT,
        num_views=config.NUM_VIEWS,
        view_bias_init=getattr(config, 'VIEW_BIAS_INIT', None),
    )
    text_fusion = TextConditionedGatedCrossAttention(
        embed_dim=embed_dim,
        num_heads=config.TEXT_FUSION_NUM_HEADS,
        dropout=config.TEXT_FUSION_DROPOUT,
        gate_bias_init=config.TEXT_GATE_BIAS_INIT,
    )
    if getattr(config, 'VIEW_BIAS_INIT', None) is not None:
        print(f"View attention bias (learnable, init): {config.VIEW_BIAS_INIT} "
              f"for views {config.VIEW_NAMES}")

    # Wrap into multi-view model
    model = MultiViewSigLIPModel(
        base_model, pooler,
        num_views=config.NUM_VIEWS,
        embed_dim=embed_dim,
        text_fusion=text_fusion,
        class_query_mode=getattr(config, 'CLASS_QUERY_MODE', 'text'),
        num_classes=config.NUM_CLASSES,
    )
    model = model.to(config.DEVICE)

    # ===== Freeze strategy: SigLIP backbone frozen, output LayerNorm trainable =====
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze trainable modules: view position, image pooler, class-query
    # fusion, and the small pretrained vision output LayerNorm.
    model.view_pos_embed.requires_grad = True
    for param in model.pooler.parameters():
        param.requires_grad = True
    for param in model.text_fusion.parameters():
        param.requires_grad = True
    if isinstance(model.class_text_queries, nn.Parameter):
        model.class_text_queries.requires_grad = True
    for param in model.base_model.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    # Print trainable parameters
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
    print("SigLIP: backbone frozen; vision post_layernorm trainable")

    # ===== Optimizer (separate LRs for new modules and pretrained LayerNorm) =====
    pooler_params = (
        list(model.pooler.parameters())
        + list(model.text_fusion.parameters())
        + [model.view_pos_embed]
    )
    if isinstance(model.class_text_queries, nn.Parameter):
        pooler_params.append(model.class_text_queries)
    vision_norm_params = list(model.base_model.vision_model.post_layernorm.parameters())
    optimizer = optim.AdamW(
        [
            {
                'params': pooler_params,
                'lr': config.POOLER_LEARNING_RATE,
                'name': 'multiview',
            },
            {
                'params': vision_norm_params,
                'lr': config.VISION_NORM_LEARNING_RATE,
                'name': 'vision_post_layernorm',
            },
        ],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=config.WEIGHT_DECAY,
    )
    print(f"\nOptimizer: AdamW, pooler LR={config.POOLER_LEARNING_RATE:.2e}, "
          f"vision norm LR={config.VISION_NORM_LEARNING_RATE:.2e}")

    # ===== Scheduler =====
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


def load_multiview_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint for MultiViewSigLIPModel."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Model weights loaded (base_model + pooler)")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Optimizer state loaded")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  Scheduler state loaded")

    info = {'start_epoch': 0, 'previous_loss': None}

    if 'epoch' in checkpoint:
        info['start_epoch'] = checkpoint['epoch'] + 1
        print(f"  Resume from Epoch {info['start_epoch']}")
    if 'loss' in checkpoint:
        info['previous_loss'] = checkpoint['loss']
        print(f"  Previous loss: {checkpoint['loss']:.4f}")

    return info
