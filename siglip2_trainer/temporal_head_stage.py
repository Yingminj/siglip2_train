import os
import re
import json

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image

from .datasets import VideoSequenceDataset, load_label_file
from .losses import SupConLoss


def precompute_video_features(model, processor, video_dir, label_dir,
                               cache_path, config, batch_size=32):
    """
    预提取训练视频的 SigLIP2 特征并缓存。
    cache 格式: {video_name: {'features': [N, D], 'labels': [N]}}
    """
    if os.path.exists(cache_path):
        print(f"  已有特征缓存，跳过提取: {cache_path}")
        return

    model.eval()
    print(f"\n  预提取视频特征 → {cache_path}")

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    cache = {}

    for vid_file in video_files:
        vid_name   = os.path.splitext(vid_file)[0]
        vid_path   = os.path.join(video_dir, vid_file)
        label_path = os.path.join(label_dir, vid_name + '_lable.txt')

        if not os.path.exists(label_path):
            print(f"  ⚠️  跳过 {vid_file}: 未找到标注")
            continue

        label_dict = load_label_file(label_path)

        cap    = cv2.VideoCapture(vid_path)
        frames = []
        labels = []
        fidx   = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            labels.append(label_dict.get(fidx, -1))
            fidx += 1
        cap.release()

        if not frames:
            continue

        all_feats = []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_imgs = frames[i:i + batch_size]
                inputs = processor(images=batch_imgs, return_tensors='pt')
                pv     = inputs['pixel_values'].to(config.DEVICE)
                vo     = model.vision_model(pixel_values=pv)
                feats  = vo.pooler_output                                # [B, D]
                feats  = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())

        cache[vid_name] = {
            'features': torch.cat(all_feats, dim=0),
            'labels':   torch.tensor(labels, dtype=torch.long),
        }
        print(f"    {vid_name}: {len(frames)} 帧  特征 {cache[vid_name]['features'].shape}")

    torch.save(cache, cache_path)
    print(f"  ✓ 特征缓存保存完成: {cache_path}（{len(cache)} 个视频）")
    model.train()


def train_temporal_head(model, head, processor, config):
    """
    Stage 2: 冻结 SigLIP2，训练时序头（LSTM / GRU / Transformer / TCN）。
    使用预提取的视频帧特征（滑动窗口序列）训练，Loss = CE + SupCon。
    """
    _head_type = getattr(config, 'TEMPORAL_HEAD_TYPE', 'lstm').upper()
    if _head_type == 'GRU':
        _dim_info = f"GRU_HIDDEN_DIM={config.GRU_HIDDEN_DIM}"
    elif _head_type == 'TRANSFORMER':
        _dim_info = f"TRANSFORMER_FF_DIM={config.TRANSFORMER_FF_DIM}"
    elif _head_type == 'TCN':
        _dim_info = f"TCN_MODEL_DIM={config.TCN_MODEL_DIM}"
    else:
        _dim_info = f"LSTM_HIDDEN_SIZE={config.LSTM_HIDDEN_SIZE}"
    print(f"\n{'='*60}")
    print(f"【{_head_type} 时序头 — Stage 2 训练】")
    print(f"  视频目录:  {config.TEMPORAL_VIDEO_DIR}")
    print(f"  标注目录:  {config.TEMPORAL_LABEL_DIR}")
    print(f"  序列长度:  {config.SEQ_LEN}  步长: {config.SEQ_STRIDE}")
    print(f"  {_dim_info}  轮数: {config.TEMPORAL_EPOCHS}")
    print(f"{'='*60}")

    # 预提取特征（首次运行耗时，之后复用缓存）
    cache_path = os.path.join(os.path.dirname(config.TEMPORAL_VIDEO_DIR),
                              'siglip2_features_cache.pt')
    precompute_video_features(model, processor,
                               config.TEMPORAL_VIDEO_DIR, config.TEMPORAL_LABEL_DIR,
                               cache_path, config)

    # 构建数据集
    seq_dataset = VideoSequenceDataset(
        features_cache_path=cache_path,
        seq_len=config.SEQ_LEN,
        stride=config.SEQ_STRIDE,
    )

    def seq_collate(batch):
        seq_features = torch.stack([item['seq_features'] for item in batch])  # [B, T, D]
        labels       = torch.stack([item['label']        for item in batch])  # [B]
        return {'seq_features': seq_features, 'labels': labels}

    seq_loader = DataLoader(
        seq_dataset,
        batch_size=config.TEMPORAL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        collate_fn=seq_collate,
    )

    # 冻结 SigLIP2
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    head.train()
    optimizer      = optim.Adam(head.parameters(), lr=config.TEMPORAL_LR)
    scheduler      = optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=config.TEMPORAL_EPOCHS, eta_min=1e-5)
    ce_loss        = nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss(temperature=0.1)

    best_loss  = float('inf')
    best_path  = os.path.join(config.MODEL_DIR, f"{config.TEMPORAL_SAVE_NAME}_best.pt")
    final_path = os.path.join(config.MODEL_DIR, f"{config.TEMPORAL_SAVE_NAME}_final.pt")

    avg_loss = float('inf')
    for epoch in range(config.TEMPORAL_EPOCHS):
        head.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(seq_loader):
            seq_features = batch['seq_features'].to(config.DEVICE)  # [B, T, D]
            labels       = batch['labels'].to(config.DEVICE)        # [B]

            optimizer.zero_grad()
            siglip_last = seq_features[:, -1, :]      # [B, D] 最后一帧 SigLIP 特征
            embedding, logits = head(seq_features)     # [B, D], [B, C]

            # CE Loss：通过 logits 分支训练时序头 + classifier
            ce = ce_loss(logits, labels)

            # 残差融合：α*siglip_last + (1-α)*时序嵌入，保留 SigLIP 判别性
            alpha = config.RESIDUAL_ALPHA
            combined = alpha * siglip_last + (1 - alpha) * embedding
            combined_norm = combined / (combined.norm(dim=-1, keepdim=True) + 1e-12)

            # SupCon 作用于融合后的 embedding，使类别中心与推理时一致
            supcon = supcon_loss_fn(combined_norm, labels)

            loss = ce + config.TEMPORAL_SUPCON_WEIGHT * supcon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            if batch_idx % 20 == 0:
                print(f"  [{_head_type} Epoch {epoch:3d}][Batch {batch_idx:3d}]"
                      f" Loss: {loss.item():.4f}  CE: {ce.item():.4f}  SupCon: {supcon.item():.4f}")

        avg_loss = float(np.mean(epoch_losses))
        scheduler.step()
        print(f"  {_head_type} Epoch {epoch} — Avg Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'lstm_state_dict': head.state_dict(),
                        'loss': avg_loss}, best_path)
            print(f"  ★ 最佳 {_head_type} checkpoint: {best_path} (loss={avg_loss:.4f})")

    torch.save({'epoch': config.TEMPORAL_EPOCHS - 1,
                'lstm_state_dict': head.state_dict(),
                'loss': avg_loss}, final_path)
    print(f"  ✓ {_head_type} 训练完成，最终权重: {final_path}")
    return head


def compute_and_save_temporal_centers(model, head, processor,
                                       train_dataset, config, epoch):
    """
    用训练图片的时序窗口计算时序头类别中心。
    从 siglip2_features_cache.pt 读取已缓存的视频特征，
    用与训练相同的滑动窗口策略，取每类所有窗口的时序嵌入均值作为类别中心。
    中心向量写入 graph_info.json 的 center_feature_siglip2_lstm 字段。
    """
    head.eval()
    _head_type = getattr(config, 'TEMPORAL_HEAD_TYPE', 'lstm').upper()

    print(f"\n{'='*60}")
    print(f"【{_head_type} 类别中心计算（视频时序窗口）】Epoch {epoch}")
    print(f"{'='*60}")

    class_list = sorted(
        train_dataset.classes,
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )

    # 从 image_root 加载各类别训练图片，构造滑动窗口序列计算时序头类别中心。
    # 比从训练视频提取特征更稳定：训练图片与 Stage 1 同源，对测试视频泛化性更好；
    # 同类不同图片构成的序列模拟了推理时稳定状态下的滚动帧缓冲，保留时序特征。
    image_root = train_dataset.image_root
    seq_len    = config.SEQ_LEN
    stride     = config.SEQ_STRIDE
    batch_size = 32

    class_embeddings = {cat: [] for cat in class_list}

    with torch.no_grad():
        for category in class_list:
            category_dir = os.path.join(image_root, category)
            if not os.path.isdir(category_dir):
                print(f"  ⚠️  {category}: 目录不存在，跳过")
                continue
            img_files = sorted([
                f for f in os.listdir(category_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ])
            if not img_files:
                print(f"  ⚠️  {category}: 无图片，跳过")
                continue

            # 批量提取 SigLIP 特征（与 Stage 1 evaluation 使用相同的 pooler_output）
            all_feats = []
            for i in range(0, len(img_files), batch_size):
                batch_imgs = []
                for fname in img_files[i:i + batch_size]:
                    img = Image.open(os.path.join(category_dir, fname)).convert('RGB')
                    img.load()
                    batch_imgs.append(img)
                inputs = processor(images=batch_imgs, return_tensors='pt')
                pv = inputs['pixel_values'].to(config.DEVICE)
                vo = model.vision_model(pixel_values=pv)
                feats = vo.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())
            all_feats = torch.cat(all_feats, dim=0)  # [N, D]

            # 图片不足 seq_len 时有放回采样补足
            if len(all_feats) < seq_len:
                idx = torch.randint(0, len(all_feats), (seq_len,))
                all_feats = all_feats[idx]

            # 滑动窗口 → 时序嵌入 → 残差融合
            for start in range(0, len(all_feats) - seq_len + 1, stride):
                seq = all_feats[start:start + seq_len].unsqueeze(0).to(config.DEVICE)  # [1, T, D]
                embed, _ = head(seq)                                                     # [1, D]
                siglip_last = all_feats[start + seq_len - 1].unsqueeze(0).to(config.DEVICE)
                alpha = config.RESIDUAL_ALPHA
                combined = alpha * siglip_last + (1 - alpha) * embed
                combined = combined / (combined.norm(dim=-1, keepdim=True) + 1e-12)
                class_embeddings[category].append(combined.squeeze(0).cpu())
            print(f"  {category}: {len(img_files)} 张图片 → {len(class_embeddings[category])} 个窗口")

    class_centers = {}
    for category in class_list:
        embeds = class_embeddings[category]
        if not embeds:
            print(f"  ⚠️  {category}: 无有效时序窗口，跳过")
            continue
        stacked = torch.stack(embeds, dim=0).numpy()   # [N, D]
        center  = np.mean(stacked, axis=0)
        center  = center / (np.linalg.norm(center) + 1e-12)
        class_centers[category] = center
        print(f"  {category}: {len(img_files)} 张图片 → {len(class_embeddings[category])} 个embedding向量 → 取均值得类别中心")

    # 写入 graph_info.json
    image_root          = train_dataset.image_root
    original_graph_path = os.path.join(image_root, 'graph_info.json')
    if not os.path.exists(original_graph_path):
        data_dir = os.path.dirname(image_root)
        original_graph_path = os.path.join(data_dir, 'graph_info.json')
    if not os.path.exists(original_graph_path):
        print(f"  ⚠️  未找到 graph_info.json: {original_graph_path}")
        return class_centers

    eval_dir        = os.path.join(config.MODEL_DIR, 'classification_viz')
    os.makedirs(eval_dir, exist_ok=True)
    graph_copy_path = os.path.join(eval_dir, f'graph_info_{_head_type.lower()}_{epoch}.json')

    import shutil
    shutil.copy2(original_graph_path, graph_copy_path)

    with open(graph_copy_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    updated = 0
    for node in graph_data.get('nodes', []):
        node_id = node.get('node_id')
        if node_id is None:
            continue
        try:
            nid_int = int(re.search(r'\d+', str(node_id)).group())
            cat_name = (f"category{nid_int}"
                        if f"category{nid_int}" in class_centers
                        else f"category_{nid_int}")
            if cat_name in class_centers:
                node['center_feature_siglip2_lstm'] = class_centers[cat_name].tolist()
                updated += 1
        except Exception:
            pass

    print(f"  graph_info: 更新 {updated}/{len(graph_data.get('nodes', []))} 个节点")

    # 紧凑格式保存
    temp_data = json.loads(json.dumps(graph_data))
    for node in temp_data.get('nodes', []):
        if 'center_feature_siglip2_lstm' in node:
            node['center_feature_siglip2_lstm'] = json.dumps(
                node['center_feature_siglip2_lstm'], separators=(',', ':'))

    with open(graph_copy_path, 'w', encoding='utf-8') as f:
        json.dump(temp_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 已保存: {graph_copy_path}")

    return class_centers
