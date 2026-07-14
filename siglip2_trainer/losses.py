import math

import torch
import torch.nn.functional as F
from torch import nn


class SigLipLoss(nn.Module):
    def forward(self, image_embeds, text_embeds, logit_scale, logit_bias,
                class_labels=None, smoothing=0.0):
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

        logits = image_embeds @ text_embeds.T
        logits = logits * logit_scale.exp() + logit_bias

        N      = logits.shape[0]

        # 如果提供了 class_labels，使用基于类别的标签矩阵
        # 这样可以正确处理 BATCH_SIZE > 1 时的情况
        if class_labels is not None:
            labels_row = class_labels.unsqueeze(1)
            labels_col = class_labels.unsqueeze(0)
            # smoothing: 将严格 ±1 软化为 ±(1-smoothing)，防止过拟合
            pos_val = 1.0 - smoothing
            neg_val = -1.0 + smoothing
            labels = torch.where(labels_row == labels_col, pos_val, neg_val).to(logits.device)
        else:
            # 兼容旧代码：如果没提供 class_labels，使用对角矩阵
            labels = 2 * torch.eye(N, device=logits.device) - 1

        loss = -nn.functional.logsigmoid(labels * logits).mean()

        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    直接在图像特征空间上施加监督对比学习：同类图像拉近、不同类推远。
    与推理阶段（最近类别中心匹配）目标高度一致。
    参考：Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [N, D]，image_embeds（L2归一化前或后均可，内部会再归一化）
            labels:   [N]，类别整数标签
        Returns:
            scalar loss
        """
        device = features.device
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-12)
        N = features.shape[0]

        # 相似度矩阵 [N, N]，除以温度
        sim = torch.matmul(features, features.T) / self.temperature

        # 正样本 mask：同类且非自身
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)
        self_mask = torch.eye(N, device=device)
        pos_mask = pos_mask - self_mask

        # 数值稳定：每行减去最大值
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # log-softmax 分母（排除自身）
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # 只对 batch 内有正样本的锚点计算损失
        num_pos = pos_mask.sum(dim=1)
        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(pos_mask * log_prob).sum(dim=1)
        loss = loss[valid] / num_pos[valid]
        return loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    (Deng et al., CVPR 2019)

    在 softmax 的角度空间中添加加性 margin m，强制同类特征在超球面上更紧凑，
    类间保持至少 m 的角度间距。

    训练时作为分类 loss 使用；推理时丢弃本模块，只用 backbone 特征做类中心匹配。
    """
    def __init__(self, feat_dim, num_classes, scale=30.0, margin=0.5):
        """
        Args:
            feat_dim:    特征维度（需与 vision encoder 输出维度一致）
            num_classes: 类别数
            scale:       缩放因子 s（典型值 30-64）
            margin:      角度间距 m（弧度，0.5 rad ≈ 28.6°）
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        # 分类权重矩阵（每行代表一个类的原型向量）
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        # 预计算 margin 相关常量
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # easy margin 阈值：cos(pi - m)
        self.th = math.cos(math.pi - margin)
        # easy margin 补偿项：sin(pi - m) * m
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels):
        """
        Args:
            features: [N, D]，L2 归一化的图像特征
            labels:   [N]，类别整数标签
        Returns:
            scalar loss
        """
        # L2 归一化 features 和 weights
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # cos(θ) = features @ weight^T  →  [N, num_classes]
        cosine = F.linear(features, weight)

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # easy margin: 当 cos(θ) < cos(π-m) 时，用线性惩罚替代，避免数值问题
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 只对 ground-truth 类应用 margin
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        # 缩放并计算 cross-entropy
        logits = logits * self.scale
        loss = F.cross_entropy(logits, labels)
        return loss
