"""model_contradiction.py — IntentContradictionNet

创新型多模态网络，用于小样本语用学意图识别（反讽 / 深层疑问检测）。
数据集特征: 文本 + 音频（2-3 秒），仅 ~4000 样本，必须严格防范过拟合。

架构路径:
    OT 软对齐  →  矛盾特征提取  →  矛盾路由 MoE (多尺度低秩 + 正交约束)
    →  多任务 + 对比学习

模块列表:
    1. OTAlignmentLayer         — 异构空间共享投影 + Log-domain Sinkhorn OT 对齐
    2. ContradictionRouter      — 基于模态差异的 MoE 路由
    3. LowRankConvExpert        — 单个低秩瓶颈 1D-Conv 专家
    4. MultiScaleMoEExperts     — 3 个多尺度专家组 (kernel 1/3/5)
    5. compute_expert_orthogonality_loss  — 专家正交惩罚函数
    6. SupConLoss               — 监督对比损失 (Khosla et al., 2020)
    7. AttentionPooling         — 可学习的注意力加权池化
    8. IntentContradictionNet   — 主模型（组装以上所有模块）
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model

from dataloader import PROSODY_FEAT_DIM


# ============================================================================
#  1. 异构空间共享投影 & OT 对齐层 (Optimal Transport Alignment Layer)
# ============================================================================
class OTAlignmentLayer(nn.Module):
    """将文本和音频特征通过 **同一组** 共享投影映射到绝对同构的语义空间，
    再用带熵正则化的对数域 Sinkhorn 算法计算最优传输对齐矩阵。

    输入:
        T ∈ R^{B × L_t × d}   文本序列特征
        A ∈ R^{B × L_a × d}   音频序列特征
    输出:
        T'        ∈ R^{B × L_t × d}   共享投影后的文本特征
        A_aligned ∈ R^{B × L_t × d}   OT 对齐后的音频特征
        S         ∈ R^{B × L_t}       Token 级别点积相似度
    """

    def __init__(
        self,
        d_model: int,
        sinkhorn_iters: int = 10,
        sinkhorn_epsilon: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 共享投影: 用同一组 LayerNorm + Linear 投影两种模态
        self.shared_norm = nn.LayerNorm(d_model)
        self.shared_linear = nn.Linear(d_model, d_model)
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = sinkhorn_epsilon
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    #  Log-domain Sinkhorn (带 NaN 异常保护)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_sinkhorn(
        cost: torch.Tensor,
        epsilon: float,
        num_iters: int,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对数域 Sinkhorn 算法，附带完备的数值下溢 / NaN 异常保护。

        Args:
            cost:       [B, L_t, L_a]  代价矩阵 (余弦距离，值域 [0, 2])
            epsilon:    熵正则化系数 (> 0)
            num_iters:  Sinkhorn 迭代次数
            text_mask:  [B, L_t]  1 = 有效 token, 0 = padding
            audio_mask: [B, L_a]  1 = 有效 frame, 0 = padding

        Returns:
            pi: [B, L_t, L_a]  最优传输计划矩阵 (行和 ≈ μ, 列和 ≈ ν)
        """
        B, M, N = cost.shape
        device = cost.device

        # —— 构造边际分布 (log 域) ——
        # 有效位置均分质量；padding 位置质量 → 0（log → -∞）
        if text_mask is not None:
            text_lengths = text_mask.sum(dim=-1, keepdim=True).clamp(min=1)   # [B, 1]
            mu = text_mask.float() / text_lengths                             # [B, L_t]
            log_mu = torch.log(mu + 1e-20)                                    # [B, L_t]
        else:
            log_mu = torch.full((B, M), -math.log(M), device=device)          # [B, L_t]

        if audio_mask is not None:
            audio_lengths = audio_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
            nu = audio_mask.float() / audio_lengths                            # [B, L_a]
            log_nu = torch.log(nu + 1e-20)                                     # [B, L_a]
        else:
            log_nu = torch.full((B, N), -math.log(N), device=device)           # [B, L_a]

        # —— 对数核 log K_{ij} = -C_{ij} / ε ——
        log_K = -cost / (epsilon + 1e-8)  # [B, L_t, L_a]

        # —— 初始化对偶变量 ——
        log_u = torch.zeros(B, M, device=device)  # [B, L_t]
        log_v = torch.zeros(B, N, device=device)  # [B, L_a]

        for _ in range(num_iters):
            # --- 更新 log_u ---
            # log_u_i = log_μ_i − logsumexp_j(log_K_{ij} + log_v_j)
            log_u = log_mu - torch.logsumexp(
                log_K + log_v.unsqueeze(1),      # [B, L_t, L_a]  broadcast log_v
                dim=-1,                           # sum over L_a
            )  # → [B, L_t]
            # ★ NaN / Inf 异常保护 ★
            log_u = torch.clamp(log_u, min=-1e9, max=1e9)
            log_u = torch.nan_to_num(log_u, nan=0.0, posinf=1e9, neginf=-1e9)

            # --- 更新 log_v ---
            # log_v_j = log_ν_j − logsumexp_i(log_K_{ij} + log_u_i)
            log_v = log_nu - torch.logsumexp(
                log_K.transpose(-2, -1) + log_u.unsqueeze(1),  # [B, L_a, L_t]
                dim=-1,                                         # sum over L_t
            )  # → [B, L_a]
            log_v = torch.clamp(log_v, min=-1e9, max=1e9)
            log_v = torch.nan_to_num(log_v, nan=0.0, posinf=1e9, neginf=-1e9)

        # —— 传输计划 π_{ij} = exp(log_u_i + log_K_{ij} + log_v_j) ——
        log_pi = (
            log_u.unsqueeze(-1)   # [B, L_t, 1]
            + log_K               # [B, L_t, L_a]
            + log_v.unsqueeze(-2) # [B, 1,   L_a]
        )  # → [B, L_t, L_a]
        pi = torch.exp(log_pi)  # [B, L_t, L_a]

        # ★ 最终安全检查 ★
        pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)

        return pi

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_feat:  [B, L_t, d]  文本序列特征 (初始投影后)
            audio_feat: [B, L_a, d]  音频序列特征 (初始投影后)
            text_mask:  [B, L_t]     文本 attention mask
            audio_mask: [B, L_a]     音频 attention mask (降采样后)

        Returns:
            T_prime:    [B, L_t, d]  共享投影后的文本特征
            A_aligned:  [B, L_t, d]  OT 对齐后的音频特征 (长度与文本对齐)
            S:          [B, L_t]     Token 级别的点积相似度序列
        """
        # —— 共享投影: LayerNorm → Linear (同一组权重投影两种模态) ——
        T_prime = self.shared_linear(self.shared_norm(text_feat))   # [B, L_t, d]
        A_prime = self.shared_linear(self.shared_norm(audio_feat))  # [B, L_a, d]
        T_prime = self.dropout(T_prime)
        A_prime = self.dropout(A_prime)

        # —— 余弦代价矩阵  C_{ij} = 1 − cos(T'_i, A'_j) ——
        T_norm = F.normalize(T_prime, p=2, dim=-1)  # [B, L_t, d]
        A_norm = F.normalize(A_prime, p=2, dim=-1)  # [B, L_a, d]
        cosine_sim = torch.bmm(
            T_norm, A_norm.transpose(-2, -1)         # [B, L_t, d] × [B, d, L_a]
        )  # → [B, L_t, L_a]
        cost = 1.0 - cosine_sim                      # [B, L_t, L_a], 值域 [0, 2]

        # —— Sinkhorn 求解 OT 传输计划 ——
        pi = self._log_sinkhorn(
            cost, self.epsilon, self.sinkhorn_iters,
            text_mask=text_mask, audio_mask=audio_mask,
        )  # [B, L_t, L_a]

        # —— 对齐后的音频特征  A_aligned = π × A' ——
        A_aligned = torch.bmm(pi, A_prime)  # [B, L_t, L_a] × [B, L_a, d] → [B, L_t, d]

        # —— Token 级别点积相似度  S_i = <T'_i, A_aligned_i> ——
        S = (T_prime * A_aligned).sum(dim=-1)  # [B, L_t]

        return T_prime, A_aligned, S


# ============================================================================
#  2. 矛盾路由器 (Contradiction Router)
# ============================================================================
class ContradictionRouter(nn.Module):
    """基于「模态差异」(而非输入特征) 的 MoE Router。

    与常规 Router 的根本区别：路由依据不是原始特征，而是 OT 对齐后的
    「文本-音频」矛盾信号（相似度 S + 特征残差 |T'−A_aligned|）。

    输入:
        S:        [B, L_t]     Token 级别相似度
        residual: [B, L_t, d]  特征残差 |T' − A_aligned|
    输出:
        G: [B, L_t, N]         N 个专家的 Softmax 归一化权重
    """

    def __init__(self, d_model: int, num_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        # 路由输入维度 = d_model (残差向量) + 1 (相似度标量)
        router_input_dim = d_model + 1
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_experts),
        )

    def forward(self, S: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S:        [B, L_t]     Token 级别的点积相似度
            residual: [B, L_t, d]  特征残差 |T' − A_aligned|

        Returns:
            G: [B, L_t, num_experts]  Softmax 归一化的路由权重
        """
        # 拼接: 相似度标量 + 残差向量 → [B, L_t, d+1]
        router_input = torch.cat(
            [S.unsqueeze(-1), residual],  # [B, L_t, 1] ⊕ [B, L_t, d]
            dim=-1,
        )  # → [B, L_t, d+1]

        logits = self.router(router_input)    # [B, L_t, num_experts]
        G = F.softmax(logits, dim=-1)         # [B, L_t, num_experts]

        return G


# ============================================================================
#  3. 低秩卷积专家 (Low-Rank Conv Expert)
# ============================================================================
class LowRankConvExpert(nn.Module):
    """低秩瓶颈 + 1D 卷积的单个专家。

    结构:  Linear(d → rank) → 1D-Conv(rank, rank, k) → Linear(rank → d)
    通过极低的 rank（例如 d//4 = 32）大幅压缩参数量，
    有效防止在 ~4000 样本上过拟合。
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2  # "same" padding

        self.down_proj = nn.Linear(d_model, rank)   # 降维  d → rank
        self.conv = nn.Conv1d(
            rank, rank, kernel_size,
            padding=padding, bias=True,
        )                                            # 1D 时序卷积 (在 rank 空间)
        self.up_proj = nn.Linear(rank, d_model)     # 升维  rank → d

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d]  输入特征序列

        Returns:
            out: [B, L, d]  专家输出 (含残差连接)
        """
        # ---- 降维 ----
        h = self.down_proj(x)   # [B, L, d] → [B, L, rank]
        h = self.act(h)

        # ---- 1D Conv (需要 [B, C, L] 格式) ----
        h = h.transpose(1, 2)   # [B, L, rank] → [B, rank, L]
        h = self.conv(h)        # [B, rank, L] → [B, rank, L]
        h = self.act(h)
        h = h.transpose(1, 2)   # [B, rank, L] → [B, L, rank]

        # ---- 升维 ----
        h = self.up_proj(h)     # [B, L, rank] → [B, L, d]
        h = self.dropout(h)

        # ---- 残差连接 + LayerNorm ----
        out = self.norm(h + x)  # [B, L, d]

        return out


# ============================================================================
#  4. 多尺度低秩卷积专家组 (Multi-scale Low-Rank Conv Experts)
# ============================================================================
class MultiScaleMoEExperts(nn.Module):
    """包含 3 个不同卷积核尺度的低秩专家:

    - Expert 0: kernel_size = 1  (音素级，捕捉极短突变)
    - Expert 1: kernel_size = 3  (词组级，捕捉短语节奏模式)
    - Expert 2: kernel_size = 5  (半句级，捕捉全局语速拖延)

    通过矛盾路由器输出的权重 G 进行加权融合。
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        self.experts = nn.ModuleList([
            LowRankConvExpert(d_model, rank, ks, dropout)
            for ks in kernel_sizes
        ])

    def forward(
        self, x: torch.Tensor, gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:             [B, L, d]            矛盾残差特征序列
            gate_weights:  [B, L, num_experts]  路由权重 (来自 ContradictionRouter)

        Returns:
            out: [B, L, d]  加权融合后的专家输出
        """
        # 每个专家独立处理输入
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],  # list of [B, L, d]
            dim=-2,                                   # → [B, L, num_experts, d]
        )

        # 路由权重扩展到特征维度
        G = gate_weights.unsqueeze(-1)                # [B, L, num_experts, 1]

        # 加权融合
        out = (expert_outputs * G).sum(dim=-2)        # [B, L, d]

        return out


# ============================================================================
#  5. 专家正交损失 (Expert Orthogonality Loss)
# ============================================================================
def compute_expert_orthogonality_loss(experts: nn.ModuleList) -> torch.Tensor:
    """计算专家正交惩罚，防止 MoE 坍塌为同质化特征。

    提取每个专家第一层投影矩阵权重 W_i (shape [rank, d])，
    计算 Frobenius 范数正交损失:

        L_ortho = Σ_{i ≠ j} ‖ W_i · W_j^T − I ‖_F²

    其中 W_i·W_j^T ∈ R^{rank × rank}，I 为 rank 阶单位阵。

    Args:
        experts: nn.ModuleList，每个元素为 LowRankConvExpert

    Returns:
        L_ortho: 标量正交损失
    """
    weights = [expert.down_proj.weight for expert in experts]  # 每个 [rank, d]
    device = weights[0].device
    rank = weights[0].shape[0]

    eye = torch.eye(rank, device=device)  # [rank, rank]
    loss = torch.tensor(0.0, device=device)

    n = len(weights)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # W_i: [rank, d]  ×  W_j^T: [d, rank]  →  Gram: [rank, rank]
            gram = torch.mm(weights[i], weights[j].T)  # [rank, rank]
            loss = loss + torch.sum((gram - eye) ** 2)  # ‖Gram − I‖_F²

    return loss


# ============================================================================
#  6. 监督对比损失 (Supervised Contrastive Loss)
# ============================================================================
class SupConLoss(nn.Module):
    """SupCon Loss (Khosla et al., NeurIPS 2020) + Feature Queue

    将同属于「深层意思」的样本在融合特征空间中拉近，
    与「浅层意思」样本推远。

    ★ Feature Queue 机制:
      batch_size=4 时正负样本对太少 (≤3对)，SupCon 几乎无效。
      引入一个 FIFO Queue 累积最近 K 个 batch 的特征 (不参与梯度)，
      使得每步计算时有效样本数 = B + queue_size，极大增加对比信号。
    """

    def __init__(self, temperature: float = 0.07, feat_dim: int = 64, queue_size: int = 64):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # FIFO 队列 (不参与梯度, 注册为 buffer 以便 .to(device) 自动迁移)
        self.register_buffer("feat_queue", torch.randn(queue_size, feat_dim))
        self.register_buffer("label_queue", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # 初始化标记: 队列未满前不使用
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def _enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        """将当前 batch 的特征入队 (FIFO)"""
        B = features.shape[0]
        ptr = int(self.queue_ptr.item())
        # 循环写入
        if ptr + B <= self.queue_size:
            self.feat_queue[ptr:ptr + B] = features.detach()
            self.label_queue[ptr:ptr + B] = labels.detach()
        else:
            overflow = (ptr + B) - self.queue_size
            self.feat_queue[ptr:] = features[:B - overflow].detach()
            self.label_queue[ptr:] = labels[:B - overflow].detach()
            self.feat_queue[:overflow] = features[B - overflow:].detach()
            self.label_queue[:overflow] = labels[B - overflow:].detach()
        self.queue_ptr[0] = (ptr + B) % self.queue_size
        if ptr + B >= self.queue_size:
            self.queue_filled[0] = True

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, D]  L2 归一化后的特征向量
            labels:   [B]     类别标签 (0 = lit, 1 = deep)

        Returns:
            loss: 标量 SupCon 损失
        """
        device = features.device
        B = features.shape[0]

        if B <= 1:
            self._enqueue(features, labels)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # —— 合并当前 batch + 队列特征 ——
        if self.queue_filled.item():
            all_features = torch.cat([features, self.feat_queue.detach()], dim=0)  # [B+Q, D]
            all_labels = torch.cat([labels, self.label_queue.detach()], dim=0)      # [B+Q]
        else:
            all_features = features     # [B, D]
            all_labels = labels          # [B]

        N = all_features.shape[0]

        # 相似度矩阵: 仅计算 anchor (当前batch) 与 all 的相似度
        sim = torch.mm(features, all_features.T) / self.temperature  # [B, N]

        # 正样本掩码: 同类且排除自身
        anchor_labels = labels.unsqueeze(1)            # [B, 1]
        all_labels_row = all_labels.unsqueeze(0)       # [1, N]
        pos_mask = (anchor_labels == all_labels_row).float()  # [B, N]

        # 自身掩码: 前 B 列中对角线位置是自身
        self_mask = torch.ones(B, N, device=device)
        self_mask[:, :B] = self_mask[:, :B] - torch.eye(B, device=device)  # [B, N]
        pos_mask = pos_mask * self_mask

        # 数值稳定: 减去每行最大值
        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max  # [B, N]

        exp_logits = torch.exp(logits) * self_mask  # [B, N]
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-8
        )  # [B, N]

        pos_count = pos_mask.sum(dim=1)  # [B]
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (
            pos_count + 1e-8
        )  # [B]

        valid = pos_count > 0
        if valid.any():
            loss = -mean_log_prob[valid].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 入队 (在 loss 计算后)
        self._enqueue(features, labels)

        return loss


# ============================================================================
#  7. 注意力池化 (Attention Pooling)
# ============================================================================
class AttentionPooling(nn.Module):
    """可学习的注意力加权池化，自动关注序列中最具判别力的时间步。"""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, L, d]  序列特征
            mask: [B, L]     1 = 有效, 0 = padding

        Returns:
            pooled: [B, d]  池化后的全局向量
        """
        scores = self.query(x).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)          # [B, L]
        pooled = (x * weights.unsqueeze(-1)).sum(1)  # [B, d]
        return pooled


# ============================================================================
#  8. 主模型: IntentContradictionNet
# ============================================================================
class IntentContradictionNet(nn.Module):
    """面向小样本语用学意图识别的多模态矛盾网络。

    完整前向路径:
        冻结编码器 → 初始投影 (768 → d) → OT 软对齐 → 矛盾残差
        → 矛盾路由 MoE (多尺度低秩 + 正交约束) → 注意力池化
        → { 主分类 (CE), 辅助 Token 预测 (BCE), 监督对比 (SupCon) } 联合损失

    防过拟合策略:
        • 编码器全部冻结 / 仅解冻顶层
        • 低秩瓶颈 (rank = d // 4) 大幅压缩参数
        • 专家正交损失防止 MoE 坍塌
        • Dropout + LayerNorm 正则化
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 2,
        proj_dim: int = 128,
        rank: int = 32,
        num_experts: int = 3,
        kernel_sizes: Optional[List[int]] = None,
        sinkhorn_iters: int = 10,
        sinkhorn_epsilon: float = 0.1,
        supcon_temperature: float = 0.07,
        dropout: float = 0.2,
        label_smoothing: float = 0.1,
        # —— 损失权重 ——
        lambda_cls: float = 1.0,
        lambda_aux: float = 0.1,
        lambda_ortho: float = 0.01,
        lambda_supcon: float = 0.1,
        # —— 冻结 ——
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        self.proj_dim = proj_dim
        self.lambda_cls = lambda_cls
        self.lambda_aux = lambda_aux
        self.lambda_ortho = lambda_ortho
        self.lambda_supcon = lambda_supcon

        # ================================================================
        # 冻结预训练编码器
        # ================================================================
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)

        if freeze_text_encoder:
            self._freeze(self.text_encoder)
        if freeze_audio_encoder:
            self._freeze(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size    # 768
        audio_hidden = self.audio_encoder.config.hidden_size  # 768

        # ================================================================
        # 初始投影  encoder_dim → proj_dim
        # ================================================================
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_hidden),
            nn.Linear(text_hidden, proj_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_hidden),
            nn.Linear(audio_hidden, proj_dim),
        )

        # ================================================================
        # 核心模块
        # ================================================================
        self.ot_layer = OTAlignmentLayer(
            d_model=proj_dim,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
            dropout=dropout,
        )

        self.router = ContradictionRouter(
            d_model=proj_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

        self.moe = MultiScaleMoEExperts(
            d_model=proj_dim,
            rank=rank,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.pool_contra = AttentionPooling(proj_dim)   # 矛盾路径池化
        self.pool_audio = AttentionPooling(proj_dim)    # 原始音频池化

        # ================================================================
        # 韵律特征分支 (18-dim → MLP → proj_dim)
        # ================================================================
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ================================================================
        # SpeechCraft 类别嵌入分支 (3 × Embedding → MLP → proj_dim)
        # ================================================================
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(
            nn.Linear(sc_embed_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ================================================================
        # 四路门控融合: 矛盾 + 音频 + 韵律 + SpeechCraft
        # ================================================================
        fusion_dim = proj_dim * 4
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # ================================================================
        # Loss Head 1 — 主任务: 二分类
        # ================================================================
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ================================================================
        # Loss Head 2 — 辅助任务: Token 级声学突变预测
        # ================================================================
        self.aux_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1),
        )

        # ================================================================
        # Loss Head 3 — 监督对比学习 (SupCon)
        # ================================================================
        self.supcon_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim // 2),
        )
        self.supcon_loss_fn = SupConLoss(
            temperature=supcon_temperature,
            feat_dim=proj_dim // 2,
            queue_size=128,
        )

    # ================================================================
    # 工具方法
    # ================================================================
    @staticmethod
    def _freeze(module: nn.Module):
        """冻结模块全部参数。"""
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze(module: nn.Module):
        """解冻模块全部参数。"""
        for p in module.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        """仅解冻音频编码器顶部 num_layers 个 Transformer 层。"""
        self._freeze(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self._unfreeze(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        """统一设置编码器冻结策略。"""
        if freeze_text:
            self._freeze(self.text_encoder)
        else:
            self._unfreeze(self.text_encoder)

        if freeze_audio:
            self._freeze(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    def _compute_reduced_audio_mask(
        self, attention_mask: torch.Tensor, hidden_seq_len: int
    ) -> torch.Tensor:
        """计算经 wav2vec2 CNN 下采样后对应的 attention mask。

        Args:
            attention_mask: [B, T_raw]   原始音频 attention mask
            hidden_seq_len: int          wav2vec2 输出的序列长度

        Returns:
            reduced: [B, hidden_seq_len] 下采样后的 mask
        """
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(
            input_lengths
        )
        B = attention_mask.shape[0]
        reduced = torch.zeros(
            B, hidden_seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for i, length in enumerate(output_lengths):
            reduced[i, : length.item()] = 1
        return reduced

    # ================================================================
    # 前向传播
    # ================================================================
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        aux_labels: Optional[torch.Tensor] = None,
        aux_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """完整前向传播 + 联合损失计算。

        Args:
            text_input_ids:       [B, L_t_raw]   BERT 输入 token ids
            text_attention_mask:  [B, L_t_raw]   BERT attention mask
            audio_input_values:   [B, T_audio]   原始波形采样值
            audio_attention_mask: [B, T_audio]   音频 attention mask
            prosody_features:     [B, 18]        韵律统计特征
            speechcraft_features: [B, 3]         SpeechCraft 类别 (long)
            labels:               [B]            主任务标签 (0=lit / 1=deep)
            aux_labels:           [B, L_t]       辅助伪标签 (token-level, 0/1 浮点)
            aux_mask:             [B, L_t]       辅助任务置信度掩码 (1=可用)

        Returns:
            dict 包含:
                logits:      [B, num_labels]  主任务 logits
                loss:        标量             加权联合总损失 (仅 labels 非 None 时)
                loss_cls:    标量             主任务 CE 损失
                loss_aux:    标量             辅助任务 BCE 损失
                loss_ortho:  标量             专家正交损失
                loss_supcon: 标量             监督对比损失
        """

        # =============================================================
        # 1. 编码器特征提取
        #    ★ 关键: 根据参数 requires_grad 动态决定是否阻断梯度
        #      文本编码器永久冻结 → 始终 no_grad
        #      音频编码器解冻后 → 必须允许梯度回传
        # =============================================================
        # —— 文本编码器 (永久冻结) ——
        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                return_dict=True,
            )
        T_raw = text_out.last_hidden_state   # [B, L_t, 768]

        # —— 音频编码器 (可能解冻) ——
        audio_encoder_frozen = not any(
            p.requires_grad for p in self.audio_encoder.parameters()
        )
        if audio_encoder_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(
                    input_values=audio_input_values,
                    attention_mask=audio_attention_mask,
                    return_dict=True,
                )
        else:
            audio_out = self.audio_encoder(
                input_values=audio_input_values,
                attention_mask=audio_attention_mask,
                return_dict=True,
            )
        A_raw = audio_out.last_hidden_state  # [B, L_a, 768]

        # 音频 mask 降采样 (原始长度 → wav2vec2 输出长度)
        if audio_attention_mask is not None:
            audio_mask_reduced = self._compute_reduced_audio_mask(
                audio_attention_mask, A_raw.shape[1]
            )  # [B, L_a]
        else:
            audio_mask_reduced = None

        # =============================================================
        # 2. 初始投影  768 → proj_dim
        # =============================================================
        T = self.text_proj(T_raw)   # [B, L_t, 768] → [B, L_t, proj_dim]
        A = self.audio_proj(A_raw)  # [B, L_a, 768] → [B, L_a, proj_dim]

        # =============================================================
        # 3. OT 软对齐
        # =============================================================
        T_prime, A_aligned, S = self.ot_layer(
            T, A,
            text_mask=text_attention_mask,
            audio_mask=audio_mask_reduced,
        )
        # T_prime:    [B, L_t, proj_dim]   共享投影后的文本特征
        # A_aligned:  [B, L_t, proj_dim]   OT 对齐后的音频特征 (长度对齐到文本)
        # S:          [B, L_t]             Token 级别点积相似度

        # =============================================================
        # 4. 矛盾残差  |T' − A_aligned|
        # =============================================================
        residual = torch.abs(T_prime - A_aligned)  # [B, L_t, proj_dim]

        # =============================================================
        # 5. 矛盾路由 (基于模态差异)
        # =============================================================
        G = self.router(S, residual)  # [B, L_t, num_experts]

        # =============================================================
        # 6. 多尺度低秩 MoE 专家
        # =============================================================
        moe_out = self.moe(residual, G)  # [B, L_t, proj_dim]

        # =============================================================
        # 7. 四路池化 → 句级表示
        # =============================================================
        # 路径 A: 矛盾信号 (MoE 输出)
        pooled_contra = self.pool_contra(moe_out, mask=text_attention_mask)  # [B, proj_dim]
        # 路径 B: 原始音频语义 (OT 对齐后的音频特征)
        pooled_audio = self.pool_audio(A_aligned, mask=text_attention_mask)  # [B, proj_dim]

        # 路径 C: 韵律特征
        B_size = pooled_contra.shape[0]
        if prosody_features is not None:
            pooled_prosody = self.prosody_mlp(prosody_features)    # [B, proj_dim]
        else:
            pooled_prosody = torch.zeros(B_size, self.proj_dim, device=pooled_contra.device)

        # 路径 D: SpeechCraft 类别嵌入
        if speechcraft_features is not None:
            sc_pitch = self.pitch_embed(speechcraft_features[:, 0])    # [B, 16]
            sc_energy = self.energy_embed(speechcraft_features[:, 1])  # [B, 16]
            sc_speed = self.speed_embed(speechcraft_features[:, 2])    # [B, 16]
            sc_cat = torch.cat([sc_pitch, sc_energy, sc_speed], dim=-1)  # [B, 48]
            pooled_sc = self.sc_mlp(sc_cat)                              # [B, proj_dim]
        else:
            pooled_sc = torch.zeros(B_size, self.proj_dim, device=pooled_contra.device)

        # 四路门控融合
        fused = torch.cat([pooled_contra, pooled_audio, pooled_prosody, pooled_sc], dim=-1)  # [B, 4*proj_dim]
        gate = self.fusion_gate(fused)    # [B, 4*proj_dim]
        fused = fused * gate              # [B, 4*proj_dim]

        pooled = pooled_contra  # SupCon 使用矛盾路径特征

        # =============================================================
        # 8. 主分类
        # =============================================================
        logits = self.classifier(fused)  # [B, num_labels]

        output: Dict[str, torch.Tensor] = {"logits": logits}

        # =============================================================
        # 9. 联合损失计算
        # =============================================================
        if labels is not None:
            # ---- (a) 主任务: 二分类 CrossEntropyLoss ----
            L_cls = self.cls_loss_fn(logits, labels)

            # ---- (b) 辅助任务: Token 级声学突变预测 ----
            #     使用 BCEWithLogitsLoss + 置信度 Mask
            aux_logits = self.aux_head(moe_out).squeeze(-1)  # [B, L_t]

            if aux_labels is not None and aux_mask is not None:
                valid = aux_mask.bool()
                if valid.any():
                    L_aux = F.binary_cross_entropy_with_logits(
                        aux_logits[valid],
                        aux_labels[valid].float(),
                    )
                else:
                    L_aux = torch.tensor(0.0, device=logits.device)
            else:
                # 未提供辅助标签时，该项为 0
                L_aux = torch.tensor(0.0, device=logits.device)

            # ---- (c) 专家正交损失 ----
            L_ortho = compute_expert_orthogonality_loss(self.moe.experts)

            # ---- (d) 监督对比损失 (SupCon) ----
            supcon_feat = self.supcon_proj(pooled)          # [B, proj_dim // 2]
            supcon_feat = F.normalize(supcon_feat, dim=-1)  # L2 归一化
            L_supcon = self.supcon_loss_fn(supcon_feat, labels)

            # ---- 加权求和 ----
            total_loss = (
                self.lambda_cls    * L_cls
                + self.lambda_aux    * L_aux
                + self.lambda_ortho  * L_ortho
                + self.lambda_supcon * L_supcon
            )

            output["loss"] = total_loss
            output["loss_cls"] = L_cls.detach()
            output["loss_aux"] = (
                L_aux.detach() if isinstance(L_aux, torch.Tensor)
                else torch.tensor(L_aux)
            )
            output["loss_ortho"] = L_ortho.detach()
            output["loss_supcon"] = L_supcon.detach()

        return output


# ============================================================================
#  Usage Example & Smoke Test
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  IntentContradictionNet — Smoke Test")
    print("=" * 70)

    # 模拟输入 (不加载真实编码器，仅验证维度流转)
    B, L_t, L_a, D = 4, 12, 80, 128

    # ---- 测试各子模块 ----
    print("\n[1] OTAlignmentLayer")
    ot = OTAlignmentLayer(d_model=D, sinkhorn_iters=5, sinkhorn_epsilon=0.1)
    T_dummy = torch.randn(B, L_t, D)
    A_dummy = torch.randn(B, L_a, D)
    t_mask = torch.ones(B, L_t)
    a_mask = torch.ones(B, L_a)
    T_p, A_al, sim = ot(T_dummy, A_dummy, text_mask=t_mask, audio_mask=a_mask)
    print(f"    T_prime: {T_p.shape}  A_aligned: {A_al.shape}  S: {sim.shape}")
    assert T_p.shape == (B, L_t, D)
    assert A_al.shape == (B, L_t, D)
    assert sim.shape == (B, L_t)

    print("\n[2] ContradictionRouter")
    router = ContradictionRouter(d_model=D, num_experts=3)
    residual = torch.abs(T_p - A_al)
    G = router(sim, residual)
    print(f"    Gate weights: {G.shape}  sum={G.sum(dim=-1)[0, :3].tolist()}")
    assert G.shape == (B, L_t, 3)

    print("\n[3] MultiScaleMoEExperts")
    moe = MultiScaleMoEExperts(d_model=D, rank=32, kernel_sizes=[1, 3, 5])
    moe_out = moe(residual, G)
    print(f"    MoE output: {moe_out.shape}")
    assert moe_out.shape == (B, L_t, D)

    print("\n[4] Expert Orthogonality Loss")
    L_ortho = compute_expert_orthogonality_loss(moe.experts)
    print(f"    L_ortho = {L_ortho.item():.4f}")

    print("\n[5] SupConLoss (with Feature Queue)")
    supcon = SupConLoss(temperature=0.07, feat_dim=D // 2, queue_size=64)
    feat = F.normalize(torch.randn(B, D // 2), dim=-1)
    lbl = torch.tensor([0, 1, 0, 1])
    L_sup = supcon(feat, lbl)
    print(f"    L_supcon = {L_sup.item():.4f}  (queue_filled={supcon.queue_filled.item()})")
    # 模拟多步入队
    for _ in range(20):
        supcon(F.normalize(torch.randn(B, D // 2), dim=-1), torch.randint(0, 2, (B,)))
    L_sup2 = supcon(feat, lbl)
    print(f"    L_supcon (after queue fill) = {L_sup2.item():.4f}  (queue_filled={supcon.queue_filled.item()})")

    print("\n[6] AttentionPooling")
    pool = AttentionPooling(D)
    pooled = pool(moe_out, mask=t_mask)
    print(f"    Pooled: {pooled.shape}")
    assert pooled.shape == (B, D)

    print("\n✅ All sub-module smoke tests passed!")

    # ---- 联合损失组合示例 ----
    print("\n" + "=" * 70)
    print("  Joint Loss Combination Example")
    print("=" * 70)
    print("""
    # 在训练循环中:
    output = model(
        text_input_ids=batch["text_input_ids"],
        text_attention_mask=batch["text_attention_mask"],
        audio_input_values=batch["audio_input_values"],
        audio_attention_mask=batch["audio_attention_mask"],
        labels=batch["labels"],
        aux_labels=batch.get("aux_labels"),   # 可选
        aux_mask=batch.get("aux_mask"),       # 可选
    )

    # output["loss"] = λ_cls * L_cls + λ_aux * L_aux
    #                + λ_ortho * L_ortho + λ_supcon * L_supcon
    #
    # 默认权重: λ_cls=1.0, λ_aux=0.1, λ_ortho=0.01, λ_supcon=0.1
    # 各项损失可通过 output["loss_cls"], output["loss_ortho"] 等单独监控

    loss = output["loss"]
    loss.backward()
    """)
