"""
Neural Architecture Models for NER Logistics Sentinel:
  - HazardNet: Multi-task residual trunk with continuous hazard regression & ordinal risk head
  - DelayNet: Class-conditional mixture network for segment excess delay & ETA confidence intervals
  - RouteNet: Wide-and-deep network for end-to-end multi-segment route delays
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with LayerNorm, SiLU activation, linear projection, and dropout."""
    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln(x)
        out = F.silu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        return residual + out


class HazardNet(nn.Module):
    """Multi-task residual network predicting continuous hazard and 3-class risk probabilities.

    Inputs:
      x_num: Continuous feature tensor of shape (batch, 27)
      x_cat: Categorical index tensor of shape (batch, 3) [road_class, terrain, weather]
    Outputs:
      mu: Continuous hazard estimate in [0, 0.995]
      probs: Calibrated ordinal probabilities [P(safe), P(risky), P(blocked)]
    """
    def __init__(self, n_num: int = 27, emb_cards: Tuple[int, ...] = (3, 4, 8),
                 emb_dim: int = 4, width: int = 256, blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in emb_cards])
        d_in = n_num + emb_dim * len(emb_cards)
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([ResBlock(width, dropout) for _ in range(blocks)])
        self.haz = nn.Linear(width, 1)
        self.risk_head = nn.Linear(width, 3)

    def forward_trunk(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb_list = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        h = self.inp(torch.cat([x_num] + emb_list, dim=-1))
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_trunk(x_num, x_cat)
        mu = 0.995 * torch.sigmoid(self.haz(h)).squeeze(-1)
        logits = self.risk_head(h)
        probs = F.softmax(logits, dim=-1)
        return mu, probs


class DelayNet(nn.Module):
    """Class-conditional mixture network for segment excess delay estimation."""
    def __init__(self, n_num: int = 27, emb_cards: Tuple[int, ...] = (3, 4, 8),
                 emb_dim: int = 4, width: int = 256, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in emb_cards])
        # Input features + mu (1) + class probabilities (3)
        d_in = n_num + emb_dim * len(emb_cards) + 4
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([ResBlock(width, dropout) for _ in range(blocks)])

        # 3 positive class-conditional delay heads
        self.head_safe = nn.Linear(width, 1)
        self.head_risky = nn.Linear(width, 1)
        self.head_blocked = nn.Linear(width, 1)

        # Quantile offset heads (P50, P90)
        self.p50_head = nn.Linear(width, 1)
        self.p90_head = nn.Linear(width, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor,
                mu: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_list = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        aux = torch.cat([mu.unsqueeze(-1), probs], dim=-1)
        h = self.inp(torch.cat([x_num] + emb_list + [aux], dim=-1))
        for b in self.blocks:
            h = b(h)

        g_safe = F.softplus(self.head_safe(h)).squeeze(-1)
        g_risky = F.softplus(self.head_risky(h)).squeeze(-1)
        g_blocked = F.softplus(self.head_blocked(h)).squeeze(-1)

        # Mixture delay = sum(p_k * g_k)
        mean_delay = (probs[:, 0] * g_safe +
                      probs[:, 1] * g_risky +
                      probs[:, 2] * g_blocked)

        p50 = mean_delay + self.p50_head(h).squeeze(-1)
        p90 = mean_delay + F.softplus(self.p90_head(h).squeeze(-1))
        p50 = F.relu(p50)
        p90 = torch.maximum(p90, p50 + 0.1)

        return mean_delay, p50, p90


class RouteNet(nn.Module):
    """Wide-and-deep network predicting route delay from route-level aggregates."""
    def __init__(self, n_in: int = 18, width: int = 128, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([ResBlock(width, dropout) for _ in range(blocks)])
        self.out_mean = nn.Linear(width, 1)
        self.out_p90 = nn.Linear(width, 1)

    def forward(self, x_route: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.inp(x_route))
        for b in self.blocks:
            h = b(h)
        mean_delay = F.softplus(self.out_mean(h)).squeeze(-1)
        p90 = mean_delay + F.softplus(self.out_p90(h)).squeeze(-1)
        return mean_delay, p90
