"""JEPA pretraining of the GUI encoder (stage 1).

The student ViT sees the screenshot with a random half of its patches zeroed; a
predictor regresses, at the masked positions, the tokens that an EMA teacher
computes on the full screenshot (smooth-L1 in latent space, stop-gradient target).
"""
import copy

import torch
import torch.nn as nn

from .models import make_vit, build_2d_sincos_pos_emb, TokenPredictor


class IJEPA(nn.Module):
    def __init__(self, backbone_name: str = "vit_large_patch16_224",
                 pretrained_backbone: bool = True, predictor_depth: int = 6,
                 predictor_heads: int = 12, num_patches_side: int = 14,
                 embed_dim: int = 1024, ema_m: float = 0.996):
        super().__init__()
        self.num_patches_side = num_patches_side
        self.num_patches = num_patches_side * num_patches_side
        self.embed_dim = embed_dim
        self.ema_m = ema_m
        self.student = make_vit(backbone_name, pretrained=pretrained_backbone)
        self.target = copy.deepcopy(self.student)
        for p in self.target.parameters():
            p.requires_grad_(False)
        pe = build_2d_sincos_pos_emb(num_patches_side, num_patches_side, embed_dim)
        self.register_buffer("predictor_pos_emb", pe.unsqueeze(0))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        for h in (predictor_heads, 16, 12, 8, 4):
            if embed_dim % h == 0:
                heads = h
                break
        self.predictor = TokenPredictor(embed_dim=embed_dim, depth=predictor_depth, num_heads=heads)

    @torch.no_grad()
    def momentum_update(self):
        for ps, pt in zip(self.student.parameters(), self.target.parameters()):
            pt.data.mul_(self.ema_m).add_(ps.data, alpha=1.0 - self.ema_m)

    def _tokens(self, vit, x):
        feats = vit.forward_features(x)
        n = feats.size(1)
        if n == self.num_patches:
            return feats
        if self.num_patches < n <= self.num_patches + 8:  # drop CLS / register tokens
            return feats[:, -self.num_patches:]
        raise RuntimeError(f"unexpected token shape {tuple(feats.shape)}")

    def forward(self, images, mask_ratio: float = 0.5):
        B, N, device = images.size(0), self.num_patches, images.device
        with torch.no_grad():
            target_tokens = self._tokens(self.target, images).detach()

        # random patch mask (screen content is spread over many small elements)
        n_mask = int(round(N * mask_ratio))
        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        mask_idx, visible_idx = perm[:, :n_mask], perm[:, n_mask:]

        masked = images.clone()
        ps = images.size(2) // self.num_patches_side
        for b in range(B):
            for idx in mask_idx[b].tolist():
                py, px = divmod(idx, self.num_patches_side)
                masked[b, :, py * ps:(py + 1) * ps, px * ps:(px + 1) * ps] = 0
        student_tokens = self._tokens(self.student, masked)

        b_idx = torch.arange(B, device=device).unsqueeze(1)
        pe = self.predictor_pos_emb.expand(B, -1, -1)
        visible = student_tokens[b_idx, visible_idx] + pe[b_idx, visible_idx]
        query = self.mask_token + pe[b_idx, mask_idx]
        pred = self.predictor(visible, query)
        target = target_tokens[b_idx, mask_idx]
        loss = nn.functional.smooth_l1_loss(pred, target)
        with torch.no_grad():
            metrics = {"loss": loss.item(),
                       "tgt_std": target.float().std(dim=0).mean().item(),
                       "pred_std": pred.float().std(dim=0).mean().item()}
        return loss, metrics
