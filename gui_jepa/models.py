"""GUI-JEPA modules: JEPA encoder, residual latent predictor, structural head and text matcher."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .actions import KINDS

DEFAULT_MATCHER = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_SPAN_RANKER = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- encoder

def make_vit(name: str = "vit_large_patch16_224", pretrained: bool = False):
    """timm ViT that returns raw patch tokens from `forward_features`."""
    m = timm.create_model(name, pretrained=pretrained, num_classes=0)
    m.global_pool = "token"
    return m


def load_encoder(encoder_ckpt: str, backbone_name: str = "vit_large_patch16_224",
                 device=torch.device("cuda")) -> nn.Module:
    """Load the frozen GUI-pretrained JEPA ViT-L/16."""
    vit = make_vit(backbone_name, pretrained=False)
    sd = torch.load(encoder_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    print(f"[encoder] {encoder_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit.to(device)


@torch.no_grad()
def extract_patch_tokens(vit: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, 224, 224) -> (B, 196, 1024) patch tokens (CLS dropped)."""
    feats = vit.forward_features(img)
    if feats.dim() == 3 and feats.size(1) == 197:
        feats = feats[:, 1:, :]
    return feats


def build_2d_sincos_pos_emb(grid_h: int, grid_w: int, embed_dim: int):
    assert embed_dim % 4 == 0
    grid_y = torch.arange(grid_h).float()
    grid_x = torch.arange(grid_w).float()
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_h, grid_w)
    pos = grid.flatten(2).permute(0, 2, 1)  # (2, H*W, 1)
    pos_y, pos_x = pos[0], pos[1]
    half = embed_dim // 2
    omega = torch.arange(half // 2).float() / (half // 2)
    omega = 1.0 / (10000 ** omega)
    out_y = torch.einsum("nd,k->nk", pos_y, omega)
    out_y = torch.cat([out_y.sin(), out_y.cos()], dim=-1)
    out_x = torch.einsum("nd,k->nk", pos_x, omega)
    out_x = torch.cat([out_x.sin(), out_x.cos()], dim=-1)
    return torch.cat([out_y, out_x], dim=-1)  # (H*W, embed_dim)


class TokenPredictor(nn.Module):
    """Transformer that reads (context tokens, query tokens) and returns the query outputs."""

    def __init__(self, embed_dim: int = 768, depth: int = 6, num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=int(embed_dim * mlp_ratio),
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visible_tokens, mask_query):
        nv = visible_tokens.size(1)
        x = torch.cat([visible_tokens, mask_query], dim=1)
        x = self.norm(self.blocks(x))
        return x[:, nv:]


# --------------------------------------------------------------------------- predictor

class LatentPredictor(nn.Module):
    """Action-conditioned latent predictor.

    Takes the screen latent z_t (B, 196, D) and a candidate's action input (B, A)
    and returns a (B, 196, D) map. With the residual form used by GUI-JEPA the
    candidate-conditioned latent is z_t + predictor(z_t, a).
    """

    def __init__(self, embed_dim: int = 1024, action_dim: int = 16,
                 num_patches_side: int = 14, depth: int = 3, heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_side = num_patches_side
        self.num_patches = num_patches_side * num_patches_side
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        pe = build_2d_sincos_pos_emb(num_patches_side, num_patches_side, embed_dim)
        self.register_buffer("pos_emb", pe.unsqueeze(0))
        self.future_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.future_query, std=0.02)
        for h in (heads, 16, 12, 8, 4):
            if embed_dim % h == 0:
                heads = h
                break
        self.predictor = TokenPredictor(embed_dim=embed_dim, depth=depth, num_heads=heads)

    def forward(self, z_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        B, N, D = z_t.shape
        action_emb = self.action_proj(action_t).unsqueeze(1)          # (B, 1, D)
        visible = torch.cat([z_t + self.pos_emb, action_emb], dim=1)  # (B, N+1, D)
        query = self.future_query.expand(B, N, D) + self.pos_emb      # (B, N, D)
        return self.predictor(visible, query)


# --------------------------------------------------------------------------- scorers

class StructuralHead(nn.Module):
    """score = kappa * cos(MLP_v([roi_pool(z~); mean(z~)]), MLP_t(cond)) + per-kind bias.

    Its only candidate-dependent visual input is the candidate-conditioned latent.
    """

    def __init__(self, zdim=1024, tdim=512, pdim=256, cond_extra=0):
        super().__init__()
        self.proj_v = nn.Sequential(nn.Linear(zdim * 2, 512), nn.GELU(),
                                    nn.Linear(512, pdim))
        self.proj_t = nn.Sequential(nn.Linear(tdim + cond_extra, 512), nn.GELU(),
                                    nn.Linear(512, pdim))
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.kind_bias = nn.Parameter(torch.zeros(len(KINDS)))

    def forward(self, roi_pred, glob_pred, cond, kind_ids):
        v = F.normalize(self.proj_v(torch.cat([roi_pred, glob_pred], -1)), dim=-1)
        t = F.normalize(self.proj_t(cond), dim=-1)
        return (v * t).sum(-1) * self.logit_scale + self.kind_bias[kind_ids]


class CrossEncoderMatcher(nn.Module):
    """MiniLM cross-encoder: score(query, text) for a list of texts.

    Used as the text matcher (query = condition text, texts = element candidates)
    and as the span ranker (query = goal, texts = candidate spans).
    """

    def __init__(self, device, model_name=DEFAULT_MATCHER, dropout=0.0):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.lm.config.hidden_size, 1)
        self.drop = nn.Dropout(dropout)
        self.device = device

    def forward(self, query_text, cand_texts):
        """Returns (len(cand_texts),) scores."""
        pairs = [(query_text[:300], c or "[empty]") for c in cand_texts]
        enc = self.tok([p[0] for p in pairs], [p[1] for p in pairs],
                       padding=True, truncation=True, max_length=96,
                       return_tensors="pt").to(self.device)
        out = self.lm(**enc).last_hidden_state[:, 0]  # CLS
        return self.head(self.drop(out)).squeeze(-1)


def load_span_ranker(ranker_dir, device, tokenizer_name=DEFAULT_SPAN_RANKER):
    """Load a trained span ranker saved by scripts/train_span_ranker.py."""
    import os
    from transformers import AutoModel
    ranker = CrossEncoderMatcher(device, tokenizer_name).to(device)
    ranker.lm = AutoModel.from_pretrained(os.path.join(ranker_dir, "ranker_lm")).to(device)
    ranker.head.load_state_dict(torch.load(os.path.join(ranker_dir, "ranker_head.pt"),
                                           map_location=device))
    ranker.eval()
    return ranker
