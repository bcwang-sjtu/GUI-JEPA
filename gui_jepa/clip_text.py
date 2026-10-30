"""Frozen CLIP text encoder (openai/clip-vit-base-patch32, 512-d, L2-normalised)."""
from typing import List

import torch

CLIP_PATH = "openai/clip-vit-base-patch32"


class CLIPTextEncoder:
    def __init__(self, device=torch.device("cuda"), model_name: str = CLIP_PATH,
                 max_length: int = 77):
        from transformers import CLIPTokenizer, CLIPTextModel
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
        self.cache = {}

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Returns (B, 512) L2-normalised embeddings; repeated texts are cached."""
        out, new_idx, new_texts = [], [], []
        for i, t in enumerate(texts):
            if t in self.cache:
                out.append(self.cache[t])
            else:
                out.append(None)
                new_idx.append(i)
                new_texts.append(t)
        if new_texts:
            inputs = self.tokenizer(new_texts, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            feats = self.model(**inputs).pooler_output
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            for k, i in enumerate(new_idx):
                v = feats[k].cpu()
                out[i] = v
                self.cache[new_texts[k]] = v
        return torch.stack(out, dim=0).to(self.device)
