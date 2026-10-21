import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def infer_parent_dims_from_batch(
    pa: dict[str, torch.Tensor],
    parents: list[str],
) -> dict[str, int]:
    parent_dims: dict[str, int] = {}
    for k in parents:
        if k not in pa:
            raise KeyError(f"Parent '{k}' not found in batch['pa']")
        v = pa[k]
        parent_dims[k] = 1 if v.ndim == 1 else int(v.shape[1])
    return parent_dims


@dataclass
class CondEmbedderConfig:
    parents: list[str]
    parent_dims: dict[str, int]
    cond_embed_dim: int

class PerAttrCondEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parents = args.parents
        self.parent_dims = args.parent_dims
        self.cond_embed_dim = args.cond_embed_dim

        assert self.cond_embed_dim % len(self.parents) == 0, (
            f"cond_embed_dim ({self.cond_embed_dim}) must be divisible by "
            f"number of parents ({len(self.parents)})"
        )

        self.parent_embed_dim = self.cond_embed_dim // len(self.parents)

        self.embeddings = nn.ModuleDict()

        for key in self.parents:
            in_dim = self.parent_dims[key]
            print(f"Creating embedder for parent '{key}' with input dim {in_dim} "
                  f"and output dim {self.parent_embed_dim}")
            self.embeddings[key] = nn.Sequential(
                nn.Linear(in_dim, self.parent_embed_dim),
                nn.SiLU(),
                nn.Linear(self.parent_embed_dim, self.parent_embed_dim),
            )

    def forward(self, y_dict, null_keys=None):
        null_keys = set() if null_keys is None else set(null_keys)
        embs = []

        for key in self.parents:
            assert key in y_dict, f"Missing key in y_dict: {key}"
            val = y_dict[key]  # [B, in_dim]

            if key in null_keys:
                emb = torch.zeros(val.shape[0], self.parent_embed_dim, device=val.device, dtype=val.dtype)
            else:
                emb = self.embeddings[key](val)

            embs.append(emb)

        out = torch.cat(embs, dim=1)   # [B, cond_embed_dim]
        out = F.normalize(out, p=2, dim=1)
        return out


class GlobalCondEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parents = args.parents
        self.parent_dims = args.parent_dims
        self.cond_embed_dim = args.cond_embed_dim

        self.input_dim = sum(self.parent_dims[k] for k in self.parents)

        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.cond_embed_dim),
            nn.SiLU(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
        )

    def forward(self, y_dict, null_keys=None):
        if null_keys is not None and len(null_keys) > 0:
            if set(null_keys) != set(self.parents):
                raise ValueError(
                    "GlobalCondEmbedder only supports full nulling, not per-attribute nulling."
                )
            batch_size = next(iter(y_dict.values())).shape[0]
            ref = next(iter(y_dict.values()))
            return torch.zeros(batch_size, self.cond_embed_dim, device=ref.device, dtype=ref.dtype)

        vals = [y_dict[k] for k in self.parents]
        concat_y = torch.cat(vals, dim=1)
        out = self.embed(concat_y)
        return F.normalize(out, p=2, dim=1)