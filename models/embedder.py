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


class LabelEmbedderMammo(nn.Module):
    """
    Embeds (density, scanner, view, age) conditioning into a single vector.
    Handles classifier-free guidance via a single drop mask per sample.
    """
    def __init__(
        self,
        hidden_size: int,
        num_density_classes: int,
        num_scanner_classes: int,
        num_view_classes: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.use_cfg = dropout_prob > 0

        # Categorical: +1 row for the CFG null token
        self.density_emb = nn.Embedding(num_density_classes + self.use_cfg, hidden_size)
        self.scanner_emb = nn.Embedding(num_scanner_classes + self.use_cfg, hidden_size)
        self.view_emb = nn.Embedding(num_view_classes + self.use_cfg, hidden_size)

        # Continuous age: scalar -> hidden_size
        self.age_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Learned null vector for age
        if self.use_cfg:
            self.age_null = nn.Parameter(torch.zeros(hidden_size))

        # Null-token indices
        self.null_density = num_density_classes
        self.null_scanner = num_scanner_classes
        self.null_view = num_view_classes

        # Init (matches DiT convention)
        nn.init.normal_(self.density_emb.weight, std=0.02)
        nn.init.normal_(self.scanner_emb.weight, std=0.02)
        nn.init.normal_(self.view_emb.weight, std=0.02)

        for m in self.age_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, labels: dict, train: bool, force_drop_ids: torch.Tensor = None):
        """
        labels: dict with keys:
            'label_density', 'label_scanner', 'label_view', 'label_age'

            - density: long tensor [B]
            - scanner: long tensor [B]
            - view: long tensor [B]
            - age: float tensor [B], assumed normalized to [-1, 1]

        train: whether CFG dropout is active
        force_drop_ids: optional [B] bool/int mask to force-drop samples
                        (used for CFG at inference)

        Returns:
            [B, hidden_size] conditioning vector
        """
        density = labels['label_density']
        scanner = labels['label_scanner']
        view = labels['label_view']
        age = labels['label_age'].float()

        B = density.shape[0]
        device = density.device

        # One joint CFG drop mask per sample
        if force_drop_ids is not None:
            drop_mask = force_drop_ids.bool()
        elif train and self.use_cfg:
            drop_mask = torch.rand(B, device=device) < self.dropout_prob
        else:
            drop_mask = torch.zeros(B, dtype=torch.bool, device=device)

        # Replace all categorical conditions with their null tokens together
        if self.use_cfg:
            density = torch.where(
                drop_mask,
                torch.full_like(density, self.null_density),
                density,
            )
            scanner = torch.where(
                drop_mask,
                torch.full_like(scanner, self.null_scanner),
                scanner,
            )
            view = torch.where(
                drop_mask,
                torch.full_like(view, self.null_view),
                view,
            )

        density_e = self.density_emb(density)
        scanner_e = self.scanner_emb(scanner)
        view_e = self.view_emb(view)

        # Continuous age
        age_e = self.age_mlp(age.unsqueeze(-1))
        if self.use_cfg:
            age_e = torch.where(
                drop_mask.unsqueeze(-1),
                self.age_null.expand_as(age_e),
                age_e,
            )

        return density_e + scanner_e + view_e + age_e