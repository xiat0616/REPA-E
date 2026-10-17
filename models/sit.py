# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(nn.Linear(hidden_size, projector_dim), nn.SiLU(), nn.Linear(projector_dim, projector_dim), nn.SiLU(), nn.Linear(projector_dim, z_dim))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                       Embedding Layers for Conditions                         #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Kept for compatibility with the original SiT code; joint conditioning uses ParentEmbedder."""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class ParentEmbedder(nn.Module):
    """Embeds age, view, density, and scanner into one conditioning vector."""
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.age_embedder = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.view_embedder = nn.Embedding(3, hidden_size)      # 2 classes + null
        self.density_embedder = nn.Embedding(5, hidden_size)  # 4 classes + null
        self.scanner_embedder = nn.Embedding(6, hidden_size)  # 5 classes + null
        self.age_null = nn.Parameter(torch.zeros(1, hidden_size))
        self.parent_proj = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, parents, train, force_drop_ids=None):
        age = parents["age"].float()
        if age.ndim == 1:
            age = age[:, None]
        view = parents["view"].long()
        density = parents["density"].long()
        scanner = parents["scanner"].long()

        if force_drop_ids is None:
            drop_ids = torch.rand(age.shape[0], device=age.device) < self.dropout_prob if train and self.dropout_prob > 0 else torch.zeros(age.shape[0], dtype=torch.bool, device=age.device)
        else:
            drop_ids = force_drop_ids.bool()

        age_emb = self.age_embedder(age)
        age_emb = torch.where(drop_ids[:, None], self.age_null.expand(age.shape[0], -1), age_emb)
        view = torch.where(drop_ids, torch.full_like(view, 2), view)
        density = torch.where(drop_ids, torch.full_like(density, 4), density)
        scanner = torch.where(drop_ids, torch.full_like(scanner, 5), scanner)

        parent_emb = torch.cat([age_emb, self.view_embedder(view), self.density_embedder(density), self.scanner_embedder(scanner)], dim=-1)
        return self.parent_proj(parent_emb)


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"])
        if "fused_attn" in block_kwargs:
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SiT(nn.Module):
    def __init__(self, path_type='edm', input_size=32, patch_size=2, in_channels=4, hidden_size=1152, decoder_hidden_size=768,
                 encoder_depth=8, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000,
                 z_dims=[768], projector_dim=2048, bn_momentum=0.1, **block_kwargs):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.parent_embedder = ParentEmbedder(hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)])
        self.projectors = nn.ModuleList([build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True)
        self.bn.reset_running_stats()
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.parent_embedder.view_embedder.weight, std=0.02)
        nn.init.normal_(self.parent_embedder.density_embedder.weight, std=0.02)
        nn.init.normal_(self.parent_embedder.scanner_embedder.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def init_bn(self, latents_scale, latents_bias):
        self.bn.running_mean = latents_bias
        self.bn.running_var = (1. / latents_scale).pow(2)

    def extract_latents_stats(self):
        return dict(latents_scale=self.bn.running_var.rsqrt(), latents_bias=self.bn.running_mean)

    def unpatchify(self, x, patch_size=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h, w = self.x_embedder.grid_size
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def interpolant(self, t, path_type=None):
        if path_type == "linear":
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = 1 - t, t, -1, 1
        elif path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def forward(self, x, parents, zs, loss_kwargs, time_input=None, noises=None):
        """
        x: (N, C, H, W) unnormalized latent tensor
        parents: dictionary containing age, view, density, and scanner
        zs: external visual features
        """
        normalized_x = self.bn(x)

        if time_input is None:
            if loss_kwargs["weighting"] == "uniform":
                time_input = torch.rand((normalized_x.shape[0], 1, 1, 1))
            elif loss_kwargs["weighting"] == "lognormal":
                sigma = torch.randn((normalized_x.shape[0], 1, 1, 1)).exp()
                if loss_kwargs["path_type"] == "linear":
                    time_input = sigma / (1 + sigma)
                elif loss_kwargs["path_type"] == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(f"Weighting scheme {loss_kwargs['weighting']} not implemented.")

        time_input = time_input.to(device=normalized_x.device, dtype=normalized_x.dtype)
        noises = torch.randn_like(normalized_x) if noises is None else noises.to(device=normalized_x.device, dtype=normalized_x.dtype)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input, path_type=loss_kwargs["path_type"])
        model_input = alpha_t * normalized_x + sigma_t * noises

        if loss_kwargs["prediction"] == "v":
            model_target = d_alpha_t * normalized_x + d_sigma_t * noises
        else:
            raise NotImplementedError()

        x = self.x_embedder(model_input) + self.pos_embed
        N, T, D = x.shape
        c = self.t_embedder(time_input.flatten()) + self.parent_embedder(parents, self.training)

        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == self.encoder_depth:
                zs_tilde = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                if loss_kwargs["align_only"]:
                    break

        x = self.unpatchify(self.final_layer(x, c))
        denoising_loss = None if loss_kwargs["align_only"] else mean_flat((x - model_target) ** 2)

        proj_loss = torch.tensor(0., device=x.device)
        bsz = zs[0].shape[0]
        for z, z_tilde in zip(zs, zs_tilde):
            for z_j, z_tilde_j in zip(z, z_tilde):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= len(zs) * bsz

        return {
            "zs_tilde": zs_tilde,
            "model_output": x,
            "denoising_loss": denoising_loss,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
        }

    @torch.no_grad()
    def inference(self, x, t, parents, force_drop_ids=None):
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.parent_embedder(parents, self.training, force_drop_ids=force_drop_ids)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))

    @torch.no_grad()
    def forward_feats(self, x, t, parents, depth):
        assert 1 <= depth <= len(self.blocks)
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.parent_embedder(parents, self.training)
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == depth:
                return x
        return None


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    if isinstance(grid_size, int):
        grid_h_size = grid_w_size = grid_size
    else:
        grid_h_size, grid_w_size = grid_size

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_1(**kwargs): return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=1, num_heads=16, **kwargs)
def SiT_XL_2(**kwargs): return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
def SiT_XL_4(**kwargs): return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
def SiT_XL_8(**kwargs): return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)
def SiT_L_1(**kwargs): return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=1, num_heads=16, **kwargs)
def SiT_L_2(**kwargs): return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)
def SiT_L_4(**kwargs): return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)
def SiT_L_8(**kwargs): return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)
def SiT_B_1(**kwargs): return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=1, num_heads=12, **kwargs)
def SiT_B_2(**kwargs): return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)
def SiT_B_4(**kwargs): return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)
def SiT_B_8(**kwargs): return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)
def SiT_S_1(**kwargs): return SiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)
def SiT_S_2(**kwargs): return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)
def SiT_S_4(**kwargs): return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
def SiT_S_8(**kwargs): return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/1': SiT_XL_1, 'SiT-XL/2': SiT_XL_2, 'SiT-XL/4': SiT_XL_4, 'SiT-XL/8': SiT_XL_8,
    'SiT-L/1': SiT_L_1, 'SiT-L/2': SiT_L_2, 'SiT-L/4': SiT_L_4, 'SiT-L/8': SiT_L_8,
    'SiT-B/1': SiT_B_1, 'SiT-B/2': SiT_B_2, 'SiT-B/4': SiT_B_4, 'SiT-B/8': SiT_B_8,
    'SiT-S/1': SiT_S_1, 'SiT-S/2': SiT_S_2, 'SiT-S/4': SiT_S_4, 'SiT-S/8': SiT_S_8,
}