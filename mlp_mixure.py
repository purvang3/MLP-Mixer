import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Mixure_Layer(nn.Module):
    def __init__(self, dim, mlp_dim, num_patches, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.channel_mix = FeedForward(num_patches, mlp_dim, dropout)
        self.token_mix = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = rearrange(x, 'b s c -> b c s')
        x = self.channel_mix(x)
        x = rearrange(x, 'b c s -> b s c')
        x += skip
        x = self.norm(x)
        x = self.token_mix(x)
        return x


class Model(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim, channels=3,
                 dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = int((image_size * image_size * channels) / (patch_size * patch_size * channels))
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pool = 'mean'
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Mixure_Layer(dim, mlp_dim, num_patches, dropout))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixure in self.layers:
            x = mixure(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)
        return x
