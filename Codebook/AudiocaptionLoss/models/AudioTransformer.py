#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
from torch import einsum
from collections import OrderedDict
from models.SpecAugment import SpecAugmentation
#from SpecAugment import SpecAugmentation
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange

""" Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwags):
        output = self.norm(x)
        output = self.fn(output, **kwags)
        return output


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):

        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim, hidden_dim)),
            ('ac1', nn.GELU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_dim, dim)),
            ('dropout2', nn.Dropout(dropout))
        ]))

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=12, dim_head=64, dropout=0.):
        '''
        dim: dim of input
        dim_head: dim of q, k, v
        '''
        super(Attention, self).__init__()

        inner_dim = dim_head * heads
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, inner_dim * 3)

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AudioTransformer(nn.Module):

    def __init__(self, patch_size, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        super(AudioTransformer, self).__init__()

        patch_height, patch_width = pair(patch_size)

        patch_dim = patch_height * patch_width  # 64 * 4 = 256 (16 * 16)

        self.bn0 = nn.BatchNorm2d(64) # 64?

        self.patch_embed = nn.Sequential(OrderedDict([
            ('rerange', Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)),
            ('proj', nn.Linear(patch_dim, dim))
        ]))

        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                               time_stripes_num=2,
                                               freq_drop_width=8,
                                               freq_stripes_num=2,
                                               mask_type='zero_value')

        self.pos_embedding = nn.Parameter(torch.randn(1, 125 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, spec):

        x = spec.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        x = self.patch_embed(x)
        #print('x ',x.shape)
        # assert 1==2
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        #print('cls_token ',cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)
        #print('x ',x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.blocks(x)

        x = self.to_latent(x)
        #print('x ',x.shape)
        return self.mlp_head(x)

class AudioTransformer_80(nn.Module):

    def __init__(self, patch_size, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        super(AudioTransformer_80, self).__init__()

        patch_height, patch_width = pair(patch_size)

        patch_dim = patch_height * patch_width  # 64 * 4 = 256 (16 * 16)

        self.bn0 = nn.BatchNorm2d(80) # 64?

        self.patch_embed = nn.Sequential(OrderedDict([
            ('rerange', Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)),
            ('proj', nn.Linear(patch_dim, dim))
        ]))

        self.spec_augmenter = SpecAugmentation(time_drop_width=80,
                                               time_stripes_num=2,
                                               freq_drop_width=8,
                                               freq_stripes_num=2,
                                               mask_type='zero_value')

        self.pos_embedding = nn.Parameter(torch.randn(1, 215 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, spec):

        x = spec.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        x = self.patch_embed(x)
        #print('x ',x.shape)
        # assert 1==2
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        #print('cls_token ',cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)
        #print('x ',x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.blocks(x)

        x = self.to_latent(x)
        #print('x ',x.shape)
        return self.mlp_head(x)

if __name__ == '__main__':
    num_classes = 527
    patch_size = (4, 80)
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_dim = 3072
    dropout = 0.1
    model = AudioTransformer_80(patch_size,
                             num_classes,
                             embed_dim,
                             depth,
                             num_heads,
                             mlp_dim,
                             dropout=dropout)
    feature = torch.randn(2, 860, 80)
    output = model(feature)

