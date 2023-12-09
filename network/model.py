import torch
import math
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention_context(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, context, skeleton):

        B_, N, C = context.shape
        query = self.q(skeleton).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        kv = self.kv(context).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = query[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class EmotionNetwork(nn.Module):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 num_slices=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.num_slices = num_slices
        self.max_support_slices = num_slices // 2


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers


        self.end_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
                                      nn.PixelShuffle(patch_size),
                                      nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                                      )

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.g_a = None
        self.g_s = None

        frames = 16
        # batch x 16 x 50 x 768 -> batch x 15 x 50 x 384
        self.context_conv1 = nn.Conv2d(frames,frames-1,kernel_size=(3,5),padding=(1,2),stride=(1,2))
        # batch x 15 x 50 x 384 -> batch x 15 x 50 x 192
        self.context_conv2 = nn.Conv2d(frames-1, frames - 1, kernel_size=(3, 5), padding=(1, 2), stride=(1, 2))
        # batch x 15 x 50 x 192 -> batch x 15 x 25 x 96
        self.context_conv3 = nn.Conv2d(frames - 1, frames - 1, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        # batch x 15 x 25 x 96 -> batch x 15 x 25 x 48
        self.context_conv4 = nn.Conv2d(frames - 1, frames - 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 2))
        # batch x 15 x 25 x 48 -> batch x 15 x 17 x 24
        self.context_conv5 = nn.Conv2d(frames - 1, frames - 1, kernel_size=(3, 3), padding=(5, 1), stride=(2, 2))

        self.contextTransformer = WindowAttention_context(15,(17,24),1)
        self.norm = norm_layer(15)
        self.mlp = Mlp(in_features=15, hidden_features=15, act_layer=nn.GELU, drop=0)

        self.transformer1 = WindowAttention(15, (17, 24), 1)
        self.norm1 = norm_layer(15)
        self.mlp1 = Mlp(in_features=15, hidden_features=15, act_layer=nn.GELU, drop=0)

        self.transformer2 = WindowAttention(15, (17, 24), 1)
        self.norm2 = norm_layer(15)
        self.mlp2 = Mlp(in_features=15, hidden_features=15, act_layer=nn.GELU, drop=0)

        self.transformer3 = WindowAttention(15, (17, 24), 1)
        self.norm3 = norm_layer(15)
        self.mlp3 = Mlp(in_features=15, hidden_features=15, act_layer=nn.GELU, drop=0)

        # self.fc1 = nn.Linear(15, 1)
        # self.fc2 = nn.Linear(17*24, 17*2)
        # self.end_layers = nn.Sequential(
        #                         nn.Linear(17 * 24, 17 * 2),
        #                         nn.GELU(),
        #                         nn.Linear(17 * 2, 17 * 1),
        #                         nn.GELU(),
        #                         nn.Linear(17 * 1, 3),
        # )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(15, 6)



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, context,skeleton):
        """Forward function."""
        x = self.context_conv1(context)
        x = self.context_conv2(x)
        x = self.context_conv3(x)
        x = self.context_conv4(x)
        x = self.context_conv5(x)
        #
        # x = torch.zeros_like(x,device=x.device)
        # print(x.shape)
        # print(skeleton.shape)

        #
        y = skeleton.repeat(1,1,1,4)
        # print(y.shape)

        x = x.permute(0, 2, 3, 1).contiguous().view(-1, 17 * 24, 15)

        y = y.permute(0, 2, 3, 1).contiguous().view(-1, 17 * 24, 15)
        # x = torch.zeros_like(y, device=y.device)
        shortcut = x
        y = self.contextTransformer(x,y)
        # y = y + x
        y = y + self.mlp(self.norm(y))

        y = self.transformer1(y)
        y = y + self.mlp1(self.norm1(y))

        y = self.transformer2(y)
        y = y + self.mlp2(self.norm2(y))

        y = self.transformer3(y)
        y = y + self.mlp3(self.norm3(y))

        y = shortcut + y

        output = self.avgpool(y.transpose(1, 2))  # [B, C, 1]
        output = torch.flatten(output, 1)
        output = self.head(output)

        return output

