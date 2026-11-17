"""SST (Scalable SR Transformer, "Rank-Factorized Implicit Neural Bias", arXiv 2603.06738;
github.com/dslisleedh/SST, MIT) re-implemented for inference and ONNX export.

Same module tree and parameter names as the upstream `sst_arch.py` (attn_type RIB, gate DWC,
upsampler pixelshuffle), so its checkpoints load with load_state_dict. Differences, none of
which change the function:
  - attention is plain matmul + softmax in the input dtype (upstream concatenates the RIB
    factors to q/k and calls FlashAttention in bf16);
  - the RIB factors (input-independent) are computed in forward, so an exporter folds them into
    two small tables per layer (q side pre-scaled by rank^-0.5), and the attention is
    concat([q * hd^-0.5, qpos]) @ concat([k, kpos])^T.
No shifted windows: the presets mix window sizes, which turns upstream's shift off.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    # name: dim, window_sizes, heads, ranks, blocks, exp_ratio, intermediate_dim
    "sst": (180, [16, 32, 64, 16, 32, 64], 6, [18, 18, 18, 34, 34, 34], 6, 1.25, 64),
    "sst_plus": (180, [16, 32, 48, 32, 48, 96], 6, [18, 18, 18, 34, 34, 34], 6, 1.25, 64),
    "sst_l": (192, [16, 32, 64, 16, 32, 64], 6, [16, 16, 16, 32, 32, 32], 8, 2, 96),
    "sst_l_plus": (192, [16, 32, 48, 32, 48, 96], 6, [16, 16, 16, 32, 32, 32], 8, 2, 96),
    "sst_xl_plus": (224, [16, 32, 48, 32, 48, 96], 7, [16, 16, 16, 32, 32, 32], 10, 2, 96),
}


class LayerNorm(nn.Module):  # channels-first, eps 1e-6
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps, self.dim = eps, dim

    def forward(self, x):
        return F.layer_norm(x.transpose(1, -1), (self.dim,), self.weight, self.bias, self.eps).transpose(1, -1)


class WindowAttention2D(nn.Module):
    def __init__(self, dim, window_size, num_heads, rank, rib_hidden_dim=32, rib_n_freqs=10):
        super().__init__()
        self.dim, self.ws, self.heads, self.rank = dim, window_size, num_heads, rank
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)
        ws = window_size
        yy, xx = torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()
        coords = 2.0 * (coords + 0.5) / ws - 1.0
        base = coords.clone()
        for i in range(rib_n_freqs):
            coords = torch.cat([coords, torch.sin(base * 2**i), torch.cos(base * 2**i)], dim=-1)
        self.register_buffer("rib_coords", coords, persistent=False)
        self.to_hidden = nn.Parameter(torch.randn(2 + 4 * rib_n_freqs, rib_hidden_dim) * 0.05)
        self.hidden_b = nn.Parameter(torch.zeros(1, rib_hidden_dim))
        self.to_q = nn.Parameter(torch.randn(num_heads, rib_hidden_dim, rank) * 0.05)
        self.to_k = nn.Parameter(torch.randn(num_heads, rib_hidden_dim, rank) * 0.05)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def rib(self):
        hid = F.relu(self.rib_coords @ self.to_hidden + self.hidden_b)  # N, hidden
        qpos = torch.einsum("nd,hdr->hnr", hid, self.to_q) * (self.rank ** -0.5)
        kpos = torch.einsum("nd,hdr->hnr", hid, self.to_k)
        return qpos, kpos  # heads, N, R

    def forward(self, x):
        b, c, h, w = x.shape
        ws, nh = self.ws, self.heads
        hd = c // nh
        gate = self.gate(x)
        ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        H, W = h + ph, w + pw
        qkv = self.to_qkv(xp)  # b, 3c, H, W
        qkv = qkv.reshape(b, 3, nh, hd, H // ws, ws, W // ws, ws).permute(1, 0, 4, 6, 2, 5, 7, 3)
        qkv = qkv.reshape(3, -1, nh, ws * ws, hd)  # 3, b*nw, heads, N, hd
        q, k, v = qkv[0], qkv[1], qkv[2]
        nw = q.shape[0]
        qpos, kpos = self.rib()
        qc = torch.cat([q * (hd ** -0.5), qpos.to(q.dtype).unsqueeze(0).expand(nw, -1, -1, -1)], dim=-1)
        kc = torch.cat([k, kpos.to(k.dtype).unsqueeze(0).expand(nw, -1, -1, -1)], dim=-1)
        if torch.onnx.is_in_onnx_export():
            o = torch.softmax(qc @ kc.transpose(-2, -1), dim=-1) @ v  # b*nw, heads, N, hd
        else:  # a few windows at a time: a 96x96 window's scores are 2 GB in fp32
            step = max(1, (1 << 29) // (nh * ws ** 4))
            o = torch.cat([torch.softmax(qc[i:i + step] @ kc[i:i + step].transpose(-2, -1), dim=-1) @ v[i:i + step]
                           for i in range(0, nw, step)])
        o = o.reshape(b, H // ws, W // ws, nh, ws, ws, hd).permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, H, W)
        o = o[:, :, :h, :w]
        return self.to_out(o * gate)


class ConvFFN(nn.Module):
    def __init__(self, dim, exp_ratio):
        super().__init__()
        hid = int(dim * exp_ratio)
        self.proj = nn.Conv2d(dim, hid, 1)
        self.dwc = nn.Conv2d(hid, hid, 3, 1, 1, groups=hid)
        self.agg = nn.Conv2d(hid, dim, 1)

    def forward(self, x):
        x = F.gelu(self.proj(x))
        x = x + F.gelu(self.dwc(x))
        return self.agg(x)


class STL(nn.Module):
    def __init__(self, dim, ws, heads, rank, exp_ratio):
        super().__init__()
        self.norm_attn = LayerNorm(dim)
        self.attn = WindowAttention2D(dim, ws, heads, rank)
        self.norm_ffn = LayerNorm(dim)
        self.ffn = ConvFFN(dim, exp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        return x + self.ffn(self.norm_ffn(x))


class SSTB(nn.Sequential):
    def __init__(self, dim, wins, heads, ranks, exp_ratio):
        super().__init__(*[STL(dim, ws, heads, r, exp_ratio) for ws, r in zip(wins, ranks)], nn.Conv2d(dim, dim, 3, 1, 1))

    def forward(self, x):
        return x + super().forward(x)


class Upsampler(nn.Module):
    def __init__(self, dim, scale, num_feat):
        super().__init__()
        self.scale = scale
        self.feature_conv = nn.Conv2d(dim, num_feat, 3, 1, 1)
        self.up_conv = nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1)
        self.final_conv = nn.Conv2d(num_feat, 3, 3, 1, 1)

    def forward(self, x, skip):
        x = F.leaky_relu(self.feature_conv(x), 0.1)
        x = F.pixel_shuffle(self.up_conv(x), self.scale)
        return self.final_conv(x) + F.interpolate(skip, scale_factor=self.scale, mode="nearest")


class SST(nn.Module):
    def __init__(self, dim, window_sizes, num_heads, ranks, n_blocks, exp_ratio, intermediate_dim, scale=4):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        self.body = nn.Sequential(*[SSTB(dim, window_sizes, num_heads, ranks, exp_ratio) for _ in range(n_blocks)],
                                  LayerNorm(dim), nn.Conv2d(dim, dim, 3, 1, 1))
        self.upsampler = Upsampler(dim, scale, intermediate_dim)

    def forward(self, x):
        feat = self.proj(x)
        feat = feat + self.body(feat)
        return self.upsampler(feat, x)


def build(name, scale=4):
    dim, wins, heads, ranks, blocks, e, inter = PRESETS[name]
    return SST(dim, wins, heads, ranks, blocks, e, inter, scale)
