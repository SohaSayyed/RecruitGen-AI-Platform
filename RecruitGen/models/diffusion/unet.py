"""
Conditional U-Net for the resume diffusion model
------------------------------------------------
input  : [x_t (noisy clean-page estimate), degraded page]  -> 2 channels, any H, W divisible by 2**(levels-1)
output : predicted clean page x0 (1 channel, [-1, 1] space)
time t : sinusoidal embedding -> MLP -> added inside every residual block
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(tdim, cout)
        self.norm2, self.conv2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, base=32, mults=(1, 2, 4), in_channels=2):
        super().__init__()
        tdim = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.stem = nn.Conv2d(in_channels, base, 3, padding=1)
        chans = [base * m for m in mults]
        self.down, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = base
        for i, c in enumerate(chans):
            self.down.append(ResBlock(prev, c, tdim))
            self.pools.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity())
            prev = c
        self.mid = ResBlock(prev, prev, tdim)
        self.up, self.ups = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            self.ups.append(nn.Conv2d(prev, chans[i], 3, padding=1))
            self.up.append(ResBlock(chans[i] * 2, chans[i], tdim))
            prev = chans[i]
        self.out = nn.Sequential(nn.GroupNorm(8, prev), nn.SiLU(), nn.Conv2d(prev, 1, 3, padding=1))

    def forward(self, x_t, cond, t):
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.stem(torch.cat([x_t, cond], dim=1))
        skips = []
        for block, pool in zip(self.down, self.pools):
            h = block(h, temb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, temb)
        for conv, block in zip(self.ups, self.up):
            h = conv(F.interpolate(h, scale_factor=2, mode="nearest"))
            h = block(torch.cat([h, skips.pop(-2)], dim=1), temb)
        # skips[-1] (bottom level) is already consumed through the mid block
        return self.out(h)
