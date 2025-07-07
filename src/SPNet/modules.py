import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


def get_completion_config(model_type: str):
    base_settings = {
        "T": [96, [3, 3, 9, 3], 0.1],
        "S": [96, [3, 3, 27, 3], 0.2],
        "B": [128, [3, 3, 27, 3], 0.3],
        "L": [192, [3, 3, 27, 3], 0.5],  # 0.4
    }[model_type]
    dims = []
    for i in range(4):
        dims.append(int(base_settings[0] * (2**i)))
    return dims, base_settings[1], base_settings[2]


class SPNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.projection = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)

    def forward(self, x):
        s, u = torch.std_mean(x, dim=1, keepdim=True)
        n_x = (x - u) / (s + self.eps)
        n_x = self.projection(n_x)
        x = x * n_x
        return x


class CNBlock(nn.Module):
    def __init__(self, dim, dp_rate=0.0, ks=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=ks, padding=(ks - 1) // 2, groups=dim),
            SPNorm(dim),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path = StochasticDepth(dp_rate, mode="batch")

    def forward(self, x):
        res = self.block(x)
        x = x + self.drop_path(res)
        return x


class Encoder(nn.Module):
    def __init__(self, in_chans, dims, depths, drop_rates):
        super().__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, all_dims[0], kernel_size=3, padding=1)
        self.downsample_layers.append(stem)
        for i in range(5):
            downsample_layer = nn.Sequential(
                SPNorm(all_dims[i]),
                nn.Conv2d(
                    all_dims[i],
                    all_dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        dp_rates = [x.item() for x in torch.linspace(0, drop_rates, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[CNBlock(dims[i], dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        outputs = []
        for i in range(6):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, out_chans, dims):
        super().__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.upsample_layers = nn.ModuleList()
        unstem = nn.Conv2d(all_dims[0], out_chans, kernel_size=3, padding=1)
        self.upsample_layers.append(unstem)
        for i in range(5):
            upsample_layer = nn.Sequential(
                SPNorm(all_dims[i + 1]),
                nn.ConvTranspose2d(
                    all_dims[i + 1],
                    all_dims[i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        for i in range(3):
            self.stages.append(CNBlock(dims[i], ks=7))  # ks=3
        self.stages.append(nn.Identity())

        self.fusion_layers = nn.ModuleList()
        for i in range(5):
            fusion_layer = nn.Conv2d(2 * all_dims[i], all_dims[i], kernel_size=1)
            self.fusion_layers.append(fusion_layer)
        self.fusion_layers.append(nn.Identity())

    def forward(self, ins):
        for i in range(5, -1, -1):
            if i == 5:
                x = ins[i]
            else:
                x = torch.cat([ins[i], x], dim=1)
            x = self.fusion_layers[i](x)
            x = self.stages[i](x)
            x = self.upsample_layers[i](x)
        return x
