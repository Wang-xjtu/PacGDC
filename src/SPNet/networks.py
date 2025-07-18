import torch
import torch.nn as nn
from .modules import Encoder, Decoder, get_completion_config
from huggingface_hub import PyTorchModelHubMixin


class CompletionNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_type="T",
    ) -> None:
        super().__init__()
        dims, depths, drop_rates = get_completion_config(model_type)
        self.encoder = Encoder(5, dims, depths, drop_rates)
        self.decoder = Decoder(1, dims)

        # initializing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, rgb, raw, hole_raw):
        x = torch.cat([rgb, raw, hole_raw], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
