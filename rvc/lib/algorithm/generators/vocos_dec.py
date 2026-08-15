import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block adapted to 1D audio signals.

    Applies a depthwise convolution followed by a pointwise (linear) expansion,
    GELU activation, pointwise projection and learnable layer scaling.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float): Initial value for the layer scale.
            Non-positive values disable layer scaling. Defaults to 1e-6.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    """
    Vocos backbone built with ConvNeXt blocks.

    Preserves the temporal resolution of the input while mapping it into the
    model dimension.

    Args:
        input_channels (int): Number of input feature channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling.
            Defaults to 1 / num_layers.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float = None,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class ISTFT(nn.Module):
    """
    Inverse Short Time Fourier Transform with "same" padding.

    Reconstructs a waveform from a complex spectrogram through overlap-add with
    window envelope normalization, trimming the padded samples so that T input
    frames produce exactly T * hop_length output samples.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same".
            Defaults to "same".
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        pad = (self.win_length - self.hop_length) // 2

        B, N, T = spec.shape

        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window.to(ifft.dtype)[None, :, None]

        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.zeros(B, output_size, dtype=ifft.dtype, device=ifft.device)
        for t in range(T):
            start = t * self.hop_length
            y[:, start : start + self.win_length] += ifft[:, :, t]
        y = y[:, pad:-pad]

        window_envelope = torch.zeros(
            output_size, dtype=self.window.dtype, device=self.window.device
        )
        for t in range(T):
            start = t * self.hop_length
            window_envelope[start : start + self.win_length] += self.window.square()
        window_envelope = window_envelope[pad:-pad]

        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT head module for predicting STFT complex coefficients.

    Predicts magnitude and phase coefficients from the backbone features and
    reconstructs the waveform with an ISTFT.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        S = mag.float() * torch.complex(torch.cos(p).float(), torch.sin(p).float())
        audio = self.istft(S)
        return audio.to(x.dtype)


class VocosDecoder(nn.Module):
    """
    Vocos neural vocoder used as the RVC decoder.

    Directly maps the latent representation (in place of a mel spectrogram)
    back into a waveform at the resolution of the model's hop length, ignoring
    fundamental frequency and speaker conditioning.

    Args:
        in_channels (int): Number of latent feature channels.
        upsample_rates (list): Upsampling rates whose product defines the hop
            length between frames.
        dim (int, optional): Hidden dimension of the model. Defaults to 512.
        intermediate_dim (int, optional): Intermediate dimension used in the
            ConvNeXt blocks. Defaults to 1536.
        num_layers (int, optional): Number of ConvNeXt blocks. Defaults to 8.
    """

    def __init__(
        self,
        in_channels: int,
        upsample_rates: list,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
    ):
        super().__init__()
        self.hop_length = int(np.prod(upsample_rates))
        n_fft = self.hop_length * 4

        self.backbone = VocosBackbone(
            input_channels=in_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
        )
        self.head = ISTFTHead(
            dim=dim,
            n_fft=n_fft,
            hop_length=self.hop_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor = None,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.backbone(x)
        audio = self.head(x)
        return audio.unsqueeze(1)

    def remove_weight_norm(self):
        pass
