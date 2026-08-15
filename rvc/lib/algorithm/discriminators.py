import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
        use_cqtd (bool): Append the BigVGAN-v2 multi-scale sub-band CQT
            discriminators (3x DiscriminatorCQT, 24 kHz). Defaults to False.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        checkpointing: bool = False,
        version: str = "v2",
        use_cqtd: bool = False,
    ):
        super().__init__()

        if version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
            resolutions = []
        elif version == "v2":
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
            resolutions = []
        elif version == "v3":
            periods = [2, 3, 5, 7, 11]
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]

        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [
                DiscriminatorR(r, use_spectral_norm=use_spectral_norm)
                for r in resolutions
            ]
        )
        if use_cqtd:
            self.discriminators.extend(
                MultiScaleSubbandCQTDiscriminator(
                    sample_rate=24000, use_spectral_norm=use_spectral_norm
                ).discriminators
            )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super().__init__()

        self.resolution = resolution
        self.lrelu_slope = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        1,
                        32,
                        (3, 9),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(
            x,
            (pad, pad),
            mode="reflect",
        ).squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.ones(win_length, device=x.device),
            center=False,
            return_complex=True,
        )

        mag = torch.norm(torch.view_as_real(x), p=2, dim=-1)  # [B, F, TT]

        return mag


class DiscriminatorCQT(torch.nn.Module):
    """
    Multi-band sub-band CQT discriminator (BigVGAN-v2).

    Adapted from the official NVIDIA BigVGAN implementation
    (MultiScaleSubbandCQTDiscriminator, based on Amphion). Operates on a
    constant-Q transform (amplitude + phase) of the waveform resampled to
    2x the sample rate, split into per-octave sub-bands.

    Args:
        sample_rate (int): Sample rate of the input waveform.
        hop_length (int): CQT hop length (at the 2x sample rate).
        n_octaves (int): Number of octaves of the CQT.
        bins_per_octave (int): CQT bins per octave.
        filters (int): Base number of filters. Defaults to 128.
        max_filters (int): Maximum number of filters. Defaults to 1024.
        filters_scale (int): Filter scaling factor. Defaults to 1.
        dilations (list): Dilation factors of the middle conv layers.
            Defaults to [1, 2, 4].
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(
        self,
        sample_rate,
        hop_length,
        n_octaves,
        bins_per_octave,
        filters=128,
        max_filters=1024,
        filters_scale=1,
        dilations=(1, 2, 4),
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.dilations = dilations

        try:
            from nnAudio import features
        except ImportError as error:
            raise ImportError(
                "nnAudio is required for the CQT discriminator. "
                f"Install it with: pip install nnAudio ({error})"
            ) from error

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.sample_rate * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.conv_pres = torch.nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                torch.nn.Conv2d(
                    2,
                    2,
                    kernel_size=(3, 9),
                    padding=self.get_2d_padding((3, 9)),
                )
            )

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            torch.nn.Conv2d(
                2,
                self.filters,
                kernel_size=(3, 9),
                padding=self.get_2d_padding((3, 9)),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                norm_f(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding((3, 9), (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            norm_f(
                torch.nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(3, 3),
                    padding=self.get_2d_padding((3, 3)),
                )
            )
        )

        self.conv_post = norm_f(
            torch.nn.Conv2d(
                out_chs,
                1,
                kernel_size=(3, 3),
                padding=self.get_2d_padding((3, 3)),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate, new_freq=self.sample_rate * 2
        )

    def get_2d_padding(self, kernel_size, dilation=(1, 1)):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x):
        fmap = []
        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for layer in self.convs:
            latent_z = layer(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(torch.nn.Module):
    """
    Multi-scale sub-band CQT discriminator (BigVGAN-v2, 24 kHz).

    Composed of 3x DiscriminatorCQT with the official v2 24 kHz configuration
    (hop lengths [512, 256, 256], 9 octaves, 24/36/48 bins per octave).
    """

    def __init__(self, sample_rate=24000, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = torch.nn.ModuleList(
            [
                DiscriminatorCQT(
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_octaves=n_octaves,
                    bins_per_octave=bins_per_octave,
                    use_spectral_norm=use_spectral_norm,
                )
                for hop_length, n_octaves, bins_per_octave in zip(
                    [512, 256, 256], [9, 9, 9], [24, 36, 48]
                )
            ]
        )
