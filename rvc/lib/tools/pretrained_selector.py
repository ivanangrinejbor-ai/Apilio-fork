import os
import sys


def _save_discriminator_pretrain(path, version):
    """Save a randomly initialized MultiPeriodDiscriminator as a D pretrain."""
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator

    import torch

    net_d = MultiPeriodDiscriminator(
        use_spectral_norm=False,
        checkpointing=False,
        version=version,
        use_cqtd=(version == "v3"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {"model": {k: v.detach().cpu() for k, v in net_d.state_dict().items()}},
        path,
    )
    print(f"Saved discriminator stub (D): {path}")


def _auto_download_convert_d(path_d, version):
    """Download the official BigVGAN discriminator checkpoint and convert it
    into an RVC pretrainD checkpoint.

    The BigVGAN MPD (5x DiscriminatorP, periods 2/3/5/7/11) is the only
    pretrained HiFi-GAN-style MPD at 24 kHz and fits both the v3 discriminator
    used by the BigVGAN vocoder (indices 1..5) and the v2 discriminator used by
    the Vocos vocoder (same indices). For version v3 the optional BigVGAN-v1
    MRD (3x DiscriminatorR) is transferred as well.

    Non-fatal: on any failure a randomly initialized D stub is saved instead,
    so training can still start.
    """
    try:
        from rvc.lib.tools.convert_bigvgan_d import convert
    except ImportError:
        print("Failed to import convert_bigvgan_d, using a random D stub.")
        _save_discriminator_pretrain(path_d, version)
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required to download the pretrained model. "
            "Install it or place the pretrain files manually."
        )
        return

    try:
        source_v2 = hf_hub_download(
            "nvidia/bigvgan_v2_24khz_100band_256x", "bigvgan_discriminator_optimizer.pt"
        )
        source_v1 = None
        if version == "v3":
            try:
                source_v1 = hf_hub_download(
                    "nvidia/bigvgan_24khz_100band", "bigvgan_discriminator_optimizer.pt"
                )
            except (SystemExit, Exception) as error:
                print(
                    f"BigVGAN-v1 (MRD) checkpoint unavailable: {error}. "
                    "Only the MPD part will be transferred."
                )
        convert(source_v2, source_v1, path_d, version=version)
    except (SystemExit, Exception) as error:
        print(f"Failed to obtain the pretrained discriminator: {error}")
        _save_discriminator_pretrain(path_d, version)


def _auto_download_convert(vocoder, path_g, path_d):
    """Download the official checkpoint and convert it into RVC pretrain format.

    Used for vocoders whose pretrain files are not bundled (BigVGAN, Vocos at
    24 kHz). Builds the G pretrain from the official weights and a matching
    D pretrain (ported from the official discriminators for BigVGAN, randomly
    initialized for Vocos). Any failure is reported but does not abort
    training.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required to download the pretrained model. "
            "Install it or place the pretrain files manually."
        )
        return

    try:
        if vocoder == "BigVGAN" and not os.path.exists(path_g):
            print("BigVGAN 24 kHz pretrain not found, converting the NVIDIA checkpoint...")
            source = hf_hub_download(
                "nvidia/bigvgan_v2_24khz_100band_256x", "bigvgan_generator.pt"
            )
            from rvc.lib.tools.convert_bigvgan import convert

            convert(source, path_g, spk_embed_dim=109)
        elif vocoder == "Vocos" and not os.path.exists(path_g):
            print("Vocos 24 kHz pretrain not found, converting the charactr checkpoint...")
            source = hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin")
            from rvc.lib.tools.convert_vocos import convert

            convert(source, path_g, spk_embed_dim=109)

        if not os.path.exists(path_d) and os.path.exists(path_g):
            if vocoder == "BigVGAN":
                _auto_download_convert_d(path_d, "v3")
            elif vocoder == "Vocos":
                _auto_download_convert_d(path_d, "v2")
            else:
                _save_discriminator_pretrain(path_d, "v2")
    except (SystemExit, Exception) as error:
        print(f"Failed to obtain the pretrained model: {error}")


def pretrained_selector(vocoder, sample_rate):
    base_path = os.path.join("rvc", "models", "pretraineds", f"{vocoder.lower()}")

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d

    if vocoder in ("BigVGAN", "Vocos") and sample_rate == 24000:
        _auto_download_convert(vocoder, path_g, path_d)
        if os.path.exists(path_g) and os.path.exists(path_d):
            return path_g, path_d
        print("Training will start without a pretrained model.")

    return "", ""
