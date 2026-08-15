import os
import sys


def _save_discriminator_pretrain(path, version):
    """Save a randomly initialized MultiPeriodDiscriminator as a D pretrain."""
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator

    import torch

    net_d = MultiPeriodDiscriminator(
        use_spectral_norm=False, checkpointing=False, version=version
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {"model": {k: v.detach().cpu() for k, v in net_d.state_dict().items()}},
        path,
    )
    print(f"Saved discriminator stub (D): {path}")


def _auto_download_convert(vocoder, path_g, path_d):
    """Download the official checkpoint and convert it into RVC pretrain format.

    Used for vocoders whose pretrain files are not bundled (BigVGAN, Vocos at
    24 kHz). Builds the G pretrain from the official weights and a matching
    randomly initialized D stub. Any failure is reported but does not abort
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
            version = "v3" if vocoder == "BigVGAN" else "v2"
            _save_discriminator_pretrain(path_d, version)
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
