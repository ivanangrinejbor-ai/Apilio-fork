import os
import sys

# RVC v2 base pretrain providing the pretrained front-end (enc_p/enc_q/flow/emb_g)
# merged into decoder-only vocoder pretrains (BigVGAN, Vocos). The 32 kHz base
# uses n_fft 1024 (513 spec bins), matching the 24 kHz BigVGAN build; the
# front-end itself is sample-rate agnostic.
BASE_PRETRAIN_REPO = "lj1995/VoiceConversionWebUI"
BASE_PRETRAIN_FILE = "pretrained_v2/f0G32k.pth"


def _old_style_weight_norm_key(key):
    """Map an RVC base pretrain key (old-style torch weight_norm) onto the
    parametrizations naming used by this project: weight_g -> original0,
    weight_v -> original1 (same convention as convert_bigvgan)."""
    if key.endswith("weight_v"):
        return key[: -len("weight_v")] + "parametrizations.weight.original1"
    if key.endswith("weight_g"):
        return key[: -len("weight_g")] + "parametrizations.weight.original0"
    return key


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


def _merge_base_encoder_flow(path_g):
    """Replace the randomly initialized RVC front-end of a converted decoder
    pretrain with the RVC v2 base pretrain weights.

    The official vocoder checkpoints (BigVGAN, Vocos) only cover the decoder;
    the encoder/flow/speaker embedding stay random, which makes fine-tuning on
    small datasets slow. The RVC v2 base pretrain (pretrained_v2/f0G32k.pth)
    provides the pretrained front-end (enc_p/enc_q/flow/emb_g). Its HiFi-GAN
    decoder weights are discarded in favor of the official decoder already
    present in path_g, and old-style weight_norm keys are remapped to the
    parametrizations naming used by this project.

    Non-fatal: on any failure the decoder-only pretrain is kept.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    try:
        base_path = hf_hub_download(BASE_PRETRAIN_REPO, BASE_PRETRAIN_FILE)
    except (SystemExit, Exception) as error:
        print(f"Failed to download the RVC base pretrain (front-end stays random): {error}")
        return

    try:
        base = torch.load(base_path, map_location="cpu", weights_only=True)
        base_state = base.get("model", base)
        converted = torch.load(path_g, map_location="cpu", weights_only=True)
        converted_state = converted["model"]

        kept, skipped = [], []
        for key, value in base_state.items():
            if key.startswith("dec."):
                continue
            target = _old_style_weight_norm_key(key)
            if target not in converted_state:
                skipped.append(key)
                continue
            if tuple(value.shape) != tuple(converted_state[target].shape):
                skipped.append(key)
                continue
            converted_state[target] = value
            kept.append(target)

        for prefix in ("enc_p.", "enc_q.", "flow.", "emb_g."):
            if not any(k.startswith(prefix) for k in kept):
                print(f"Base pretrain has no {prefix}* weights; front-end stays random.")
                return

        torch.save(converted, path_g)
        print(
            f"Merged RVC base front-end into the pretrainG checkpoint: "
            f"{len(kept)} tensors ({len(skipped)} skipped)."
        )
    except (SystemExit, Exception) as error:
        print(f"Failed to merge the base pretrain front-end: {error}")


def _auto_download_convert(vocoder, path_g, path_d):
    """Download the official checkpoint and convert it into RVC pretrain format.

    Used for vocoders whose pretrain files are not bundled (BigVGAN, Vocos at
    24 kHz). Builds the G pretrain from the official weights and a matching
    D pretrain (ported from the official discriminators for BigVGAN, randomly
    initialized for Vocos). The RVC base front-end (encoder/flow/emb_g) is
    merged in afterwards so fine-tuning starts from a pretrained front-end
    instead of random weights. Any failure is reported but does not abort
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
            _merge_base_encoder_flow(path_g)
        elif vocoder == "Vocos" and not os.path.exists(path_g):
            print("Vocos 24 kHz pretrain not found, converting the charactr checkpoint...")
            source = hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin")
            from rvc.lib.tools.convert_vocos import convert

            convert(source, path_g, spk_embed_dim=109)
            _merge_base_encoder_flow(path_g)

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
