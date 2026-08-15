"""
Convert official NVIDIA BigVGAN discriminator checkpoints into an Applio/RVC
pretrainD checkpoint ({"model": <MultiPeriodDiscriminator state dict>}).

The official checkpoints ("bigvgan_discriminator_optimizer.pt", containing the
"mpd"/"mrd" state dicts) use the old-style torch weight_norm ("weight_v" /
"weight_g" keys), while this project's MultiPeriodDiscriminator uses new-style
parametrizations. This script remaps the shared sub-discriminators and leaves
everything else randomly initialized for fine-tuning:

- BigVGAN-v2 MPD: 5x DiscriminatorP (periods [2, 3, 5, 7, 11]) -> project
  indices 1..5 (the project's DiscriminatorS sits at index 0 and has no NVIDIA
  counterpart). This works for both the v2 discriminator (used by the Vocos
  vocoder) and the v3 discriminator (used by the BigVGAN vocoder).
- BigVGAN-v1 MRD (optional, v3 only): 3x DiscriminatorR (resolutions
  [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]) -> project v3 indices
  6..8. The v2 checkpoint replaces MRD with a CQT discriminator that has no
  RVC counterpart, hence the v1 checkpoint is used for this part.

The discriminator never sees the generator's latent/NSF inputs, so this
transfer is independent of the RVC generator modifications (z=192, F0, NSF).

Usage:
    # BigVGAN (v3: MPD + MRD):
    python rvc/lib/tools/convert_bigvgan_d.py \
        --output assets/pretrained_bigvgan/pretrained_bigvgan_24k_d.pth

    # Vocos (v2: MPD only, downloads are automatic):
    python rvc/lib/tools/convert_bigvgan_d.py --version v2 \
        --output assets/pretrained_vocos/pretrained_vocos_24k_d.pth

    # or with local files:
    python rvc/lib/tools/convert_bigvgan_d.py \
        --input-v2 /path/to/bigvgan_discriminator_optimizer.pt \
        --input-v1 /path/to/bigvgan_v1_discriminator_optimizer.pt \
        --output assets/pretrained_bigvgan/pretrained_bigvgan_24k_d.pth
"""

import argparse
import os
import sys

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator

# Index offsets into the project's MultiPeriodDiscriminator.
# Project layout (v3): 0 = DiscriminatorS, 1..5 = DiscriminatorP (periods
# [2,3,5,7,11]), 6..8 = DiscriminatorR. The official MPD holds only the 5
# DiscriminatorP modules, the official MRD holds the 3 DiscriminatorR modules.
MPD_INDEX_OFFSET = 1
MRD_INDEX_OFFSET = 6


def _remap_weight_norm(rest: str) -> str:
    # Old-style weight_norm (weight = g * v / ||v||) -> new-style
    # parametrizations (original0 = g, original1 = v).
    if rest.endswith("weight_v"):
        return rest[: -len("weight_v")] + "parametrizations.weight.original1"
    if rest.endswith("weight_g"):
        return rest[: -len("weight_g")] + "parametrizations.weight.original0"
    return rest


def official_to_project(official_key: str, index_offset: int) -> str:
    parts = official_key.split(".")
    if parts and parts[0] == "module":
        parts = parts[1:]
    if len(parts) < 3 or parts[0] != "discriminators":
        raise AssertionError(f"unexpected official key: {official_key}")
    sub_index = int(parts[1])
    rest = ".".join(parts[2:])
    return f"discriminators.{sub_index + index_offset}.{_remap_weight_norm(rest)}"


def transfer(official_sub: dict, project_state: dict, mapped_keys: set, offset: int):
    """Copy compatible tensors from an official sub state dict into the project
    state dict. Returns (mapped_count, skipped_count)."""
    mapped = 0
    skipped = 0
    for key, value in official_sub.items():
        if not key.startswith("discriminators."):
            skipped += 1
            continue
        target = official_to_project(key, offset)
        if target not in project_state:
            sys.exit(f"Key mismatch: '{key}' -> '{target}' not in project discriminator")
        if tuple(value.shape) != tuple(project_state[target].shape):
            sys.exit(
                f"Shape mismatch: '{key}' -> '{target}' "
                f"({tuple(value.shape)} vs {tuple(project_state[target].shape)})"
            )
        project_state[target] = value
        mapped_keys.add(target)
        mapped += 1
    return mapped, skipped


def load_official_state(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        sys.exit(f"Unrecognized checkpoint format: {path}")
    return ckpt


def convert(input_v2: str, input_v1: str | None, output_path: str, version: str) -> None:
    if version == "v2":
        print("NOTE: version 'v2' has no DiscriminatorR modules; MRD transfer is skipped.")

    print(f"Loading official BigVGAN-v2 checkpoint: {input_v2}")
    ckpt_v2 = load_official_state(input_v2)
    mpd = ckpt_v2.get("mpd")
    if not isinstance(mpd, dict):
        sys.exit(f"Checkpoint does not contain an 'mpd' state dict: {sorted(ckpt_v2)[:8]}")

    print(f"Building RVC MultiPeriodDiscriminator (version={version})...")
    net_d = MultiPeriodDiscriminator(
        use_spectral_norm=False, checkpointing=False, version=version
    )
    project_state = net_d.state_dict()
    mapped_keys = set()
    skipped_total = 0

    print("Transferring MPD (5x DiscriminatorP, periods 2/3/5/7/11)...")
    mapped, skipped = transfer(mpd, project_state, mapped_keys, MPD_INDEX_OFFSET)
    skipped_total += skipped

    if version == "v3" and input_v1:
        if os.path.isfile(input_v1):
            print(f"Loading official BigVGAN-v1 checkpoint (MRD): {input_v1}")
            ckpt_v1 = load_official_state(input_v1)
            mrd = ckpt_v1.get("mrd")
            if not isinstance(mrd, dict):
                print(
                    f"NOTE: v1 checkpoint has no 'mrd' state dict "
                    f"({sorted(ckpt_v1)[:8]}); DiscriminatorR stays random."
                )
            else:
                print("Transferring MRD (3x DiscriminatorR)...")
                mapped2, skipped2 = transfer(mrd, project_state, mapped_keys, MRD_INDEX_OFFSET)
                mapped += mapped2
                skipped_total += skipped2
        else:
            print(f"NOTE: v1 checkpoint not found ({input_v1}); DiscriminatorR stays random.")

    missing = sorted(set(project_state) - mapped_keys)
    print(f"Transferred {len(mapped_keys)} tensors ({skipped_total} non-discriminator keys skipped).")
    if missing:
        print(
            f"Left randomly initialized ({len(missing)}): "
            + ", ".join(missing[:8])
            + ("..." if len(missing) > 8 else "")
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(
        {
            "model": dict(project_state),
            "epoch": 0,
            "iteration": 0,
            "optimizer": {},
            "scaler": {},
            "learning_rate": 1e-4,
        },
        output_path,
    )
    print(f"Saved pretrainD checkpoint: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA BigVGAN discriminator checkpoints into an RVC pretrainD checkpoint."
    )
    parser.add_argument(
        "--input-v2",
        type=str,
        help="Path to the BigVGAN-v2 bigvgan_discriminator_optimizer.pt (used when --hf-repo-v2 is not wanted).",
    )
    parser.add_argument(
        "--input-v1",
        type=str,
        help="Optional path to the BigVGAN-v1 bigvgan_discriminator_optimizer.pt (MRD part).",
    )
    parser.add_argument(
        "--hf-repo-v2",
        type=str,
        default="nvidia/bigvgan_v2_24khz_100band_256x",
        help="Hugging Face repo id for the v2 checkpoint (used when --input-v2 is not given).",
    )
    parser.add_argument(
        "--hf-repo-v1",
        type=str,
        default="nvidia/bigvgan_24khz_100band",
        help="Hugging Face repo id for the v1 checkpoint (used when --input-v1 is not given).",
    )
    parser.add_argument(
        "--no-mrd",
        action="store_true",
        help="Skip the optional BigVGAN-v1 MRD transfer (DiscriminatorR stays random).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/pretrained_bigvgan/pretrained_bigvgan_24k_d.pth",
        help="Output RVC pretrainD checkpoint path.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3",
        choices=["v2", "v3"],
        help="Project discriminator version to build (default: v3, used by the BigVGAN vocoder).",
    )
    args = parser.parse_args()

    input_v2 = args.input_v2
    if not input_v2:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            sys.exit(f"huggingface_hub is required for --hf-repo-v2: {error}")
        input_v2 = hf_hub_download(args.hf_repo_v2, "bigvgan_discriminator_optimizer.pt")
        print(f"Downloaded: {input_v2}")

    input_v1 = args.input_v1
    if not args.no_mrd and not input_v1:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            sys.exit(f"huggingface_hub is required for --hf-repo-v1: {error}")
        try:
            input_v1 = hf_hub_download(args.hf_repo_v1, "bigvgan_discriminator_optimizer.pt")
            print(f"Downloaded: {input_v1}")
        except (SystemExit, Exception) as error:
            print(f"NOTE: failed to download the v1 (MRD) checkpoint: {error}")
            input_v1 = None

    convert(input_v2, input_v1, args.output, args.version)


if __name__ == "__main__":
    main()