"""
Upload models (or any files/directories) from a local path to Google Drive
using rclone. Works on Kaggle notebooks and on a local machine.

Setup (one time):
  1. Install rclone on your PC:  https://rclone.org/downloads/
     (Windows: winget install rclone)
  2. Create the remote (authenticate with your Google account):
        rclone config
     name it "gdrive", scope "Full access to all files" or "Drive file",
     and DO NOT set a shared folder.
  3. Show the generated config:   rclone config show
     (on Windows it lives in %USERPROFILE%\.config\rclone\rclone.conf)
  4. On Kaggle, run in a cell to make rclone available:
        !curl https://rclone.org/install.sh | sudo bash
     then create the config file there (same content as step 3):
        !mkdir -p ~/.config/rclone && cat > ~/.config/rclone/rclone.conf << 'EOF'
        ...paste config from step 3...
        EOF

Usage:
  python tools/upload_to_gdrive.py <file_or_dir> [more...] \
      [--remote gdrive] [--folder "RVC models"]

Examples:
  python tools/upload_to_gdrive.py logs/my-project/my-project_best.pth
  python tools/upload_to_gdrive.py logs/my-project --folder "RVC models"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Upload files/directories to Google Drive via rclone."
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to upload")
    parser.add_argument(
        "--remote",
        default="gdrive",
        help="rclone remote name (default: gdrive; add ':' yourself or not)",
    )
    parser.add_argument(
        "--folder",
        default="",
        help="Destination folder in Drive (created if missing), e.g. 'RVC models'",
    )
    args = parser.parse_args()

    remote = args.remote if args.remote.endswith(":") else args.remote + ":"
    dest = remote + args.folder if args.folder else remote
    if args.folder:
        run(["rclone", "mkdir", dest])

    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"SKIP (not found): {path}", file=sys.stderr)
            continue
        if path.is_dir():
            run(["rclone", "copy", str(path), dest, "--create-empty-src-dirs"])
        else:
            run(["rclone", "copyto", str(path), f"{dest}/{path.name}"])
        print(f"UPLOADED: {path} -> {dest}/{path.name if path.is_file() else ''}")


if __name__ == "__main__":
    main()
