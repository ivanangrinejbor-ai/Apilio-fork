"""Google Drive backup helpers for training checkpoints (via rclone).

Connect flow is a two-phase OAuth:
  begin_connect()  -> returns the Google authorization URL (or error)
  finish_connect(code) -> sends the code back to rclone and verifies the link

The rclone config is stored at ~/.config/rclone/rclone.conf and is shared
between the web UI process and the training subprocess.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

RCLONE_REMOTE = "gdrive"
DEFAULT_FOLDER = "RVC models"

_connect_lock = threading.Lock()
_connect_proc = None
_connect_replied = False
_mkdir_cache = set()


def rclone_exe():
    """Return the path to the rclone binary, or None."""
    return shutil.which("rclone")


def config_path():
    return os.path.expanduser(os.path.join("~", ".config", "rclone", "rclone.conf"))


def remote_configured(remote=RCLONE_REMOTE):
    """True if the rclone config file contains a section for the remote."""
    path = config_path()
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return f"[{remote}]" in f.read()
    except Exception:
        return False


def install_rclone():
    """Try to install rclone, return (ok, message)."""
    if rclone_exe():
        return True, "rclone is already installed."
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                [
                    "winget",
                    "install",
                    "rclone",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode == 0:
                return True, "rclone installed. Restart the app so it finds rclone."
            return False, f"Auto-install failed: {proc.stderr.strip()[:200]}"
        except FileNotFoundError:
            return False, (
                "rclone is not installed. Install it from https://rclone.org/downloads/ "
                "and restart the app."
            )

    # Linux/macOS: download the official binary directly (no sudo needed,
    # works as root on Kaggle) and fall back to the install script.
    import tempfile
    import urllib.request
    import zipfile

    if sys.platform == "darwin":
        url = "https://downloads.rclone.org/rclone-current-osx-amd64.zip"
    else:
        url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
    try:
        zip_path = os.path.join(tempfile.gettempdir(), "rclone-current.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            member = next(
                name for name in archive.namelist() if name.endswith("/rclone")
            )
            data = archive.read(member)
        target = "/usr/local/bin/rclone"
        with open(target, "wb") as f:
            f.write(data)
        os.chmod(target, 0o755)
        if rclone_exe():
            return True, "rclone installed to /usr/local/bin."
        return False, "rclone downloaded but the binary is not on PATH. Restart the app."
    except Exception as error:
        install_script_error = None
        try:
            proc = subprocess.run(
                "curl -L https://rclone.org/install.sh | bash",
                shell=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode == 0 and rclone_exe():
                return True, "rclone installed via install script."
            install_script_error = proc.stderr.strip()[:200]
        except Exception as script_error:
            install_script_error = str(script_error)
        return (
            False,
            "Could not install rclone (direct download failed: "
            + f"{error}; install script: {install_script_error})",
        )


def check_connection(remote=RCLONE_REMOTE):
    """Return (ok, message) by listing the remote root."""
    exe = rclone_exe()
    if not exe:
        return False, "rclone is not installed. Click 'Connect Google Drive' to install it."
    if not remote_configured(remote):
        return False, (
            "Google Drive is not connected. Click 'Connect Google Drive' to log in "
            "(full access to your Google Drive)."
        )
    proc = subprocess.run(
        [exe, "lsd", remote + ":"], capture_output=True, text=True, timeout=60
    )
    if proc.returncode == 0:
        return True, f"Connected to {remote}:"
    return False, f"Connection failed: {proc.stderr.strip()[:200]}"


def _pump_stdout(proc, buf):
    for line in proc.stdout:
        buf.append(line.decode("utf-8", errors="replace"))


def _text(buf):
    return "".join(buf)


def begin_connect(remote=RCLONE_REMOTE):
    """Start the OAuth flow. Returns (url, message); url is '' on error."""
    global _connect_proc, _connect_replied
    exe = rclone_exe()
    if not exe:
        ok, msg = install_rclone()
        if not ok:
            return "", msg
        exe = rclone_exe()
        if not exe:
            return "", "rclone install finished but the binary was not found. Restart the app."
    try:
        version = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=15
        )
        if version.returncode != 0:
            return "", (
                "rclone is installed but broken ('rclone --version' failed: "
                f"{version.stderr.strip()[:150]}). Delete it and click Connect again."
            )
    except Exception as error:
        return "", f"rclone --version failed: {error}"
    if remote_configured(remote):
        cmd = [exe, "-q", "config", "reconnect", remote + ":"]
    else:
        cmd = [exe, "-q", "config", "create", remote, "drive", "scope", "drive"]
    with _connect_lock:
        _connect_proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _connect_replied = False
    buf = []
    thread = threading.Thread(target=_pump_stdout, args=(_connect_proc, buf), daemon=True)
    thread.start()
    deadline = time.time() + 90
    while time.time() < deadline:
        text = _text(buf)
        if "accounts.google.com" in text:
            url = next(
                (line.strip() for line in text.splitlines() if "accounts.google.com" in line),
                "",
            )
            return url, (
                "Open the URL in your browser, log in with full access to your Google "
                "Drive, copy the verification code, paste it below and press 'Confirm Code'."
            )
        if not _connect_replied:
            lowered = text.lower()
            try:
                if "auto config" in lowered:
                    _connect_proc.stdin.write(b"n\n")
                    _connect_proc.stdin.flush()
                    _connect_replied = True
                elif "press enter" in lowered:
                    _connect_proc.stdin.write(b"\n")
                    _connect_proc.stdin.flush()
                    _connect_replied = True
            except Exception:
                pass
        if _connect_proc.poll() is not None:
            break
        time.sleep(0.2)
    rc = _connect_proc.poll() if _connect_proc else None
    try:
        _connect_proc.kill()
    except Exception:
        pass
    detail = _text(buf)[-500:]
    if not detail.strip():
        detail = "(rclone produced no output)"
    return "", f"Could not get the Google authorization URL (rclone exited with code {rc}): {detail}"


def finish_connect(code, remote=RCLONE_REMOTE):
    """Send the verification code back to rclone. Returns (ok, message)."""
    global _connect_proc
    with _connect_lock:
        proc = _connect_proc
        _connect_proc = None
    if proc is None:
        return False, "No pending connection flow. Click 'Connect Google Drive' again."
    try:
        if proc.poll() is None:
            proc.stdin.write((code.strip() + "\n").encode("utf-8"))
            proc.stdin.flush()
            proc.wait(timeout=90)
        out = proc.stdout.read().decode("utf-8", errors="replace")
    except Exception as e:
        return False, f"Connection flow failed: {e}"
    if proc.returncode == 0:
        ok, msg = check_connection(remote)
        return ok, msg
    return False, "Google Drive login failed: " + out[-300:]


def _mkdir_once(exe, dest):
    if dest in _mkdir_cache:
        return
    subprocess.run([exe, "mkdir", dest], capture_output=True, timeout=60)
    _mkdir_cache.add(dest)


def upload_files(paths, remote=RCLONE_REMOTE, folder=DEFAULT_FOLDER):
    """Upload a list of local files to remote:folder. Returns (ok, message)."""
    exe = rclone_exe()
    if not exe:
        return False, "rclone is not installed."
    if not remote_configured(remote):
        return False, "Google Drive is not connected."
    dest = f"{remote}:{folder}"
    _mkdir_once(exe, dest)
    failed = []
    for path in paths:
        if not os.path.exists(path):
            continue
        proc = subprocess.run(
            [exe, "copyto", path, f"{dest}/{os.path.basename(path)}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            failed.append(f"{os.path.basename(path)}: {proc.stderr.strip()[:150]}")
    if failed:
        return False, "Upload failed for: " + "; ".join(failed)
    return True, f"Uploaded {len([p for p in paths if os.path.exists(p)])} file(s) to {dest}"


def upload_async(paths, remote=RCLONE_REMOTE, folder=None, wait=False):
    """Fire-and-forget checkpoint upload used by the training loop.

    Uses GDRIVE_FOLDER env var when folder is not given. Does nothing (silently)
    when rclone is missing or Drive is not connected.
    """
    folder = folder or os.environ.get("GDRIVE_FOLDER") or DEFAULT_FOLDER
    exe = rclone_exe()
    if not exe or not remote_configured(remote):
        print("Google Drive backup: rclone missing or not connected, skipping upload.")
        return
    dest = f"{remote}:{folder}"
    _mkdir_once(exe, dest)
    for path in paths:
        if not os.path.exists(path):
            continue
        target = f"{dest}/{os.path.basename(path)}"
        if wait:
            proc = subprocess.run([exe, "copyto", path, target], capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(f"Google Drive upload failed for {path}: {proc.stderr.strip()[:150]}")
        else:
            subprocess.Popen(
                [exe, "copyto", path, target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print(f"Google Drive backup: uploading {path}")


def sync_logs(folder=DEFAULT_FOLDER, logs_root=None, remote=RCLONE_REMOTE):
    """Upload all .pth/.index files from logs/<model>/ to remote:folder/<model>/."""
    exe = rclone_exe()
    if not exe:
        return False, "rclone is not installed."
    if not remote_configured(remote):
        return False, "Google Drive is not connected."
    logs_root = logs_root or os.path.join(os.getcwd(), "logs")
    if not os.path.isdir(logs_root):
        return False, f"Logs directory not found: {logs_root}"
    base = f"{remote}:{folder}"
    _mkdir_once(exe, base)
    models = sorted(
        name
        for name in os.listdir(logs_root)
        if os.path.isdir(os.path.join(logs_root, name))
    )
    if not models:
        return False, "No trained models found in the logs directory."
    failed = []
    for name in models:
        src = os.path.join(logs_root, name)
        dest = f"{base}/{name}"
        proc = subprocess.run(
            [
                exe,
                "copy",
                src,
                dest,
                "--include",
                "*.pth",
                "--include",
                "*.index",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            failed.append(f"{name}: {proc.stderr.strip()[:150]}")
    if failed:
        return False, "Sync failed for: " + "; ".join(failed)
    return True, f"Synced {len(models)} model(s) to {base}"