# Make sure the config file exists
import os
import shutil
import sys

# We need the CWD for finding the config file, but while we're at it, add it to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# TODO: This path is regenerated all over the place in Applio
# should probably be in a static module for everything to reference
CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

# The base config file to start from
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")

if not os.path.exists(CONFIG_PATH):
    print("Config file not found. Creating fresh from template.")
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

# Plataform config
from rvc.lib.platform import platform_config

platform_config()

import argparse
import types
import gradio as gr
import pathlib
import logging

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

_ARG_PARSER = argparse.ArgumentParser(
    description="Applio Web UI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_ARG_PARSER.add_argument(
    "--port", type=int, default=DEFAULT_PORT, help="Server port (default: %(default)s)"
)
_ARG_PARSER.add_argument(
    "--server-name",
    type=str,
    default=DEFAULT_SERVER_NAME,
    help="Server hostname (default: %(default)s)",
)
_ARG_PARSER.add_argument(
    "--share", action="store_true", help="Create a public Gradio share link"
)
_ARG_PARSER.add_argument(
    "--open", action="store_true", help="Open the browser automatically"
)
_ARG_PARSER.add_argument(
    "--client", action="store_true", help="Enable client mode (mounts realtime API)"
)
_ARG_PARSER.add_argument(
    "--username",
    type=str,
    default="",
    help="Enable authentication with this username (requires --password)",
)
_ARG_PARSER.add_argument(
    "--password",
    type=str,
    default="",
    help="Password for authentication (requires --username)",
)
_args, _ = _ARG_PARSER.parse_known_args()
client_mode = _args.client
_has_share = _args.share
_has_open = _args.open

# Authentication: CLI arguments override the config file
from assets.auth import (
    auth_enabled,
    check_credentials,
    generate_api_token,
    load_auth_config,
    set_api_token,
)

if _args.username or _args.password:
    import json as _json

    _auth_config = load_auth_config()
    _auth_config["enabled"] = bool(_args.username and _args.password)
    _auth_config["username"] = _args.username
    _auth_config["password"] = _args.password
    _config_path = os.path.join(now_dir, "assets", "config.json")
    with open(_config_path, "r", encoding="utf-8") as _f:
        _cfg = _json.load(_f)
    _cfg["auth"] = _auth_config
    with open(_config_path, "w", encoding="utf-8") as _f:
        _json.dump(_cfg, _f, indent=2, ensure_ascii=False)

_auth = None
if auth_enabled():
    _auth = check_credentials
    if client_mode:
        set_api_token(generate_api_token())
    print("Authentication enabled. Log in with the configured username and password.")

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress ConnectionResetError on Windows when a remote peer forcibly closes the
# connection during asyncio shutdown (WinError 10054 / ProactorBasePipeTransport).
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _orig_ccl = _pe._ProactorBasePipeTransport._call_connection_lost

    def _ccl_patched(self, exc):
        try:
            _orig_ccl(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _ccl_patched

# Fix Gradio NoneType error when entering an invalid value
gr.Number.preprocess = types.MethodType(
    lambda self, payload: (
        None
        if payload is None
        or (self.minimum is not None and payload < self.minimum)
        or (self.maximum is not None and payload > self.maximum)
        else self.round_to_precision(payload, self.precision)
    ),
    gr.Number,
)

# detect gradio
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab
from tabs.realtime.realtime import realtime_tab
from tabs.tensorboard.tensorboard import tensorboard_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Start Discord presence if enabled
from tabs.settings.sections.presence import load_config_presence

if load_config_presence():
    from assets.discord_presence import RPCManager

    RPCManager.start_presence()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "ParityError/Interstellar"


def get_main_js():
    js_code = pathlib.Path(
        os.path.join(now_dir, "tabs", "realtime", "main.js")
    ).read_text()
    if client_mode and auth_enabled():
        from assets.auth import API_TOKEN

        js_code = f"window.__APILIO_API_TOKEN = '{API_TOKEN}';\n" + js_code
    return js_code


# Define Gradio interface
with gr.Blocks(
    title="Applio",
    **(
        {
            "theme": my_applio,
            "css": "footer{display:none !important}",
            "js": (f"() => {{\n{get_main_js()}\n}}" if client_mode else None),
        }
        if not GRADIO_6
        else {}
    ),
) as Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "A simple, high-quality voice conversion tool focused on ease of use and performance."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/wY7gmqTyEV) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Realtime")):
        realtime_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()

    with gr.Tab(i18n("TensorBoard")):
        tensorboard_tab()

    gr.Markdown("""
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Applio, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Applio disclaims liability and reserves the right to amend these terms.
    </div>
    """)


def launch_gradio(server_name: str, server_port: int) -> None:
    app, _, _ = Applio.launch(
        favicon_path="assets/ICON.ico",
        share=_has_share,
        inbrowser=_has_open,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=client_mode,
        auth=_auth,
        **(
            {
                "theme": my_applio,
                "css": "footer{display:none !important}",
                "js": (get_main_js() if client_mode else None),
            }
            if GRADIO_6
            else {}
        ),
    )

    # Mount TensorBoard proxy so it's accessible from any origin
    from rvc.lib.tools.launch_tensorboard import get_tb_url
    import httpx
    from fastapi import Request, Response

    @app.api_route(
        "/tensorboard/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    @app.api_route(
        "/tensorboard",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def tb_proxy(request: Request, path: str = ""):
        tb_url = get_tb_url()
        if not tb_url:
            return Response("TensorBoard not started", status_code=503)
        url = f"{tb_url.rstrip('/')}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in ["host"]
                },
                content=await request.body(),
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    if client_mode:
        import time
        from rvc.realtime.client import app as fastapi_app

        app.mount("/api", fastapi_app)

        while True:
            time.sleep(5)

    from rvc.infer.infer_api import app as infer_api_app

    app.mount("/api/infer", infer_api_app)


if __name__ == "__main__":
    port = _args.port
    server = _args.server_name

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
