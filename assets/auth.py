import hmac
import json
import os
import secrets

now_dir = os.getcwd()

API_TOKEN = None


def load_auth_config():
    config_path = os.path.join(now_dir, "assets", "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        auth = config.get("auth", {})
        return {
            "enabled": bool(auth.get("enabled", False)),
            "username": auth.get("username", ""),
            "password": auth.get("password", ""),
        }
    except Exception:
        return {"enabled": False, "username": "", "password": ""}


def auth_enabled():
    cfg = load_auth_config()
    return cfg["enabled"] and bool(cfg["username"]) and bool(cfg["password"])


def check_credentials(username, password):
    cfg = load_auth_config()
    return hmac.compare_digest(username or "", cfg["username"]) and hmac.compare_digest(
        password or "", cfg["password"]
    )


def set_api_token(token):
    global API_TOKEN
    API_TOKEN = token


def check_api_token(token):
    if not auth_enabled():
        return True
    if API_TOKEN is None or not token:
        return False
    return hmac.compare_digest(token, API_TOKEN)


def generate_api_token():
    return secrets.token_urlsafe(32)
