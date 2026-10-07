#!/usr/bin/env python3
"""Trio Chat: a shared chat for you + Claude + Gemini (agy).

Python 3.9+ standard library only. Run:  python trio_chat.py
Opens http://127.0.0.1:8765 . Listens on localhost only.

Claude is called as `claude -p` (history goes through stdin),
Gemini as `agy -p` (history goes as an argument). Both use YOUR subscriptions.

Chat settings (the "Settings" button):
  - shared instruction: what we are working on (both AIs see it);
  - extra instruction separately for Claude and for Gemini;
  - model for each AI (empty = default model).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", int(os.environ.get("TRIO_PORT", "8765"))
HERE = Path(__file__).resolve().parent
DATA = HERE / "chats.json"
DEFAULTS_FILE = HERE / "defaults.json"
CWD = HERE / "sandbox_cwd"  # empty folder so the agents do not wander around the project
USER = "Anton"
TIMEOUT = 330  # seconds per reply
MAX_INSTR = 4000  # max characters in one instruction

# bin can be overridden by environment variables (for tests or custom paths);
# limit is the max number of history characters passed to the model.
MODELS = {
    "claude": {"name": "Claude", "maker": "Anthropic",
               "bin": os.environ.get("TRIO_CLAUDE_BIN", "claude"), "limit": 60000},
    "gemini": {"name": "Gemini", "maker": "Google",
               "bin": os.environ.get("TRIO_AGY_BIN", "agy"), "limit": 12000},
}
# Claude model catalogue for the dropdown. Aliases (sonnet/opus/haiku) are accepted by
# `claude --model`; full names are API identifiers. Which of them your plan offers
# is unknown to the app: if a name is rejected, the chat shows claude's error.
CLAUDE_MODELS = [
    {"value": "sonnet", "label": "Sonnet", "note": "latest version"},
    {"value": "opus", "label": "Opus", "note": "latest version"},
    {"value": "haiku", "label": "Haiku", "note": "latest version"},
    {"value": "claude-sonnet-5-5", "label": "Sonnet 5.5", "note": "exact version"},
    {"value": "claude-opus-5-5", "label": "Opus 5.5", "note": "exact version"},
    {"value": "claude-fable-5-1", "label": "Fable 5.1", "note": "exact version"},
    {"value": "claude-haiku-4-5-20251001", "label": "Haiku 4.5", "note": "exact version"},
]
RAW_MODELS_FILE = HERE / "agy_models_raw.txt"
# The model name goes into the command line, so only safe characters are allowed
# and a leading "-" is forbidden so it is not mistaken for a flag.
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()\-\[\]:/]{0,79}$")

LOCK = threading.Lock()


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            shutil.copy(path, path.with_suffix(".broken.json"))
    return default


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), "utf-8")
    os.replace(tmp, path)


DB = load_json(DATA, [])
EMPTY_SETTINGS = {"shared": "", "claude": "", "gemini": "", "claude_model": "", "gemini_model": ""}


def save_db():
    save_json(DATA, DB)


def clean_settings(src):
    """Returns validated settings; raises ValueError on an unsafe model name."""
    out = dict(EMPTY_SETTINGS)
    src = src if isinstance(src, dict) else {}
    for k in ("shared", "claude", "gemini"):
        out[k] = str(src.get(k) or "").strip()[:MAX_INSTR]
    for k in ("claude_model", "gemini_model"):
        v = str(src.get(k) or "").strip()
        if v and not MODEL_RE.match(v):
            raise ValueError(f"Invalid model name: {v!r}")
        out[k] = v
    return out


def get_defaults():
    try:
        return clean_settings(load_json(DEFAULTS_FILE, {}))
    except ValueError:
        return dict(EMPTY_SETTINGS)


def cfg(chat):
    """Chat settings, also for old chats that do not have them yet."""
    try:
        return clean_settings(chat.get("settings"))
    except ValueError:
        return dict(EMPTY_SETTINGS)


def find(chat_id):
    return next((c for c in DB if c["id"] == chat_id), None)


# ---------- prompt and model calls ----------

def transcript(chat, limit):
    names = {"user": USER, "claude": "Claude", "gemini": "Gemini"}
    lines = [f"[{names[m['who']]}]: {m['text']}" for m in chat["messages"]]
    out, total = [], 0
    for line in reversed(lines):  # take the newest messages while they fit
        if total + len(line) > limit and out:
            break
        out.append(line[-limit:])
        total += len(line)
    return "\n\n".join(reversed(out))


def build_prompt(chat, who):
    me = MODELS[who]["name"]
    other_key = "gemini" if who == "claude" else "claude"
    other = MODELS[other_key]["name"]
    s = cfg(chat)

    roster = [f"- {USER}: a human, the user. They lead the conversation and decide who replies.",
              f"- Claude: the AI Claude (made by Anthropic)."
              ,
              f"- Gemini: the AI Gemini (made by Google)."]
    if chat["kind"] == "shared":
        scene = (f"This is a shared chat of three participants: {USER}, Claude and Gemini. "
                 f"You are {me}. The other AI is {other}; its messages in the history carry its name, "
                 f"they are not your words.")
        who_block = "Participants:\n" + "\n".join(roster)
    else:
        scene = (f"This is a private conversation between the user {USER} (a human) and you. You are {me}. "
                 f"There are no other AIs in this conversation (if the history contains messages from {other}, "
                 f"they were copied from an earlier shared chat).")
        who_block = f"Participants:\n- {USER}: a human, the user.\n- {me}: the AI {me} (made by {MODELS[who]['maker']}), that is you."

    parts = [scene, who_block]
    if s["shared"]:
        parts.append("Shared instruction (what we are working on):\n" + s["shared"])
    if s[who]:
        parts.append(f"Extra instruction for you only ({me}):\n" + s[who])
    parts.append("Reply rules: answer in English unless asked for another language. "
                 "Be brief and to the point. Write only as yourself, without a name prefix. "
                 "Do not use tools and do not read files: reply with text only.")
    head = "\n\n".join(parts)
    hist = transcript(chat, MODELS[who]["limit"])
    return (f"{head}\n\n=== Chat history ===\n{hist}\n=== End of history ===\n\n"
            f"Write your next message ({me}).")


def build_cmd(who, exe, model, prompt):
    """All flags come BEFORE -p and the prompt is last: this is safe whether -p takes a value
    or is a boolean flag."""
    if who == "claude":
        cmd = [exe]
        if model:
            cmd += ["--model", model]
        cmd += ["-p", "Follow the instructions in the input and output only the reply text."]
        return cmd, prompt
    cmd = [exe]
    if model:
        cmd += ["--model", model]
    cmd += ["--mode", "plan", "--print-timeout", "300s", "-p", prompt]
    return cmd, ""


def run_model(who, prompt, model=""):
    m = MODELS[who]
    exe = shutil.which(m["bin"]) or m["bin"]
    CWD.mkdir(exist_ok=True)
    cmd, stdin = build_cmd(who, exe, model, prompt)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=TIMEOUT, cwd=CWD, creationflags=flags)
    except FileNotFoundError:
        return None, f"Program \"{m['bin']}\" not found. Check that it is in PATH."
    except subprocess.TimeoutExpired:
        return None, f"{m['name']} did not reply within {TIMEOUT} s."
    text = (r.stdout or "").strip()
    print(f"[{m['name']}{' / ' + model if model else ''}] code {r.returncode}, "
          f"{time.time() - t0:.1f} s, {len(text)} chars")
    if r.returncode != 0 or not text:
        err = ((r.stderr or "").strip() or text)[-400:]  # claude prints auth errors to stdout
        hint = (" Empty output from agy -p is a known bug, try `agy update`."
                if who == "gemini" and not text else "")
        if model and err:
            hint += " If the model name is the cause, check it in the list (agy models / claude --help)."
        return None, f"{m['name']}: code {r.returncode}, empty reply.{hint} {err}".strip()
    return text, None


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_models_cache = {"t": 0, "list": [], "err": None}
ID_RE = re.compile(r"[a-z0-9][a-z0-9._]*(?:-[a-z0-9._]+)+")
DEFAULT_MARK = re.compile(r"\s*(\(default\)|\*|\u2713|\u2714)\s*$", re.I)


def parse_models(text):
    """Parses the output of `agy models`. The format is not known in advance, so parsing is flexible:
    one line per model; if a line has several columns (separated by a tab or 2+ spaces),
    the value is the column that looks like an id (gemini-3-pro-high) and the label is the column with spaces."""
    out, seen = [], set()
    for raw in ANSI.sub("", text or "").splitlines():
        line = re.sub(r"[\u2502\u2503\u2506\u2551|]", "  ", raw).strip().lstrip("-*\u2022>").strip()
        if not line or set(line) <= set("-=\u2500\u2501+ ") or line.endswith(":") or line.isupper():
            continue
        is_default = bool(DEFAULT_MARK.search(line))
        line = DEFAULT_MARK.sub("", line)
        cols = [c.strip() for c in re.split(r"\t|\s{2,}", line) if c.strip()]
        if not cols:
            continue
        id_col = next((c for c in cols if ID_RE.fullmatch(c)), None)
        name_col = next((c for c in cols if " " in c), None)
        if id_col:
            value, label = id_col, (name_col or id_col)
        else:
            value, label = cols[0], (name_col or cols[0])
        if not MODEL_RE.match(value) or value in seen:
            continue
        seen.add(value)
        out.append({"value": value, "label": label, "note": "agy default" if is_default else (value if label != value else "")})
        if len(out) >= 60:
            break
    return out


def list_gemini_models(force=False):
    """Model list from `agy models` (cached for 60 s). Raw output is saved to agy_models_raw.txt."""
    if not force and _models_cache["t"] and time.time() - _models_cache["t"] < 60:
        return _models_cache["list"], _models_cache["err"]
    m = MODELS["gemini"]
    exe = shutil.which(m["bin"]) or m["bin"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    lst, err = [], None
    try:
        CWD.mkdir(exist_ok=True)
        r = subprocess.run([exe, "models"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=25, cwd=CWD, creationflags=flags)
        try:
            RAW_MODELS_FILE.write_text((r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""), "utf-8")
        except OSError:
            pass
        lst = parse_models(r.stdout or "")
        if not lst:
            err = "agy models returned no model list."
    except FileNotFoundError:
        err = f"Program \"{m['bin']}\" not found."
    except subprocess.TimeoutExpired:
        err = "agy models did not reply within 25 s."
    _models_cache.update(t=time.time(), list=lst, err=err)
    return lst, err


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _ok(self):
        host = self.headers.get("Host", "").split(":")[0]
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1", "localhost"):
            return False
        return not origin or origin in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        if not self._ok():
            return self._send(403, "{}")
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if self.path == "/api/state":
            with LOCK:
                out = [dict(c, settings=cfg(c)) for c in DB]
            return self._json({"chats": out, "defaults": get_defaults()})
        if self.path == "/api/models/claude":
            return self._json({"models": CLAUDE_MODELS, "error": None})
        if self.path.startswith("/api/models/gemini"):
            lst, err = list_gemini_models(force="refresh=1" in self.path)
            return self._json({"models": lst, "error": err})
        self._send(404, "{}")

    def do_POST(self):
        if not self._ok() or "application/json" not in self.headers.get("Content-Type", ""):
            return self._send(403, "{}")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except Exception:
            return self._json({"error": "bad json"}, 400)
        route = self.path
        if route == "/api/chat":
            kind = body.get("kind")
            model = body.get("model")
            if kind not in ("shared", "solo") or (kind == "solo" and model not in MODELS):
                return self._json({"error": "bad chat"}, 400)
            chat = {"id": uuid.uuid4().hex[:8], "title": (body.get("title") or "Untitled")[:80],
                    "kind": kind, "model": model if kind == "solo" else None,
                    "settings": get_defaults(), "messages": []}
            with LOCK:
                DB.insert(0, chat)
                save_db()
            return self._json(chat)
        if route == "/api/message":
            with LOCK:
                chat = find(body.get("chat_id"))
                text = (body.get("text") or "").strip()
                if not chat or not text:
                    return self._json({"error": "bad message"}, 400)
                chat["messages"].append({"who": "user", "text": text, "ts": time.time()})
                save_db()
            return self._json({"ok": True})
        if route == "/api/reply":
            who = body.get("who")
            with LOCK:
                chat = find(body.get("chat_id"))
                if not chat or who not in MODELS or (chat["kind"] == "solo" and chat["model"] != who):
                    return self._json({"error": "bad reply request"}, 400)
                prompt = build_prompt(chat, who)
                model = cfg(chat)[who + "_model"]
            text, err = run_model(who, prompt, model)  # outside the lock: this is slow
            if err:
                return self._json({"error": err})
            with LOCK:
                chat["messages"].append({"who": who, "text": text, "ts": time.time()})
                save_db()
            return self._json({"ok": True})
        if route == "/api/settings":
            try:
                new = clean_settings(body.get("settings"))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            with LOCK:
                chat = find(body.get("chat_id"))
                if not chat:
                    return self._json({"error": "bad chat"}, 400)
                chat["settings"] = new
                save_db()
            if body.get("as_default"):
                save_json(DEFAULTS_FILE, new)
            return self._json({"ok": True})
        if route == "/api/fork":
            model = body.get("model")
            with LOCK:
                src = find(body.get("chat_id"))
                if not src or model not in MODELS:
                    return self._json({"error": "bad fork"}, 400)
                chat = {"id": uuid.uuid4().hex[:8], "title": f"{src['title']} (copy → {MODELS[model]['name']})",
                        "kind": "solo", "model": model, "settings": cfg(src),
                        "messages": json.loads(json.dumps(src["messages"]))}
                DB.insert(0, chat)
                save_db()
            return self._json(chat)
        if route == "/api/delete":
            with LOCK:
                DB[:] = [c for c in DB if c["id"] != body.get("chat_id")]
                save_db()
            return self._json({"ok": True})
        self._send(404, "{}")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Trio Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400..600&family=Onest:wght@400..650&display=swap" rel="stylesheet">
<style>
@property --c1{syntax:'<color>';inherits:true;initial-value:#27434a}
@property --c2{syntax:'<color>';inherits:true;initial-value:#27434a}
:root{
  --abyss:#0a1417;--rail:#081114;--surface:#0f1d21;--raise:#152a30;--line:#1c3036;--line2:#2a464d;
  --text:#e5eeeb;--muted:#8aa39f;--faint:#5c7773;--danger:#ff9087;
  --claude:240 179 94;--gemini:147 165 255;--anton:127 214 194;
  --ui:'Onest',system-ui,'Segoe UI',sans-serif;--read:'Literata',Georgia,'Times New Roman',serif;
  --ease:cubic-bezier(.2,.8,.2,1);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--abyss);color:var(--text);font:15px/1.5 var(--ui);-webkit-font-smoothing:antialiased;overflow:hidden}
button,input,textarea{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid rgb(var(--anton));outline-offset:2px}
::selection{background:rgb(var(--anton)/.28)}
*{scrollbar-width:thin;scrollbar-color:var(--line2) transparent}

.app{display:grid;grid-template-columns:292px 1fr;height:100dvh}

/* ---------- left rail ---------- */
.rail{background:var(--rail);border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
.rail-head{padding:18px 16px 12px;display:flex;flex-direction:column;gap:14px}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:16px;letter-spacing:.01em}
.trio{display:inline-flex}
.trio i{width:16px;height:16px;border-radius:50%;border:2px solid var(--rail);margin-left:-5px}
.trio i:first-child{margin-left:0}
.trio .c{background:rgb(var(--claude))}.trio .g{background:rgb(var(--gemini))}.trio .u{background:rgb(var(--anton))}
.btn{display:inline-flex;align-items:center;justify-content:center;white-space:nowrap}
nav#list{flex:1;overflow:auto;padding:4px 10px 14px;display:flex;flex-direction:column;gap:2px}
.item{position:relative;text-align:left;border:0;background:transparent;padding:10px 12px 10px 14px;border-radius:12px;display:grid;grid-template-columns:1fr auto;gap:2px 10px;transition:background .2s}
.item:hover{background:var(--surface)}
.item.on{background:var(--raise)}
.item.on::before{content:'';position:absolute;left:0;top:11px;bottom:11px;width:3px;border-radius:3px;background:var(--text)}
.item .t{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.item .n{color:var(--faint);font-size:13px;align-self:center}
.item .k{grid-column:1/-1;display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
.dots{display:inline-flex}
.dots i{width:9px;height:9px;border-radius:50%;margin-left:-3px;border:1.5px solid var(--rail)}
.dots i:first-child{margin-left:0}
.item.on .dots i{border-color:var(--raise)}
.dots .c{background:rgb(var(--claude))}.dots .g{background:rgb(var(--gemini))}.dots .u{background:rgb(var(--anton))}
.rail-empty{color:var(--faint);padding:10px 12px;font-size:14px}

/* ---------- main stage ---------- */
.stage{display:flex;flex-direction:column;min-width:0;min-height:0;position:relative}
.top{display:flex;align-items:center;gap:12px;padding:14px 22px;border-bottom:1px solid var(--line);min-height:68px}
.iconbtn{width:38px;height:38px;border-radius:11px;border:1px solid transparent;background:transparent;display:inline-grid;place-items:center;color:var(--muted);transition:background .2s,color .2s}
.iconbtn:hover{background:var(--surface);color:var(--text)}
.top[hidden]{display:none}
#burger{display:none}
.heading{flex:1;min-width:0}
.heading h1{margin:0;font-size:18px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.who-line{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13.5px;margin-top:2px}
.top-actions{display:flex;gap:6px;position:relative}
.top-actions .btn{padding:8px 12px}
#menu{position:absolute;right:0;top:calc(100% + 8px);min-width:250px;background:var(--surface);border:1px solid var(--line2);border-radius:14px;padding:6px;z-index:30;box-shadow:0 18px 50px -12px rgb(0 0 0/.7);opacity:0;transform:translateY(-6px) scale(.98);transform-origin:top right;pointer-events:none;transition:opacity .18s,transform .18s var(--ease)}
#menu.open{opacity:1;transform:none;pointer-events:auto}
#menu button{display:flex;width:100%;text-align:left;gap:10px;align-items:center;padding:9px 11px;border:0;background:transparent;border-radius:9px}
#menu button:hover{background:var(--raise)}
#menu .sep{height:1px;background:var(--line);margin:5px 4px}
#menu .bad{color:var(--danger)}

#empty{margin:auto;max-width:430px;text-align:center;padding:24px;display:flex;flex-direction:column;align-items:center;gap:14px}
#empty .orbs{display:flex;margin-bottom:6px}
.orb{width:54px;height:54px;border-radius:50%;display:grid;place-items:center;font-weight:600;font-size:20px;border:3px solid var(--abyss);margin-left:-12px}
.orb:first-child{margin-left:0}
.orb.c{background:rgb(var(--claude));color:#2b1c05}.orb.g{background:rgb(var(--gemini));color:#10163f}.orb.u{background:rgb(var(--anton));color:#06241d}
#empty h2{margin:0;font-size:22px;font-weight:600}
#empty p{margin:0;color:var(--muted);line-height:1.6}
.load-in .orb{animation:orb-in .7s var(--ease) both}
.load-in .orb:nth-child(2){animation-delay:.12s}.load-in .orb:nth-child(3){animation-delay:.24s}
@keyframes orb-in{from{opacity:0;transform:translateX(-26px) scale(.8)}to{opacity:1;transform:none}}

#convo{display:none;flex-direction:column;flex:1;min-height:0}
#thread{flex:1;overflow:auto;padding:28px 22px 12px;scroll-behavior:smooth}
.lane{max-width:780px;margin:0 auto;display:flex;flex-direction:column;gap:26px}
.thread-empty{color:var(--faint);text-align:center;padding:50px 10px;line-height:1.7}
.m{display:grid;grid-template-columns:34px minmax(0,1fr);gap:14px}
.m.in{animation:msg-in .45s var(--ease) both}
@keyframes msg-in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.av{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-weight:600;font-size:14px}
.m.claude .av{background:rgb(var(--claude)/.16);color:rgb(var(--claude))}
.m.gemini .av{background:rgb(var(--gemini)/.16);color:rgb(var(--gemini))}
.m .head{display:flex;align-items:baseline;gap:10px;margin-bottom:3px}
.m .name{font-weight:600}
.m.claude .name{color:rgb(var(--claude))}.m.gemini .name{color:rgb(var(--gemini))}
.m .time{color:var(--faint);font-size:12.5px}
.body{font-family:var(--read);font-size:16.5px;line-height:1.7;overflow-wrap:anywhere;white-space:pre-wrap;max-width:70ch}
.body code{font-family:ui-monospace,Consolas,monospace;font-size:.86em;background:var(--surface);border:1px solid var(--line);padding:1px 6px;border-radius:6px}
.body strong{font-weight:600}
.codewrap{position:relative;margin:10px 0;white-space:normal}
.codewrap pre{margin:0;padding:14px 16px;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:auto;font:13.5px/1.6 ui-monospace,Consolas,monospace;white-space:pre;color:#d6e6e2;max-width:100%}
.codewrap code{background:none;border:0;padding:0;font-size:inherit;border-radius:0}
.codewrap button{position:absolute;top:8px;right:8px;padding:3px 9px;font-size:12px;border-radius:7px;border:1px solid var(--line2);background:var(--raise);color:var(--muted);opacity:0;transition:opacity .15s}
.codewrap:hover button,.codewrap button:focus-visible{opacity:1}
.m.user{display:flex;justify-content:flex-end}
.m.user .ubox{max-width:min(82%,60ch)}
.m.user .bubble{background:var(--raise);border:1px solid var(--line);padding:11px 16px;border-radius:18px 18px 5px 18px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}
.m.user .time{display:block;text-align:right;margin-top:4px}
.thinking .dotrow{display:flex;gap:6px;padding:10px 0}
.thinking .dotrow i{width:7px;height:7px;border-radius:50%;background:currentColor;animation:pulse 1.2s ease-in-out infinite}
.thinking .dotrow i:nth-child(2){animation-delay:.16s}.thinking .dotrow i:nth-child(3){animation-delay:.32s}
.thinking.claude .dotrow{color:rgb(var(--claude))}.thinking.gemini .dotrow{color:rgb(var(--gemini))}
@keyframes pulse{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1.15)}}
.errrow{display:flex;gap:14px;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:14px 16px;border:1px solid rgb(255 144 135/.4);background:rgb(255 144 135/.08);border-radius:14px;line-height:1.55}
.errrow span{flex:1;min-width:220px;overflow-wrap:anywhere}

/* ---------- input ---------- */
.dock{padding:6px 22px 20px}
.composer{--c1:var(--line2);--c2:var(--line2);max-width:780px;margin:0 auto;padding:1px;border-radius:20px;background:linear-gradient(100deg,var(--c1),var(--c2));transition:--c1 .4s ease,--c2 .4s ease,box-shadow .4s ease;box-shadow:0 10px 40px -22px var(--c1)}
.composer:focus-within{box-shadow:0 12px 56px -16px var(--c1)}
.composer[data-target=claude]{--c1:rgb(var(--claude));--c2:rgb(var(--claude))}
.composer[data-target=gemini]{--c1:rgb(var(--gemini));--c2:rgb(var(--gemini))}
.composer[data-target=both]{--c1:rgb(var(--claude));--c2:rgb(var(--gemini))}
.composer-in{background:var(--surface);border-radius:19px;padding:14px 16px 12px}
#ta{display:block;width:100%;border:0;background:transparent;resize:none;outline:0;min-height:30px;max-height:220px;line-height:1.55;font-size:15.5px}
#ta::placeholder{color:var(--faint)}
.bar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:10px;flex-wrap:wrap}
.resp{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.resp>span{color:var(--muted);font-size:13.5px}
.seg{display:inline-flex;gap:2px;padding:3px;background:var(--abyss);border-radius:999px;border:1px solid var(--line)}
.seg label{position:relative;cursor:pointer}
.seg input{position:absolute;opacity:0;inset:0;cursor:pointer}
.seg .chip{white-space:nowrap;display:inline-flex;align-items:center;gap:7px;padding:6px 13px;border-radius:999px;color:var(--muted);transition:background .22s,color .22s;font-size:14px}
.seg .chip::before{content:'';width:8px;height:8px;border-radius:50%;background:var(--faint);transition:transform .22s var(--ease)}
.seg .k-claude::before{background:rgb(var(--claude))}.seg .k-gemini::before{background:rgb(var(--gemini))}
.seg .k-both::before{background:linear-gradient(90deg,rgb(var(--claude)) 50%,rgb(var(--gemini)) 50%)}
.seg label:hover .chip{color:var(--text)}
.seg{position:relative}
.seg .pill{position:absolute;top:3px;bottom:3px;left:0;width:0;border-radius:999px;background:var(--raise);box-shadow:inset 0 0 0 1px var(--line2);pointer-events:none;transition:transform .42s var(--ease),width .42s var(--ease)}
.seg .pill.nomove{transition:none}
.seg label{z-index:1}
.seg input:checked+.chip{color:var(--text)}
.seg input:checked+.chip::before{transform:scale(1.35)}
.seg label:active .chip{transform:scale(.95)}
.seg .chip{transition:color .22s,transform .15s}
.seg input:focus-visible+.chip{outline:2px solid rgb(var(--anton));outline-offset:1px}
.send{display:inline-flex;align-items:center;white-space:nowrap}
.send:disabled{opacity:.4;cursor:default}
.hint{text-align:center;color:var(--faint);font-size:12.5px;margin-top:10px}

/* ---------- settings ---------- */
.veil{position:fixed;inset:0;background:rgb(3 8 10/.62);opacity:0;pointer-events:none;transition:opacity .3s;z-index:40}
.veil.on{opacity:1;pointer-events:auto}
.drawer{position:fixed;top:0;right:0;bottom:0;width:min(470px,100vw);background:var(--rail);border-left:1px solid var(--line);z-index:50;display:flex;flex-direction:column;transform:translateX(104%);transition:transform .4s var(--ease);box-shadow:-30px 0 80px -30px rgb(0 0 0/.8)}
.drawer.open{transform:none}
.dr-head{display:flex;align-items:center;justify-content:space-between;padding:18px 22px;border-bottom:1px solid var(--line)}
.dr-head h2{margin:0;font-size:18px;font-weight:600}
.dr-body{flex:1;overflow:auto;padding:20px 22px;display:flex;flex-direction:column;gap:22px}
.field{display:flex;flex-direction:column;gap:7px}
.field>label,.field>.lab{font-weight:500}
.field small{color:var(--muted);line-height:1.5}
.field textarea,.field input{width:100%;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 12px;outline:0;transition:border-color .2s,box-shadow .2s;line-height:1.5}
.field textarea{min-height:92px;resize:vertical}
.field textarea:focus,.field input:focus{border-color:var(--line2);box-shadow:0 0 0 3px rgb(var(--anton)/.14)}
.who-block{border-left:3px solid var(--faint);padding-left:16px;display:flex;flex-direction:column;gap:14px}
.who-block.claude{border-color:rgb(var(--claude))}.who-block.gemini{border-color:rgb(var(--gemini))}
.who-block h3{margin:0;font-size:15px;font-weight:600}
.who-block.claude h3{color:rgb(var(--claude))}.who-block.gemini h3{color:rgb(var(--gemini))}
.dr-foot{padding:14px 22px 18px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:10px}
.dr-foot .row{display:flex;gap:8px;flex-wrap:wrap}
.dr-foot .btn{flex:1}
.note{color:var(--muted);font-size:13.5px;line-height:1.5}

/* ---------- dialogs and toasts ---------- */
dialog{border:1px solid var(--line2);background:var(--surface);color:var(--text);border-radius:20px;padding:0;width:min(460px,calc(100vw - 32px));box-shadow:0 30px 90px -20px rgb(0 0 0/.85)}
dialog::backdrop{background:rgb(3 8 10/.66)}
dialog[open]{animation:pop .28s var(--ease)}
@keyframes pop{from{opacity:0;transform:translateY(12px) scale(.97)}to{opacity:1;transform:none}}
.dlg{padding:24px;display:flex;flex-direction:column;gap:18px}
.dlg h2{margin:0;font-size:19px;font-weight:600}
.dlg .actions{display:flex;gap:8px;flex-direction:row-reverse;justify-content:flex-start}
.kinds{display:flex;flex-direction:column;gap:8px}
.kind{position:relative;display:flex;gap:12px;align-items:center;padding:12px 14px;border:1px solid var(--line);border-radius:14px;cursor:pointer;transition:border-color .2s,background .2s}
.kind:hover{background:var(--raise)}
.kind input{position:absolute;opacity:0;inset:0;cursor:pointer}
.kind:has(input:checked){border-color:var(--text);background:var(--raise)}
.kind:has(input:focus-visible){outline:2px solid rgb(var(--anton));outline-offset:2px}
.kind b{display:block;font-weight:500}.kind small{color:var(--muted)}
#toast{position:fixed;left:50%;bottom:26px;transform:translate(-50%,20px);background:var(--text);color:var(--abyss);padding:10px 18px;border-radius:999px;font-weight:500;opacity:0;pointer-events:none;transition:opacity .25s,transform .35s var(--ease);z-index:80}
#toast.on{opacity:1;transform:translate(-50%,0)}

/* ---------- model picker ---------- */
.picker{position:relative}
.pk-btn{width:100%;display:flex;align-items:center;justify-content:space-between;gap:10px;text-align:left;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 12px;transition:border-color .2s,box-shadow .2s,background .2s}
.pk-btn:hover{border-color:var(--line2)}
.pk-btn[aria-expanded=true]{border-color:var(--line2);box-shadow:0 0 0 3px rgb(var(--anton)/.14)}
.pk-val{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pk-val small{color:var(--muted);margin-left:8px}
.pk-chev{flex:none;color:var(--muted);transition:transform .3s var(--ease)}
.pk-btn[aria-expanded=true] .pk-chev{transform:rotate(180deg)}
.pk-panel{position:fixed;z-index:90;background:var(--surface);border:1px solid var(--line2);border-radius:14px;padding:6px;max-height:min(300px,45vh);overflow:auto;box-shadow:0 20px 60px -12px rgb(0 0 0/.8);opacity:0;transform:translateY(-6px) scale(.98);transform-origin:top center;pointer-events:none;transition:opacity .18s,transform .22s var(--ease);visibility:hidden}
.pk-panel.up{transform-origin:bottom center;transform:translateY(6px) scale(.98)}
.pk-panel.open{opacity:1;transform:none;pointer-events:auto;visibility:visible}
.pk-opt{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:9px;cursor:pointer;transition:background .15s;line-height:1.35}
.pk-opt.hot{background:var(--raise)}
.pk-opt .tx{flex:1;min-width:0}
.pk-opt .tx small{display:block;color:var(--muted);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pk-opt .ck{width:16px;height:16px;flex:none;color:rgb(var(--anton));opacity:0;transform:scale(.4);transition:opacity .2s,transform .3s var(--ease)}
.pk-opt[aria-selected=true] .ck{opacity:1;transform:none}
.pk-sep{height:1px;background:var(--line);margin:5px 4px}
.pk-custom{display:grid;grid-template-rows:0fr;transition:grid-template-rows .35s var(--ease),margin .35s var(--ease);margin-top:0}
.pk-custom>div{overflow:hidden}
.pk-custom.on{grid-template-rows:1fr;margin-top:8px}
.pk-custom input{width:100%;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 12px;outline:0}
.pk-custom input:focus{border-color:var(--line2);box-shadow:0 0 0 3px rgb(var(--anton)/.14)}

/* ---------- smoothness and animations ---------- */
.btn,.iconbtn,.send{position:relative;overflow:hidden}
.iconbtn{transition:background .2s,color .2s,transform .15s}
.iconbtn:active{transform:scale(.9)}
.item{transition:background .2s,transform .15s}
.item:active{transform:scale(.985)}
.kind{transition:border-color .2s,background .2s,transform .15s}
.kind:active{transform:scale(.99)}
.pk-btn:active{transform:scale(.99)}
.ripple{position:absolute;border-radius:50%;background:rgb(255 96 78);opacity:.3;z-index:0;transform:scale(0);pointer-events:none;animation:ripple .65s ease-out forwards}
@keyframes ripple{to{transform:scale(1);opacity:0}}
.send svg{transition:transform .2s}
.send:hover:not(:disabled) svg{transform:translateY(-2px)}
.send.fly svg{animation:fly .55s var(--ease)}
@keyframes fly{0%{transform:translateY(0);opacity:1}45%{transform:translateY(-16px);opacity:0}46%{transform:translateY(16px);opacity:0}100%{transform:translateY(0);opacity:1}}
.item.enter{animation:item-in .5s var(--ease) both;animation-delay:calc(var(--i,0)*45ms)}
@keyframes item-in{from{opacity:0;transform:translateX(-14px)}to{opacity:1;transform:none}}
.item.just::before{animation:bar-in .45s var(--ease) both;transform-origin:center}
@keyframes bar-in{from{transform:scaleY(0)}to{transform:none}}
.lane.swap{animation:lane-in .42s var(--ease) both}
@keyframes lane-in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.heading.swap{animation:head-in .4s var(--ease) both}
@keyframes head-in{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
.m.in .av{animation:av-pop .55s var(--ease) both .06s}
@keyframes av-pop{from{transform:scale(.4);opacity:0}to{transform:none;opacity:1}}
.m.thinking .av{animation:breathe 1.8s ease-in-out infinite}
@keyframes breathe{0%,100%{box-shadow:0 0 0 0 currentColor}50%{box-shadow:0 0 0 7px transparent}}
.m.user.in .bubble{animation:bubble-in .45s var(--ease) both}
@keyframes bubble-in{from{opacity:0;transform:translateY(10px) scale(.96);transform-origin:bottom right}to{opacity:1;transform:none}}
.errrow{animation:msg-in .4s var(--ease) both}
.composer.pulse{animation:cpulse .6s var(--ease)}
@keyframes cpulse{0%{transform:scale(1)}35%{transform:scale(1.012)}100%{transform:scale(1)}}
.drawer.open .dr-body>*{animation:rise-in .55s var(--ease) both}
.drawer.open .dr-body>:nth-child(1){animation-delay:.12s}.drawer.open .dr-body>:nth-child(2){animation-delay:.2s}
.drawer.open .dr-body>:nth-child(3){animation-delay:.28s}.drawer.open .dr-body>:nth-child(4){animation-delay:.36s}
.drawer.open .dr-foot{animation:rise-in .55s var(--ease) both .3s}
dialog[open] .kind{animation:rise-in .45s var(--ease) both}
dialog[open] .kind:nth-child(1){animation-delay:.08s}dialog[open] .kind:nth-child(2){animation-delay:.15s}dialog[open] .kind:nth-child(3){animation-delay:.22s}
@keyframes rise-in{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
#menu button{transition:background .15s,transform .15s}
#menu button:active{transform:scale(.98)}
#menu.open button{animation:rise-in .35s var(--ease) both}
#menu.open button:nth-child(2){animation-delay:.04s}#menu.open button:nth-child(3){animation-delay:.08s}
.orbs .orb{animation:float 6s ease-in-out infinite}
.orbs .orb:nth-child(2){animation-delay:-2s}.orbs .orb:nth-child(3){animation-delay:-4s}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.load-in .orb{animation:orb-in .7s var(--ease) both,float 6s ease-in-out .9s infinite}
.load-in .orb:nth-child(2){animation-delay:.12s,1.1s}.load-in .orb:nth-child(3){animation-delay:.24s,1.3s}
@media (min-width:861px){
  .intro .rail{animation:rail-in .8s var(--ease) both}
  .intro .top{animation:head-in .7s .1s var(--ease) both}
  .intro #thread{animation:rise-in .8s .2s var(--ease) both}
  .intro .dock{animation:rise-in .8s .32s var(--ease) both}
  @keyframes rail-in{from{opacity:0;transform:translateX(-28px)}to{opacity:1;transform:none}}
}

/* ---------- buttons: glass pill with a red glow ---------- */
@property --g{syntax:'<number>+';inherits:true;initial-value:255 59 48}
.btn,.send{
  --g:255 59 48;--mx:28%;--my:50%;
  gap:9px;padding:10px 20px;border:1px solid transparent;border-radius:999px;color:var(--text);font-weight:500;isolation:isolate;
  background:linear-gradient(180deg,#131718,#0a0d0e) padding-box,
    linear-gradient(135deg,rgb(255 255 255/.34),rgb(255 255 255/.05) 36%,rgb(255 255 255/.1) 66%,rgb(255 255 255/.26)) border-box;
  background-origin:border-box;
  box-shadow:inset 0 1px 0 rgb(255 255 255/.07),0 8px 24px -14px rgb(0 0 0/.9);
  transition:transform .18s var(--ease),box-shadow .45s,color .3s,--g .5s ease}
.btn::after,.send::after{content:'';position:absolute;inset:0;z-index:-1;border-radius:inherit;pointer-events:none;
  background:radial-gradient(circle 120px at var(--mx) var(--my),rgb(var(--g)/.62),rgb(var(--g)/.2) 42%,transparent 72%);
  opacity:0;transition:opacity .5s ease}
.btn:hover::after,.btn:focus-visible::after,.send:hover:not(:disabled)::after,.send:focus-visible::after{opacity:1}
.btn:hover,.send:hover:not(:disabled){box-shadow:inset 0 1px 0 rgb(255 255 255/.1),0 10px 34px -12px rgb(var(--g)/.55)}
.btn:active,.send:active:not(:disabled){transform:scale(.965)}
.btn:disabled{opacity:.45;cursor:default}
.btn svg,.send svg{position:relative;z-index:1}

.btn.primary{font-weight:600}
.btn.primary::before{content:'';order:99;flex:none;width:2px;height:17px;border-radius:2px;background:rgb(var(--g));box-shadow:0 0 10px rgb(var(--g)/.9);animation:caret 1.15s steps(1) infinite}
@keyframes caret{50%{opacity:0}}
.btn.primary svg{color:rgb(var(--g));transition:transform .55s var(--ease)}
.btn.primary:hover svg{transform:rotate(90deg)}
.btn.primary::after{opacity:.32}
.btn.primary:not(:hover):not(:focus-visible)::after{animation:ambient 4.5s ease-in-out infinite}
@keyframes ambient{50%{opacity:.52}}
.btn.primary:hover::after,.btn.primary:focus-visible::after{opacity:1}

.btn.danger{color:#ffb8b0;
  background:linear-gradient(180deg,#221010,#150909) padding-box,
    linear-gradient(135deg,rgb(255 120 105/.6),rgb(255 90 75/.1) 40%,rgb(255 120 105/.4)) border-box;
  background-origin:border-box}
.btn.danger::after{opacity:.4}
.btn.danger:hover::after{opacity:1}

.send{font-weight:600}
.send svg{color:rgb(var(--g));filter:drop-shadow(0 0 6px rgb(var(--g)/.75))}
.send::after{opacity:.3}
.send[data-target=claude]{--g:240 179 94}
.send[data-target=gemini]{--g:147 165 255}
.send[data-target=both]{--g:240 179 94}
.send[data-target=both]::after{background:radial-gradient(circle 90px at 22% 50%,rgb(240 179 94/.6),transparent 70%),radial-gradient(circle 90px at 78% 50%,rgb(147 165 255/.6),transparent 70%)}
.send:hover:not(:disabled)::after{opacity:1}
.iconbtn:focus-visible{box-shadow:0 0 0 2px rgb(255 59 48/.5)}

/* ---------- narrow screens ---------- */
@media (max-width:860px){
  .app{grid-template-columns:1fr}
  .rail{position:fixed;inset:0 auto 0 0;width:min(320px,86vw);z-index:45;transform:translateX(-104%);transition:transform .4s var(--ease)}
  .rail.open{transform:none;box-shadow:30px 0 80px -30px rgb(0 0 0/.8)}
  #burger{display:inline-grid}
  .top{padding:12px 14px}
  #thread{padding:20px 14px 8px}.dock{padding:6px 14px 14px}
  .top-actions .btn .lbl{display:none}
  .resp{width:100%}.seg{flex-wrap:wrap;border-radius:18px}.seg .chip{padding:6px 11px}
  .send{margin-left:auto}
  .m{grid-template-columns:28px minmax(0,1fr);gap:11px}.av{width:28px;height:28px;font-size:12.5px}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms!important;animation-iteration-count:1!important;transition-duration:.001ms!important;scroll-behavior:auto!important}
}
</style>
</head>
<body class="intro">
<div class="app" id="app">
  <aside class="rail" id="rail" aria-label="Chat list">
    <div class="rail-head">
      <div class="brand"><span class="trio" aria-hidden="true"><i class="c"></i><i class="g"></i><i class="u"></i></span>Trio Chat</div>
      <button class="btn primary" id="newBtn" type="button">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M8 3v10M3 8h10"/></svg>New chat</button>
    </div>
    <nav id="list" aria-label="Chats"></nav>
  </aside>
  <div class="veil" id="railVeil"></div>

  <main class="stage">
    <header class="top" id="top" hidden>
      <button class="iconbtn" id="burger" type="button" aria-label="Open chat list">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 6h14M3 10h14M3 14h14"/></svg></button>
      <div class="heading"><h1 id="title"></h1><div class="who-line" id="whoLine"></div></div>
      <div class="top-actions">
        <button class="btn" id="bset" type="button">
          <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M3 6h8M15 6h2M3 14h2M9 14h8"/><circle cx="13" cy="6" r="2"/><circle cx="7" cy="14" r="2"/></svg><span class="lbl">Settings</span></button>
        <button class="iconbtn" id="bmore" type="button" aria-label="More actions" aria-haspopup="menu" aria-expanded="false">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><circle cx="4.5" cy="10" r="1.6"/><circle cx="10" cy="10" r="1.6"/><circle cx="15.5" cy="10" r="1.6"/></svg></button>
        <div id="menu" role="menu">
          <button type="button" role="menuitem" id="mfc">Continue privately with Claude</button>
          <button type="button" role="menuitem" id="mfg">Continue privately with Gemini</button>
          <div class="sep"></div>
          <button type="button" role="menuitem" id="mdel" class="bad">Delete chat</button>
        </div>
      </div>
    </header>

    <section id="empty" class="load-in">
      <div class="orbs" aria-hidden="true"><span class="orb c">C</span><span class="orb g">G</span><span class="orb u">A</span></div>
      <h2>Create your first chat</h2>
      <p>In a shared chat Claude and Gemini see the whole conversation, and you decide who replies. A private chat is with just one of them.</p>
      <button class="btn primary" id="newBtn2" type="button">New chat</button>
    </section>

    <section id="convo">
      <div id="thread" aria-live="polite"><div class="lane" id="lane"></div></div>
      <div class="dock">
        <div class="composer" id="composer" data-target="none">
          <div class="composer-in">
            <textarea id="ta" rows="1" placeholder="Write a message" aria-label="Message"></textarea>
            <div class="bar">
              <div class="resp"><span id="respLab">Who replies</span>
                <div class="seg" id="seg" role="radiogroup" aria-labelledby="respLab">
                  <span class="pill nomove" id="pill" aria-hidden="true"></span>
                  <label id="l_none"><input type="radio" name="t" value="none" checked><span class="chip">Nobody</span></label>
                  <label id="l_claude"><input type="radio" name="t" value="claude"><span class="chip k-claude">Claude</span></label>
                  <label id="l_gemini"><input type="radio" name="t" value="gemini"><span class="chip k-gemini">Gemini</span></label>
                  <label id="l_both"><input type="radio" name="t" value="both"><span class="chip k-both">Both in turn</span></label>
                </div>
              </div>
              <button class="send" id="send" type="button" data-target="none" disabled>
                <span id="sendLab">Send</span>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 13V3M3.5 7.5 8 3l4.5 4.5"/></svg>
              </button>
            </div>
          </div>
        </div>
        <div class="hint">Ctrl+Enter sends. With no text, the button just asks the selected participant to reply.</div>
      </div>
    </section>
  </main>

  <div class="veil" id="drawerVeil"></div>
  <aside class="drawer" id="drawer" aria-label="Chat settings" inert>
    <div class="dr-head"><h2>Chat settings</h2>
      <button class="iconbtn" id="dclose" type="button" aria-label="Close">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 4l10 10M14 4 4 14"/></svg></button></div>
    <div class="dr-body">
      <div class="field"><label for="s_shared">What we are working on</label>
        <textarea id="s_shared" placeholder="Example: we are discussing the app architecture, answer briefly and name the risks."></textarea>
        <small>Every AI in the chat sees this instruction.</small></div>
      <div class="who-block claude" id="f_claude"><h3>Claude</h3>
        <div class="field"><label for="s_claude">Extra for Claude only</label>
          <textarea id="s_claude" placeholder="Example: you make the decision and explain the choice."></textarea></div>
        <div class="field"><span class="lab" id="lab_cm">Model</span>
          <div class="picker" id="pk_claude"></div><input type="hidden" id="s_claude_model">
          <small id="cnote">The aliases Sonnet, Opus and Haiku always mean the latest version.</small></div></div>
      <div class="who-block gemini" id="f_gemini"><h3>Gemini</h3>
        <div class="field"><label for="s_gemini">Extra for Gemini only</label>
          <textarea id="s_gemini" placeholder="Example: you look for weak spots and check the conclusions."></textarea></div>
        <div class="field"><span class="lab" id="lab_gm">Model</span>
          <div class="picker" id="pk_gemini"></div><input type="hidden" id="s_gemini_model">
          <small id="gnote">The list comes from agy models.</small></div></div>
      <p class="note">Each AI gets a reminder of who is in the chat: Anton is a human, Claude is the AI Claude, Gemini is the AI Gemini.</p>
    </div>
    <div class="dr-foot">
      <div class="row"><button class="btn primary" id="ssave" type="button">Save</button>
        <button class="btn" id="sdef" type="button">Save for new chats</button></div>
      <button class="btn" id="sref" type="button">Refresh Gemini model list</button>
    </div>
  </aside>
</div>

<dialog id="dlgNew" aria-labelledby="dnT"><form method="dialog" class="dlg" id="fNew">
  <h2 id="dnT">New chat</h2>
  <div class="field"><label for="nTitle">Title</label><input id="nTitle" maxlength="80" placeholder="Example: Architecture review" autocomplete="off"></div>
  <div class="kinds" role="radiogroup" aria-label="Chat type">
    <label class="kind"><input type="radio" name="kind" value="shared" checked><span class="dots"><i class="u"></i><i class="c"></i><i class="g"></i></span><span><b>Shared chat</b><small>Anton, Claude and Gemini in one conversation</small></span></label>
    <label class="kind"><input type="radio" name="kind" value="solo:claude"><span class="dots"><i class="c"></i></span><span><b>Private with Claude</b><small>A conversation with Claude only</small></span></label>
    <label class="kind"><input type="radio" name="kind" value="solo:gemini"><span class="dots"><i class="g"></i></span><span><b>Private with Gemini</b><small>A conversation with Gemini only</small></span></label>
  </div>
  <div class="actions"><button class="btn primary" value="ok" type="submit">Create</button><button class="btn" value="cancel" type="submit" formnovalidate>Cancel</button></div>
</form></dialog>

<dialog id="dlgConfirm" aria-labelledby="dcT"><form method="dialog" class="dlg">
  <h2 id="dcT"></h2><p class="note" id="dcP" style="margin:0"></p>
  <div class="actions"><button class="btn danger" value="ok" type="submit" id="dcOk">Delete</button><button class="btn" value="cancel" type="submit" autofocus>Cancel</button></div>
</form></dialog>

<div id="toast" role="status"></div>

<script>
const NAME={user:'Anton',claude:'Claude',gemini:'Gemini'};
const INIT={claude:'C',gemini:'G'};
const $=s=>document.querySelector(s);
let chats=[],cur=null,working=false,pending=null,errInfo=null,renderedChat=null,seenCount=0,drawerFor=null;
let knownIds=null,lastActive=null,headChat=null;
const pickers={};

async function api(p,b){const r=await fetch(p,{method:b?'POST':'GET',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined});return r.json();}
const chat=()=>chats.find(c=>c.id===cur);
function h(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;}
function fmtTime(ts){try{return new Date(ts*1000).toLocaleTimeString('uk-UA',{hour:'2-digit',minute:'2-digit'});}catch(e){return '';}}
function dotsFor(c){const d=h('span','dots');const add=k=>d.appendChild(h('i',k));
  if(c.kind==='shared'){add('u');add('c');add('g');}else add(c.model==='claude'?'c':'g');return d;}
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('on');clearTimeout(toast.t);toast.t=setTimeout(()=>t.classList.remove('on'),2200);}

/* ---------- simple formatting: ```code```, `code`, **bold** ---------- */
function rich(text,box){
  const parts=text.split('```');
  parts.forEach((seg,i)=>{
    if(i%2===1){
      const nl=seg.indexOf('\n');let code=seg;
      if(nl>-1&&/^[\w+#.-]*$/.test(seg.slice(0,nl).trim()))code=seg.slice(nl+1);
      code=code.replace(/\n$/,'');
      const w=h('div','codewrap'),pre=h('pre'),cd=h('code',null,code);pre.appendChild(cd);
      const b=h('button',null,'Copy');b.type='button';
      b.onclick=async()=>{try{await navigator.clipboard.writeText(code);b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1400);}catch(e){}};
      w.append(pre,b);box.appendChild(w);
    }else{
      if(i>0)seg=seg.replace(/^\n/,'');
      if(i<parts.length-1)seg=seg.replace(/\n$/,'');
      seg.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/).forEach(tok=>{
        if(!tok)return;
        if(tok.length>2&&tok[0]==='`'&&tok.endsWith('`'))box.appendChild(h('code',null,tok.slice(1,-1)));
        else if(tok.length>4&&tok.startsWith('**')&&tok.endsWith('**'))box.appendChild(h('strong',null,tok.slice(2,-2)));
        else box.appendChild(document.createTextNode(tok));
      });
    }
  });
}

/* ---------- rendering ---------- */
function renderList(){
  const list=$('#list');list.innerHTML='';
  if(!chats.length){list.appendChild(h('div','rail-empty','No chats yet.'));knownIds=new Set();return;}
  const firstTime=knownIds===null;if(firstTime)knownIds=new Set();
  const justActive=cur&&cur!==lastActive;lastActive=cur;
  let n=0;
  for(const c of chats){
    const isNew=!knownIds.has(c.id);knownIds.add(c.id);
    const b=h('button','item'+(c.id===cur?' on':'')+(isNew?' enter':'')+(justActive&&c.id===cur?' just':''));b.type='button';
    if(isNew)b.style.setProperty('--i',String(firstTime?Math.min(n++,8):0));
    b.appendChild(h('span','t',c.title));
    b.appendChild(h('span','n',c.messages.length?String(c.messages.length):''));
    const k=h('span','k');k.appendChild(dotsFor(c));
    k.appendChild(h('span',null,c.kind==='shared'?'Shared':'Private with '+NAME[c.model]));
    b.appendChild(k);
    if(c.id===cur)b.setAttribute('aria-current','true');
    b.onclick=()=>{selectChat(c.id);closeRail();};
    list.appendChild(b);
  }
}
function renderHeader(c){
  const hd=document.querySelector('.heading');
  if(headChat!==c.id){headChat=c.id;hd.classList.remove('swap');void hd.offsetWidth;hd.classList.add('swap');}
  $('#title').textContent=c.title;
  const w=$('#whoLine');w.innerHTML='';w.appendChild(dotsFor(c));
  w.appendChild(h('span',null,c.kind==='shared'?'Anton, Claude and Gemini':'Anton and '+NAME[c.model]));
}
function renderThread(){
  const c=chat();if(!c)return;
  const lane=$('#lane');lane.innerHTML='';
  const fresh=renderedChat!==c.id;
  if(fresh){seenCount=c.messages.length;renderedChat=c.id;lane.classList.remove('swap');void lane.offsetWidth;lane.classList.add('swap');}
  if(!c.messages.length&&!(pending&&pending.chat===c.id)){
    lane.appendChild(h('div','thread-empty',c.kind==='shared'
      ?'Write a message and choose who replies: Claude, Gemini or both in turn.'
      :'Write a message and '+NAME[c.model]+' will reply.'));
  }
  c.messages.forEach((m,i)=>{
    const row=h('div','m '+m.who+(i>=seenCount?' in':''));
    if(m.who==='user'){
      const wrap=h('div','ubox');
      wrap.append(h('div','bubble',m.text),h('span','time',fmtTime(m.ts)));
      row.appendChild(wrap);
    }else{
      row.appendChild(h('div','av',INIT[m.who]));
      const col=h('div');const head=h('div','head');head.append(h('span','name',NAME[m.who]),h('span','time',fmtTime(m.ts)));
      const body=h('div','body');rich(m.text,body);col.append(head,body);row.appendChild(col);
    }
    lane.appendChild(row);
  });
  seenCount=c.messages.length;
  if(pending&&pending.chat===c.id){
    const row=h('div','m thinking in '+pending.who);
    row.appendChild(h('div','av',INIT[pending.who]));
    const col=h('div');const head=h('div','head');head.appendChild(h('span','name',NAME[pending.who]));
    const dr=h('div','dotrow');dr.append(h('i'),h('i'),h('i'));dr.setAttribute('aria-label',NAME[pending.who]+' is thinking');
    col.append(head,dr);row.appendChild(col);lane.appendChild(row);
  }
  if(errInfo&&errInfo.chat===c.id){
    const e=h('div','errrow');e.appendChild(h('span',null,errInfo.msg));
    const b=h('button','btn','Try again');b.type='button';
    b.onclick=()=>{const w=errInfo.who,id=errInfo.chat;errInfo=null;runAsk(id,[w]);};
    e.appendChild(b);lane.appendChild(e);
  }
  const th=$('#thread');
  th.style.scrollBehavior=fresh?'auto':'smooth';
  requestAnimationFrame(()=>{th.scrollTop=th.scrollHeight;});
}
function renderComposer(){
  const c=chat();if(!c)return;
  const sh=c.kind==='shared';
  $('#l_claude').style.display=(sh||c.model==='claude')?'':'none';
  $('#l_gemini').style.display=(sh||c.model==='gemini')?'':'none';
  $('#l_both').style.display=sh?'':'none';
  const cur_t=getTarget();
  if((cur_t==='both'&&!sh)||(cur_t==='claude'&&!(sh||c.model==='claude'))||(cur_t==='gemini'&&!(sh||c.model==='gemini')))setTarget('none');
  updateSend();positionPill(false);
}
function renderAll(){
  renderList();
  const c=chat();
  $('#top').hidden=!c;$('#empty').style.display=c?'none':'flex';$('#convo').style.display=c?'flex':'none';
  if(c){renderHeader(c);renderThread();renderComposer();}
}
async function load(){const r=await api('/api/state');chats=r.chats;if(cur&&!chat())cur=null;renderAll();}

function selectChat(id){
  if(cur===id)return;
  cur=id;errInfo=null;
  const c=chat();setTarget(c&&c.kind==='solo'?c.model:'none');
  renderAll();
  if(drawerFor!==null)openDrawerFor(c);
  $('#ta').focus({preventScroll:true});
}

/* ---------- composer ---------- */
function getTarget(){const r=document.querySelector('input[name=t]:checked');return r?r.value:'none';}
function setTarget(v){const el=document.querySelector('input[name=t][value='+v+']');if(el)el.checked=true;
  const comp=$('#composer'),changed=comp.dataset.target!==v;
  comp.dataset.target=v;$('#send').dataset.target=v;updateSend();positionPill(changed);
  if(changed&&v!=='none'){comp.classList.remove('pulse');void comp.offsetWidth;comp.classList.add('pulse');}}
/* sliding indicator of the selected responder */
function positionPill(animate){
  const pill=$('#pill'),seg=$('#seg'),inp=document.querySelector('input[name=t]:checked');if(!inp)return;
  const lab=inp.closest('label');if(!lab||lab.offsetWidth===0)return;
  if(!animate)pill.classList.add('nomove');
  pill.style.width=lab.offsetWidth+'px';pill.style.transform='translate('+lab.offsetLeft+'px,'+(lab.offsetTop-3)+'px)';
  pill.style.height=lab.offsetHeight+'px';
  if(!animate){void pill.offsetWidth;pill.classList.remove('nomove');}
}
function updateSend(){
  const t=getTarget(),text=$('#ta').value.trim();
  const lab=text?'Send':(t==='none'?'Send':t==='both'?'Ask both':'Ask '+NAME[t]);
  $('#sendLab').textContent=lab;
  $('#send').disabled=working||(!text&&t==='none');
}
function autosize(){const ta=$('#ta');ta.style.height='auto';ta.style.height=Math.min(ta.scrollHeight,220)+'px';}
async function askOne(id,who){
  pending={chat:id,who};renderThread();
  let r;try{r=await api('/api/reply',{chat_id:id,who});}catch(e){r={error:'Cannot reach the app. Check that trio_chat.py is running.'};}
  pending=null;
  if(r.error)errInfo={chat:id,who,msg:r.error};
  await load();return !r.error;
}
async function runAsk(id,order){
  working=true;updateSend();errInfo=null;
  for(const w of order){if(!(await askOne(id,w)))break;}
  working=false;updateSend();
}
async function send(){
  const text=$('#ta').value.trim(),t=getTarget();
  if((!text&&t==='none')||working)return;
  const id=cur;errInfo=null;
  const sb=$('#send');sb.classList.remove('fly');void sb.offsetWidth;sb.classList.add('fly');
  if(text){working=true;updateSend();await api('/api/message',{chat_id:id,text});$('#ta').value='';autosize();await load();working=false;}
  const order=t==='both'?['claude','gemini']:t==='none'?[]:[t];
  if(order.length)await runAsk(id,order);else updateSend();
}

/* ---------- settings ---------- */
const KEYS=['shared','claude','gemini','claude_model','gemini_model'];
function fillSettings(c){const s=c.settings||{};
  for(const k of ['shared','claude','gemini'])$('#s_'+k).value=s[k]||'';
  pickers.claude.setValue(s.claude_model);pickers.gemini.setValue(s.gemini_model);
  const sh=c.kind==='shared';
  $('#f_claude').style.display=(sh||c.model==='claude')?'':'none';
  $('#f_gemini').style.display=(sh||c.model==='gemini')?'':'none';}
/* ---------- custom model dropdown ---------- */
const CHECK='<svg class="ck" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5 6.5 12 13 4.5"/></svg>';
function createPicker(root,hidden,labelId){
  const P={opts:[],value:'',open:false,hot:0,custom:false};
  const btn=h('button','pk-btn');btn.type='button';btn.setAttribute('aria-haspopup','listbox');btn.setAttribute('aria-expanded','false');btn.setAttribute('aria-labelledby',labelId);
  const val=h('span','pk-val');
  const chev=document.createElementNS('http://www.w3.org/2000/svg','svg');chev.setAttribute('class','pk-chev');chev.setAttribute('width','16');chev.setAttribute('height','16');chev.setAttribute('viewBox','0 0 16 16');chev.setAttribute('fill','none');chev.setAttribute('stroke','currentColor');chev.setAttribute('stroke-width','2');chev.setAttribute('stroke-linecap','round');chev.innerHTML='<path d="M4 6l4 4 4-4"/>';
  btn.append(val,chev);
  const panel=h('div','pk-panel');panel.setAttribute('role','listbox');panel.tabIndex=-1;
  const cwrap=h('div','pk-custom'),cin=h('div'),cinp=h('input');cinp.placeholder='Enter the exact model name';cinp.setAttribute('aria-label','Custom model name');cinp.autocomplete='off';
  cin.appendChild(cinp);cwrap.appendChild(cin);
  root.append(btn,cwrap);document.body.appendChild(panel);
  const all=()=>[{value:'',label:'Default',note:'the model the program picks itself'},...P.opts,{value:'__custom',label:'Other model…',note:'type a name manually'}];
  function labelOf(v){if(!v)return{label:'Default'};const o=P.opts.find(x=>x.value===v);return o?o:{label:v,note:'custom name'};}
  function paintBtn(){const o=labelOf(P.value);val.textContent='';val.append(document.createTextNode(o.label));
    if(o.note&&P.value){const s=h('small',null,o.note);val.appendChild(s);}}
  function setCustom(on){P.custom=on;cwrap.classList.toggle('on',on);}
  function build(){panel.innerHTML='';const list=all();
    list.forEach((o,i)=>{
      if(o.value==='__custom')panel.appendChild(h('div','pk-sep'));
      const row=h('div','pk-opt'+(i===P.hot?' hot':''));row.setAttribute('role','option');row.id=root.id+'_o'+i;
      const sel=(o.value==='__custom')?P.custom:(!P.custom&&o.value===P.value);row.setAttribute('aria-selected',String(sel));
      const tx=h('span','tx');tx.appendChild(document.createTextNode(o.label));if(o.note)tx.appendChild(h('small',null,o.note));
      row.appendChild(tx);row.insertAdjacentHTML('beforeend',CHECK);
      row.onmouseenter=()=>{P.hot=i;panel.querySelectorAll('.pk-opt').forEach((r,j)=>r.classList.toggle('hot',j===i));};
      row.onclick=()=>choose(o);panel.appendChild(row);});
  }
  function place(){const r=btn.getBoundingClientRect();const room=innerHeight-r.bottom-14;const up=room<220&&r.top>room;
    panel.classList.toggle('up',up);panel.style.left=r.left+'px';panel.style.width=r.width+'px';
    if(up){panel.style.top='auto';panel.style.bottom=(innerHeight-r.top+8)+'px';panel.style.maxHeight=Math.min(300,r.top-20)+'px';}
    else{panel.style.bottom='auto';panel.style.top=(r.bottom+8)+'px';panel.style.maxHeight=Math.min(300,Math.max(140,room))+'px';}}
  function openP(){if(P.open)return;P.open=true;P.t=Date.now();const list=all();P.hot=Math.max(0,list.findIndex(o=>o.value===(P.custom?'__custom':P.value)));
    build();place();panel.classList.add('open');btn.setAttribute('aria-expanded','true');
    const hot=panel.querySelectorAll('.pk-opt')[P.hot];if(hot)hot.scrollIntoView({block:'nearest'});}
  function closeP(){if(!P.open)return;P.open=false;panel.classList.remove('open');btn.setAttribute('aria-expanded','false');}
  function choose(o){
    if(o.value==='__custom'){setCustom(true);closeP();cinp.value=(P.value&&!P.opts.some(x=>x.value===P.value))?P.value:'';setTimeout(()=>cinp.focus(),320);return;}
    setCustom(false);P.value=o.value;hidden.value=o.value;paintBtn();closeP();btn.focus();}
  btn.onclick=()=>P.open?closeP():openP();
  btn.addEventListener('keydown',e=>{
    const n=all().length;
    if(!P.open){if(['ArrowDown','ArrowUp','Enter',' '].includes(e.key)){e.preventDefault();openP();}return;}
    if(e.key==='Escape'){e.preventDefault();e.stopPropagation();closeP();}
    else if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();P.hot=(P.hot+(e.key==='ArrowDown'?1:n-1))%n;
      const rows=panel.querySelectorAll('.pk-opt');rows.forEach((r,j)=>r.classList.toggle('hot',j===P.hot));rows[P.hot].scrollIntoView({block:'nearest'});}
    else if(e.key==='Home'||e.key==='End'){e.preventDefault();P.hot=e.key==='Home'?0:n-1;const rows=panel.querySelectorAll('.pk-opt');rows.forEach((r,j)=>r.classList.toggle('hot',j===P.hot));rows[P.hot].scrollIntoView({block:'nearest'});}
    else if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(all()[P.hot]);}
    else if(e.key==='Tab')closeP();
  });
  cinp.addEventListener('input',()=>{P.value=cinp.value.trim();hidden.value=P.value;paintBtn();});
  document.addEventListener('mousedown',e=>{if(P.open&&!panel.contains(e.target)&&!btn.contains(e.target))closeP();});
  addEventListener('resize',closeP);
  document.addEventListener('scroll',e=>{if(P.open&&Date.now()-P.t>350&&!panel.contains(e.target))closeP();},true);
  P.setOptions=list=>{P.opts=list||[];paintBtn();if(P.open)build();};
  P.setValue=v=>{v=v||'';P.value=v;hidden.value=v;setCustom(false);paintBtn();};
  P.close=closeP;
  paintBtn();return P;
}
function initPickers(){
  pickers.claude=createPicker($('#pk_claude'),$('#s_claude_model'),'lab_cm');
  pickers.gemini=createPicker($('#pk_gemini'),$('#s_gemini_model'),'lab_gm');
}
async function loadModels(refresh){
  const g=$('#gnote');g.textContent='Looking for Gemini models…';
  api('/api/models/claude').then(r=>pickers.claude.setOptions(r.models||[])).catch(()=>{});
  try{const r=await api('/api/models/gemini'+(refresh?'?refresh=1':''));
    pickers.gemini.setOptions(r.models||[]);
    g.textContent=r.error?(r.error+' You can choose "Other model…" and type a name manually.'):('Models found: '+r.models.length+'.');
  }catch(e){g.textContent='Could not get the list. You can type a name manually via "Other model…".';}
}
function openDrawerFor(c){drawerFor=c.id;fillSettings(c);}
function openDrawer(){const c=chat();if(!c)return;openDrawerFor(c);
  $('#drawer').inert=false;$('#drawer').classList.add('open');$('#drawerVeil').classList.add('on');
  loadModels();setTimeout(()=>$('#s_shared').focus({preventScroll:true}),250);}
function closeDrawer(){drawerFor=null;pickers.claude.close();pickers.gemini.close();$('#drawer').classList.remove('open');$('#drawerVeil').classList.remove('on');$('#drawer').inert=true;}
async function saveSettings(asDefault){
  const s={};for(const k of KEYS)s[k]=$('#s_'+k).value;
  const r=await api('/api/settings',{chat_id:cur,settings:s,as_default:!!asDefault});
  if(r.error){toast(r.error);return;}
  await load();toast(asDefault?'Saved and will apply to new chats':'Settings saved');
}

/* ---------- menu, drawer, dialogs ---------- */
function setMenu(open){$('#menu').classList.toggle('open',open);$('#bmore').setAttribute('aria-expanded',String(open));}
function openRail(){$('#rail').classList.add('open');$('#railVeil').classList.add('on');}
function closeRail(){$('#rail').classList.remove('open');$('#railVeil').classList.remove('on');}
function ask2(title,text,ok){return new Promise(res=>{
  const d=$('#dlgConfirm');$('#dcT').textContent=title;$('#dcP').textContent=text;$('#dcOk').textContent=ok;
  d.onclose=()=>res(d.returnValue==='ok');d.returnValue='cancel';d.showModal();});}
function openNew(){const d=$('#dlgNew');$('#nTitle').value='';document.querySelector('input[name=kind][value=shared]').checked=true;d.returnValue='cancel';d.showModal();$('#nTitle').focus();}
async function forkTo(model){setMenu(false);const c=await api('/api/fork',{chat_id:cur,model});cur=null;await load();selectChat(c.id);}

$('#newBtn').onclick=openNew;$('#newBtn2').onclick=openNew;
$('#dlgNew').addEventListener('close',async()=>{
  if($('#dlgNew').returnValue!=='ok')return;
  const [kind,model]=document.querySelector('input[name=kind]:checked').value.split(':');
  const title=$('#nTitle').value.trim()||'Untitled';
  const c=await api('/api/chat',{title,kind,model});await load();selectChat(c.id);closeRail();
});
$('#burger').onclick=openRail;$('#railVeil').onclick=closeRail;
$('#bset').onclick=openDrawer;$('#dclose').onclick=closeDrawer;$('#drawerVeil').onclick=closeDrawer;
$('#ssave').onclick=()=>saveSettings(false);$('#sdef').onclick=()=>saveSettings(true);$('#sref').onclick=()=>loadModels(true);
$('#bmore').onclick=e=>{e.stopPropagation();setMenu(!$('#menu').classList.contains('open'));};
document.addEventListener('click',e=>{if(!e.target.closest('#menu'))setMenu(false);});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){setMenu(false);if($('#drawer').classList.contains('open'))closeDrawer();closeRail();}});
$('#mfc').onclick=()=>forkTo('claude');$('#mfg').onclick=()=>forkTo('gemini');
$('#mdel').onclick=async()=>{setMenu(false);const c=chat();if(!c)return;
  if(!(await ask2('Delete chat "'+c.title+'"?','All messages in this chat will be deleted. This cannot be undone.','Delete')))return;
  await api('/api/delete',{chat_id:c.id});cur=null;renderedChat=null;closeDrawer();await load();};
$('#send').onclick=send;
$('#ta').addEventListener('input',()=>{autosize();updateSend();});
$('#ta').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();send();}});
document.querySelectorAll('input[name=t]').forEach(r=>r.addEventListener('change',()=>setTarget(r.value)));
/* ripple on press */
document.addEventListener('pointerdown',e=>{
  const b=e.target.closest&&e.target.closest('.btn,.send,.iconbtn');if(!b||b.disabled)return;
  const r=b.getBoundingClientRect(),d=Math.max(r.width,r.height)*2,s=document.createElement('span');
  s.className='ripple';s.style.cssText='width:'+d+'px;height:'+d+'px;left:'+(e.clientX-r.left-d/2)+'px;top:'+(e.clientY-r.top-d/2)+'px';
  b.appendChild(s);setTimeout(()=>s.remove(),700);
});
/* glow follows the cursor */
(function(){
  const rm=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const st=new WeakMap();let raf=0;const act=new Set();
  function tick(){
    raf=0;
    act.forEach(b=>{const s=st.get(b);
      s.x+=(s.tx-s.x)*.18;s.y+=(s.ty-s.y)*.18;
      b.style.setProperty('--mx',s.x.toFixed(1)+'px');b.style.setProperty('--my',s.y.toFixed(1)+'px');
      if(Math.abs(s.tx-s.x)<.4&&Math.abs(s.ty-s.y)<.4)act.delete(b);});
    if(act.size)raf=requestAnimationFrame(tick);
  }
  document.addEventListener('pointermove',e=>{
    const b=e.target.closest&&e.target.closest('.btn,.send');if(!b||b.disabled)return;
    const r=b.getBoundingClientRect();let s=st.get(b);
    const x=e.clientX-r.left,y=e.clientY-r.top;
    if(!s){s={x:x,y:y,tx:x,ty:y};st.set(b,s);}
    s.tx=x;s.ty=y;
    if(rm){s.x=x;s.y=y;b.style.setProperty('--mx',x+'px');b.style.setProperty('--my',y+'px');return;}
    act.add(b);if(!raf)raf=requestAnimationFrame(tick);
  });
})();
addEventListener('resize',()=>positionPill(false));
if(document.fonts&&document.fonts.ready)document.fonts.ready.then(()=>positionPill(false));
setTimeout(()=>document.body.classList.remove('intro'),1800);
initPickers();
load();
</script>
</body>
</html>
"""


def main():
    # On Windows, when piped (IDE, log), the console is often cp1252 and fails on non-Latin text.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for k, m in MODELS.items():
        if not shutil.which(m["bin"]):
            print(f"WARNING: \"{m['bin']}\" not found in PATH, {m['name']} replies will not work.")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Trio Chat: {url}  (Ctrl+C to stop)")
    if not os.environ.get("TRIO_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
