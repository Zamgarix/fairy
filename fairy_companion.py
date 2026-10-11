"""
Fairy Companion - a self-improving desktop assistant (PyQt5 + Ollama).

What it does
  * Chat with a small local model through Ollama (streamed, so no more timeouts).
  * Reads and edits THIS file: paste a code snippet (in chat or in the Code Lab tab)
    and it is merged into the script by function/class/method name, imports are
    added, a backup is saved, the result is test-started, and then you restart.
  * Ask the AI to write a snippet for you (/improve ...), review it, then apply it.

Chat commands:  /help  /read [name]  /outline  /improve <request>  /apply
                /preview  /undo  /clear

Requirements:  pip install PyQt5 requests      and      ollama serve
Optional:      put fairy.gif next to this file; edit fairy_config.json to tune.
"""
import ast
import difflib
import html
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time

import requests
from PyQt5.QtCore import Qt, QTimer, QPoint, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QMovie, QTextCursor
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPlainTextEdit, QPushButton, QTabWidget,
                             QTextEdit, QVBoxLayout, QWidget)

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(__file__)
CHARACTER_GIF = os.path.join(SCRIPT_PATH, "fairy.gif")
CONFIG_PATH = os.path.join(SCRIPT_PATH, "fairy_config.json")
BACKUP_DIR = os.path.join(SCRIPT_PATH, "fairy_backups")
SMOKE_TEST = "--smoke-test" in sys.argv

DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "model": "qwen2.5:0.5b",
    "keep_alive": "30m",        # keep the model in RAM so later replies are fast
    "connect_timeout": 5,       # seconds to reach Ollama
    "read_timeout": 180,        # seconds to wait for the NEXT chunk (covers cold start)
    "chat_num_ctx": 2048,
    "chat_num_predict": 400,
    "code_num_ctx": 4096,
    "code_num_predict": 700,
    "history_messages": 8,
}

CHAT_SYSTEM = "You are Fairy, a friendly desktop companion. Keep answers short and clear."
CODE_SYSTEM = (
    "You are a careful Python 3 / PyQt5 developer improving a desktop-pet script.\n"
    "Reply with exactly ONE ```python code block and nothing else.\n"
    "The block must contain ONLY the new or changed functions/classes, plus any imports they need.\n"
    "To change a method of an existing class, write it as a top-level `def name(self, ...)` "
    "- do not wrap it in the class.\n"
    "Keep changes small. Never output the whole file."
)

HELP_TEXT = (
    "Paste code in a ```python fence and I'll merge it into my own file.\n"
    "Methods: add a line `# target: ClassName` (or fill 'Target class' in Code Lab).\n"
    "Start with `# FULL-FILE` on the first line to replace the whole file.\n\n"
    "/read [name]       show my source, or one function/class/method\n"
    "/outline           list my classes and functions\n"
    "/improve <idea>    ask the AI to write a snippet (you review before applying)\n"
    "/preview           show what the snippet in Code Lab would change\n"
    "/apply             apply the snippet in Code Lab\n"
    "/undo              restore the previous backup\n"
    "/clear             clear this chat"
)


# --------------------------------------------------------------------------- config
def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def extract_code_blocks(text):
    """Return the code inside ``` fences (handles a missing closing fence)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks)
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)$", text, re.S)
    return m.group(1).strip("\n") if m else ""


# --------------------------------------------------------------------------- patching
class PatchError(Exception):
    pass


FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def _span(node):
    """(0-based start line, exclusive end line), including decorators."""
    start = node.lineno
    for d in getattr(node, "decorator_list", []):
        start = min(start, d.lineno)
    return start - 1, node.end_lineno


def _indent(text, n=4):
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


class SourcePatcher:
    """Reads this script and merges snippets into it by name (AST based)."""

    REQUIRED = ("SourcePatcher", "ChatWindow", "Lively3DFairy")

    def __init__(self, path, backup_dir, keep=25):
        self.path = path
        self.backup_dir = backup_dir
        self.keep = keep

    # ---- reading
    def read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()

    def outline(self, source=None):
        tree = ast.parse(source or self.read())
        out = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out.append(f"class {node.name}  (line {node.lineno})")
                for sub in node.body:
                    if isinstance(sub, FUNCS):
                        args = ", ".join(a.arg for a in sub.args.args)
                        out.append(f"    def {sub.name}({args})  (line {sub.lineno})")
            elif isinstance(node, FUNCS):
                args = ", ".join(a.arg for a in node.args.args)
                out.append(f"def {node.name}({args})  (line {node.lineno})")
        return "\n".join(out)

    def find_definition(self, name, source=None):
        source = source or self.read()
        lines = source.splitlines()
        for node in ast.parse(source).body:
            if getattr(node, "name", None) == name and isinstance(node, FUNCS + (ast.ClassDef,)):
                s, e = _span(node)
                return "\n".join(lines[s:e])
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, FUNCS) and name in (sub.name, f"{node.name}.{sub.name}"):
                        s, e = _span(sub)
                        return "\n".join(lines[s:e])
        return None

    def context_for(self, request, max_chars=2500):
        """Small prompt context: outline + the bodies of definitions the request names."""
        source = self.read()
        parts = ["FILE OUTLINE:\n" + self.outline(source)]
        words = set(re.findall(r"[a-z_]\w+", request.lower()))
        used = 0
        for node in ast.parse(source).body:
            candidates = [node] + (list(node.body) if isinstance(node, ast.ClassDef) else [])
            for n in candidates:
                if isinstance(n, FUNCS) and n.name.lower() in words:
                    body = self.find_definition(n.name, source) or ""
                    if used + len(body) <= max_chars:
                        parts.append("RELEVANT CODE:\n" + body)
                        used += len(body)
        return "\n\n".join(parts)

    # ---- planning a change
    def plan(self, snippet, target_class=None):
        """Return (new_source, [human readable actions]). Nothing is written."""
        snippet = textwrap.dedent(snippet).strip("\n")
        if not snippet.strip():
            raise PatchError("The snippet is empty.")
        if snippet.startswith("# FULL-FILE"):
            body = snippet.split("\n", 1)[1] if "\n" in snippet else ""
            if not body.strip():
                raise PatchError("# FULL-FILE needs the new file contents below it.")
            return body.rstrip("\n") + "\n", ["replace the whole file"]

        m = re.search(r"^#\s*target:\s*([A-Za-z_]\w*)", snippet, re.M)
        target_class = (target_class or "").strip() or (m.group(1) if m else None)
        try:
            stree = ast.parse(snippet)
        except SyntaxError as e:
            raise PatchError(f"Snippet has a syntax error on line {e.lineno}: {e.msg}")

        source = self.read()
        otree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        slines = snippet.splitlines()

        def seg(node):
            s, e = _span(node)
            return "\n".join(slines[s:e])

        top = {n.name: n for n in otree.body if isinstance(n, FUNCS + (ast.ClassDef,))}
        classes = {n.name: n for n in otree.body if isinstance(n, ast.ClassDef)}
        assigns = {n.targets[0].id: n for n in otree.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
        imports = [n for n in otree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        known_imports = {ast.dump(n) for n in imports}
        guard = next((n for n in otree.body if _is_main_guard(n)), None)

        if imports:
            import_pos = max(n.end_lineno for n in imports)
        elif (otree.body and isinstance(otree.body[0], ast.Expr)
              and isinstance(getattr(otree.body[0], "value", None), ast.Constant)
              and isinstance(otree.body[0].value.value, str)):
            import_pos = otree.body[0].end_lineno
        else:
            import_pos = 0
        const_pos = import_pos
        for n in otree.body:
            if isinstance(n, FUNCS + (ast.ClassDef,)) or _is_main_guard(n):
                break
            if isinstance(n, ast.Assign):
                const_pos = max(const_pos, n.end_lineno)
        code_pos = guard.lineno - 1 if guard else len(lines)

        edits, actions = [], []   # edit = (start, end, text, seq)

        def add(start, end, text):
            edits.append((start, end, text, len(edits)))

        for node in stree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if ast.dump(node) in known_imports:
                    continue
                known_imports.add(ast.dump(node))
                add(import_pos, import_pos, seg(node) + "\n")
                actions.append(f"add import: {seg(node)}")

            elif isinstance(node, FUNCS + (ast.ClassDef,)):
                text = seg(node)
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                first = node.args.args[0].arg if (isinstance(node, FUNCS) and node.args.args) else ""
                if first in ("self", "cls"):
                    cls = self._pick_class(classes, node.name, target_class)
                    old = next((s for s in cls.body if isinstance(s, FUNCS) and s.name == node.name), None)
                    if old:
                        s, e = _span(old)
                        add(s, e, _indent(text) + "\n")
                        actions.append(f"replace method {cls.name}.{node.name}")
                    else:
                        add(cls.end_lineno, cls.end_lineno, "\n" + _indent(text) + "\n")
                        actions.append(f"add method {cls.name}.{node.name}")
                elif node.name in top:
                    s, e = _span(top[node.name])
                    add(s, e, text + "\n")
                    actions.append(f"replace {kind} {node.name}")
                else:
                    new = text + "\n\n\n" if guard else "\n\n" + text + "\n"
                    add(code_pos, code_pos, new)
                    actions.append(f"add {kind} {node.name}")

            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)):
                name = node.targets[0].id
                if name in assigns:
                    s, e = _span(assigns[name])
                    add(s, e, seg(node) + "\n")
                    actions.append(f"replace constant {name}")
                else:
                    add(const_pos, const_pos, seg(node) + "\n")
                    actions.append(f"add constant {name}")

            elif _is_main_guard(node):
                if guard:
                    s, e = _span(guard)
                    add(s, e, seg(node) + "\n")
                    actions.append("replace the `if __name__ == '__main__'` block")
                else:
                    add(len(lines), len(lines), "\n\n" + seg(node) + "\n")
                    actions.append("add the `if __name__ == '__main__'` block")

            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)):
                continue
            else:
                raise PatchError(f"Snippet line {node.lineno}: a top-level "
                                 f"`{type(node).__name__}` statement can't be placed. "
                                 "Wrap it in a function or method.")

        if not edits:
            raise PatchError("Nothing to change - everything in the snippet is already in the file.")
        self._check_overlaps(edits)
        # Apply bottom-up so earlier line numbers stay valid; at equal positions,
        # replacements go first, then insertions in reverse order (keeps their order).
        for start, end, text, _ in sorted(edits, key=lambda e: (e[0], e[1] > e[0], e[3]), reverse=True):
            lines[start:end] = [text]
        new_source = "".join(lines)
        if not new_source.endswith("\n"):
            new_source += "\n"
        return new_source, actions

    @staticmethod
    def _pick_class(classes, method, target):
        if target:
            if target not in classes:
                raise PatchError(f"No class named {target}. Classes: {', '.join(classes)}")
            return classes[target]
        owners = [c for c in classes.values()
                  if any(isinstance(s, FUNCS) and s.name == method for s in c.body)]
        if len(owners) == 1:
            return owners[0]
        if len(owners) > 1:
            raise PatchError(f"`{method}` exists in several classes ({', '.join(c.name for c in owners)}). "
                             "Add `# target: ClassName` or fill in 'Target class'.")
        raise PatchError(f"`{method}` is a new method - tell me which class gets it: add "
                         f"`# target: ClassName` (one of: {', '.join(classes)}) or fill in 'Target class'.")

    @staticmethod
    def _check_overlaps(edits):
        for i, a in enumerate(edits):
            for b in edits[i + 1:]:
                a_rep, b_rep = a[1] > a[0], b[1] > b[0]
                if a_rep and b_rep and a[0] < b[1] and b[0] < a[1]:
                    raise PatchError("Two parts of the snippet change the same code "
                                     "(is a name defined twice?).")
                for ins, rep in ((a, b), (b, a)):
                    if ins[1] == ins[0] and rep[1] > rep[0] and rep[0] < ins[0] < rep[1]:
                        raise PatchError("The snippet would insert code inside code it also replaces.")

    # ---- safety checks
    def validate(self, source):
        try:
            tree = ast.parse(source)
            compile(source, self.path, "exec")
        except (SyntaxError, ValueError) as e:
            raise PatchError(f"The result would not compile (line {getattr(e, 'lineno', '?')}): {e}")
        names = {n.name for n in tree.body if hasattr(n, "name")}
        missing = [n for n in self.REQUIRED if n not in names]
        if missing:
            raise PatchError("Refusing: the result would remove core part(s): " + ", ".join(missing))
        if not any(_is_main_guard(n) for n in tree.body):
            raise PatchError("Refusing: the result would remove the `if __name__ == '__main__'` block.")

    def smoke_test(self, source, timeout=12):
        """Start the candidate headless; a crash on startup means we don't write it."""
        candidate = os.path.join(os.path.dirname(self.path), ".fairy_candidate.py")
        with open(candidate, "w", encoding="utf-8") as f:
            f.write(source)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        try:
            proc = subprocess.run([sys.executable, candidate, "--smoke-test"], env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return True, f"started fine (no crash within {timeout}s)"
        except OSError as e:
            return False, str(e)
        finally:
            try:
                os.remove(candidate)
            except OSError:
                pass
        if proc.returncode == 0:
            return True, "started and exited cleanly"
        return False, (proc.stderr or proc.stdout or "unknown error")[-1500:]

    # ---- writing
    def diff(self, old, new):
        return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                            "current", "proposed", n=2))

    def _write_atomic(self, text):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.path)

    def commit(self, new_source):
        os.makedirs(self.backup_dir, exist_ok=True)
        backup = os.path.join(self.backup_dir, time.strftime("fairy_%Y%m%d_%H%M%S.py"))
        with open(backup, "w", encoding="utf-8") as f:
            f.write(self.read())
        self._write_atomic(new_source)
        for old in self._backups()[:-self.keep]:
            try:
                os.remove(os.path.join(self.backup_dir, old))
            except OSError:
                pass
        return backup

    def _backups(self):
        if not os.path.isdir(self.backup_dir):
            return []
        return sorted(f for f in os.listdir(self.backup_dir) if f.endswith(".py"))

    def undo(self):
        backups = self._backups()
        if not backups:
            raise PatchError("There are no backups to restore.")
        name = backups[-1]
        path = os.path.join(self.backup_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        ast.parse(text)
        self._write_atomic(text)
        os.remove(path)
        return name


# --------------------------------------------------------------------------- workers
class OllamaWorker(QThread):
    """Streams a chat completion so the GUI stays responsive and nothing times out
    while a small model is still loading or thinking."""
    token = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg, messages, options):
        super().__init__()
        self.cfg = dict(cfg)
        self.messages = messages
        self.options = options
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        c = self.cfg
        payload = {"model": c["model"], "messages": self.messages, "stream": True,
                   "keep_alive": c["keep_alive"], "options": self.options}
        parts = []
        try:
            with requests.post(c["ollama_url"] + "/api/chat", json=payload, stream=True,
                               timeout=(c["connect_timeout"], c["read_timeout"])) as r:
                if r.status_code != 200:
                    hint = f" (try: ollama pull {c['model']})" if r.status_code == 404 else ""
                    raise RuntimeError(f"Ollama answered HTTP {r.status_code}: {r.text[:200]}{hint}")
                for line in r.iter_lines():
                    if self._cancel:
                        self.failed.emit("Stopped.")
                        return
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        parts.append(piece)
                        self.token.emit(piece)
                    if data.get("done"):
                        break
        except requests.ConnectionError:
            self.failed.emit(f"Can't reach Ollama at {c['ollama_url']}. Is it running? (ollama serve)")
        except requests.ReadTimeout:
            self.failed.emit(f"No data from the model for {c['read_timeout']}s. "
                             "Try again, or raise read_timeout in fairy_config.json.")
        except Exception as e:  # noqa: BLE001 - show any failure in the UI
            self.failed.emit(f"Error: {e}")
        else:
            self.done.emit("".join(parts))


class StartupWorker(QThread):
    """Lists installed models and pre-loads the chosen one into memory."""
    status = pyqtSignal(str)
    models = pyqtSignal(list, str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)

    def run(self):
        c = self.cfg
        try:
            r = requests.get(c["ollama_url"] + "/api/tags", timeout=c["connect_timeout"])
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            self.status.emit("Ollama isn't reachable - start it with: ollama serve")
            return
        if not names:
            self.status.emit(f"No models installed - run: ollama pull {c['model']}")
            return
        chosen = c["model"] if c["model"] in names else next((n for n in names if "qwen" in n), names[0])
        self.models.emit(names, chosen)
        self.status.emit(f"Loading {chosen} into memory...")
        try:
            requests.post(c["ollama_url"] + "/api/generate",
                          json={"model": chosen, "keep_alive": c["keep_alive"]},
                          timeout=(c["connect_timeout"], c["read_timeout"]))
            self.status.emit("Ready")
        except Exception as e:  # noqa: BLE001
            self.status.emit(f"Couldn't pre-load the model: {e}")


# --------------------------------------------------------------------------- UI
def mono_font():
    font = QFont("Consolas")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(9)
    return font


class ChatInput(QPlainTextEdit):
    """Enter sends, Shift+Enter inserts a newline (so pasted code stays intact)."""
    submitted = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class ChatWindow(QWidget):
    def __init__(self, fairy_mascot):
        super().__init__()
        self.mascot = fairy_mascot
        self.cfg = load_config()
        self.patcher = SourcePatcher(FILE_PATH, BACKUP_DIR)
        self.history = []
        self.worker = None
        self.startup = None
        self.busy = False
        self.mode = "chat"

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowTitle("Fairy Companion")
        self.resize(480, 580)
        self._build_ui()
        self._note("Hi! Type a message, or paste code in a ```python fence to upgrade my script. "
                   "/help lists commands.")

        QApplication.instance().aboutToQuit.connect(self.shutdown)
        if not SMOKE_TEST:
            self.startup = StartupWorker(self.cfg)
            self.startup.status.connect(self.status.setText)
            self.startup.models.connect(self.on_models)
            self.startup.start()

    # ---- layout
    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.status = QLabel("Starting...")
        self.status.setWordWrap(True)
        self.model_box = QComboBox()
        self.model_box.addItem(self.cfg["model"])
        self.model_box.currentTextChanged.connect(self.on_model_changed)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_worker)
        self.stop_btn.hide()
        top.addWidget(self.status, 1)
        top.addWidget(self.model_box)
        top.addWidget(self.stop_btn)
        root.addLayout(top)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # Chat tab
        chat_tab = QWidget()
        cl = QVBoxLayout(chat_tab)
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        cl.addWidget(self.chat_view)
        row = QHBoxLayout()
        self.input_box = ChatInput()
        self.input_box.setPlaceholderText("Ask Fairy... (Enter = send, Shift+Enter = new line)")
        self.input_box.setFixedHeight(70)
        self.input_box.submitted.connect(self.on_submit)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_submit)
        row.addWidget(self.input_box)
        row.addWidget(self.send_btn)
        cl.addLayout(row)
        self.tabs.addTab(chat_tab, "Chat")

        # Code Lab tab
        lab = QWidget()
        ll = QVBoxLayout(lab)
        ll.addWidget(QLabel("Snippet to merge into this script (functions, classes, methods, imports):"))
        self.snippet_box = QPlainTextEdit()
        self.snippet_box.setFont(mono_font())
        self.snippet_box.setPlaceholderText("def new_feature(...):\n    ...\n\n"
                                            "# target: ChatWindow   <- needed for NEW methods")
        ll.addWidget(self.snippet_box, 3)

        trow = QHBoxLayout()
        trow.addWidget(QLabel("Target class:"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("optional, for new methods")
        trow.addWidget(self.target_edit)
        ll.addLayout(trow)

        brow = QHBoxLayout()
        for label, slot in (("Preview", self.preview), ("Apply to file", self.apply_from_box),
                            ("Undo last", self.undo_last)):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            brow.addWidget(btn)
        ll.addLayout(brow)

        grow = QHBoxLayout()
        self.ai_request = QLineEdit()
        self.ai_request.setPlaceholderText("Ask the AI to write a snippet, e.g. 'make move_pet faster'")
        self.ai_request.returnPressed.connect(self.generate_from_field)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self.generate_from_field)
        grow.addWidget(self.ai_request)
        grow.addWidget(self.gen_btn)
        ll.addLayout(grow)

        self.lab_log = QPlainTextEdit()
        self.lab_log.setReadOnly(True)
        self.lab_log.setFont(mono_font())
        ll.addWidget(self.lab_log, 3)

        vrow = QHBoxLayout()
        for label, slot in (("View file", lambda: self.show_in_lab(self.patcher.read())),
                            ("Outline", lambda: self.show_in_lab(self.patcher.outline()))):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            vrow.addWidget(btn)
        ll.addLayout(vrow)
        self.tabs.addTab(lab, "Code Lab")

    # ---- small helpers
    def _note(self, text):
        self.chat_view.append(f"<i style='color:gray'>{html.escape(text).replace(chr(10), '<br>')}</i>")

    def _user_say(self, text):
        self.chat_view.append(f"<b>You:</b> {html.escape(text).replace(chr(10), '<br>')}")

    def _log(self, text):
        self.lab_log.appendPlainText(text)

    def show_in_lab(self, text):
        self.lab_log.setPlainText(text)
        self.tabs.setCurrentIndex(1)

    def set_busy(self, busy):
        self.busy = busy
        self.stop_btn.setVisible(busy)
        self.send_btn.setEnabled(not busy)
        self.gen_btn.setEnabled(not busy)

    def on_models(self, names, chosen):
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(names)
        self.model_box.setCurrentText(chosen)
        self.model_box.blockSignals(False)
        self.cfg["model"] = chosen

    def on_model_changed(self, name):
        if name:
            self.cfg["model"] = name
            save_config(self.cfg)
            self.status.setText(f"Model: {name} (loads on first use)")

    # ---- input routing
    def on_submit(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        if self.busy:
            self._note("Still working on the last request - press Stop to cancel it.")
            return
        self.input_box.clear()
        if text.startswith("/"):
            self.handle_command(text)
            return
        code = extract_code_blocks(text) if "```" in text else ""
        if code:
            self._user_say(f"(pasted {len(code.splitlines())} lines of code)")
            self.snippet_box.setPlainText(code)
            self.apply_snippet(code, self.target_edit.text())
            return
        self._user_say(text)
        self.chat(text)

    def handle_command(self, text):
        cmd, _, arg = text[1:].partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        try:
            if cmd == "help":
                self._note(HELP_TEXT)
            elif cmd == "read":
                source = self.patcher.read()
                body = self.patcher.find_definition(arg, source) if arg else source
                self.show_in_lab(body if body else f"No definition named {arg!r}.")
            elif cmd == "outline":
                self.show_in_lab(self.patcher.outline())
            elif cmd == "undo":
                self.undo_last()
            elif cmd == "apply":
                self.apply_from_box()
            elif cmd == "preview":
                self.preview()
            elif cmd == "improve":
                if arg:
                    self.generate_snippet(arg)
                else:
                    self._note("Usage: /improve <what you want changed>")
            elif cmd == "clear":
                self.chat_view.clear()
                self.history.clear()
            else:
                self._note(f"Unknown command /{cmd}. Try /help.")
        except (OSError, SyntaxError, PatchError) as e:
            self._note(f"Error: {e}")

    # ---- talking to the model
    def start_worker(self, messages, mode):
        prefix = "chat" if mode == "chat" else "code"
        options = {"num_ctx": self.cfg[f"{prefix}_num_ctx"],
                   "num_predict": self.cfg[f"{prefix}_num_predict"],
                   "temperature": 0.7 if mode == "chat" else 0.2}
        self.mode = mode
        self.set_busy(True)
        self.status.setText("Thinking... (the first reply after loading can take a moment)")
        self.worker = OllamaWorker(self.cfg, messages, options)
        self.worker.token.connect(self.on_token)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def chat(self, text):
        keep = self.cfg["history_messages"]
        messages = ([{"role": "system", "content": CHAT_SYSTEM}]
                    + self.history[-keep:] + [{"role": "user", "content": text}])
        self._pending_user = text
        self.chat_view.append("<b>Fairy:</b> ")
        self.start_worker(messages, "chat")

    def generate_from_field(self):
        request = self.ai_request.text().strip()
        if request and not self.busy:
            self.generate_snippet(request)

    def generate_snippet(self, request):
        if self.busy:
            return
        target = self.target_edit.text().strip()
        try:
            context = self.patcher.context_for(request)
        except (OSError, SyntaxError) as e:
            self._note(f"Couldn't read my own file: {e}")
            return
        prompt = f"{context}\n\nRequest: {request}"
        if target:
            prompt += f"\nTarget class: {target}"
        self.snippet_box.clear()
        self.show_in_lab("Generating a snippet...")
        messages = [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": prompt}]
        self.start_worker(messages, "code")

    def on_token(self, piece):
        target = self.chat_view if self.mode == "chat" else self.snippet_box
        target.moveCursor(QTextCursor.End)
        target.insertPlainText(piece)

    def on_done(self, full_text):
        self.set_busy(False)
        self.status.setText("Ready")
        if self.mode == "chat":
            self.history += [{"role": "user", "content": self._pending_user},
                             {"role": "assistant", "content": full_text}]
            self.history = self.history[-self.cfg["history_messages"] * 2:]
            self.chat_view.append("")
        else:
            code = extract_code_blocks(full_text) or full_text.strip()
            self.snippet_box.setPlainText(code)
            self._log("Snippet generated. Review it, then press 'Apply to file'.")
            self.preview()

    def on_failed(self, message):
        self.set_busy(False)
        self.status.setText("Error" if message != "Stopped." else "Ready")
        if self.mode == "chat":
            self._note(message)
        else:
            self._log(message)

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()

    # ---- code lab actions
    def preview(self):
        snippet = self.snippet_box.toPlainText()
        try:
            new, actions = self.patcher.plan(snippet, self.target_edit.text())
            self.patcher.validate(new)
        except PatchError as e:
            self._log(f"Cannot apply: {e}")
            self.tabs.setCurrentIndex(1)
            return
        self._log("Would do:\n  - " + "\n  - ".join(actions))
        self._log(self.patcher.diff(self.patcher.read(), new) or "(no textual change)")
        self.tabs.setCurrentIndex(1)

    def apply_from_box(self):
        self.apply_snippet(self.snippet_box.toPlainText(), self.target_edit.text())

    def apply_snippet(self, snippet, target_class):
        try:
            old = self.patcher.read()
            new, actions = self.patcher.plan(snippet, target_class)
            self.patcher.validate(new)
        except (OSError, PatchError) as e:
            self._note(f"Not applied: {e}")
            self._log(f"Not applied: {e}")
            return
        if new == old:
            self._note("That snippet is identical to what's already in the file.")
            return

        box = QMessageBox(self)
        box.setWindowTitle("Apply code change?")
        box.setText("This will update " + os.path.basename(FILE_PATH) + ":\n- " + "\n- ".join(actions))
        box.setInformativeText("A backup is saved and the new code is test-started before anything is written. "
                               "Open 'Show Details' to see the diff.")
        box.setDetailedText(self.patcher.diff(old, new))
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec_() != QMessageBox.Yes:
            self._note("Cancelled - file unchanged.")
            return

        self.status.setText("Test-starting the new code...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, info = self.patcher.smoke_test(new)
        finally:
            QApplication.restoreOverrideCursor()
        self.status.setText("Ready")
        if not ok:
            self._note("The new code crashed on startup, so I did NOT write it.")
            self._log("Startup test failed:\n" + info)
            self.tabs.setCurrentIndex(1)
            return

        try:
            backup = self.patcher.commit(new)
        except OSError as e:
            self._note(f"Couldn't write the file: {e}")
            return
        self._note(f"Saved ({'; '.join(actions)}). Backup: {os.path.basename(backup)}")
        answer = QMessageBox.question(self, "Restart?", "Changes saved. Restart now to load them?",
                                      QMessageBox.Yes | QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.restart_app()

    def undo_last(self):
        try:
            name = self.patcher.undo()
        except (OSError, SyntaxError, PatchError) as e:
            self._note(f"Undo failed: {e}")
            return
        self._note(f"Restored {name}. Restart to load it.")
        if QMessageBox.question(self, "Restart?", "Restored the previous version. Restart now?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.restart_app()

    def restart_app(self):
        args = [a for a in sys.argv[1:] if a != "--smoke-test"]
        subprocess.Popen([sys.executable, FILE_PATH] + args)
        QApplication.quit()

    def shutdown(self):
        for w in (self.worker, self.startup):
            if w is not None and w.isRunning():
                if hasattr(w, "cancel"):
                    w.cancel()
                if not w.wait(1500):
                    w.terminate()
                    w.wait(500)


class Lively3DFairy(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(120, 120)

        layout = QVBoxLayout()
        self.avatar_label = QLabel()
        self.avatar_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.avatar_label)
        self.setLayout(layout)

        if os.path.exists(CHARACTER_GIF):
            self.movie = QMovie(CHARACTER_GIF)
            self.movie.setScaledSize(QSize(100, 100))
            self.avatar_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.avatar_label.setText("🧚✨")
            self.avatar_label.setStyleSheet("font-size: 46px;")

        self.dx = 0
        self.dy = 0
        self.dragging = False
        self.drag_position = QPoint()
        self.press_pos = QPoint()
        self.screen_geo = QApplication.primaryScreen().availableGeometry()

        self.chat_box = ChatWindow(self)

        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.update_state)
        self.state_timer.start(2000)

        self.pos_timer = QTimer(self)
        self.pos_timer.timeout.connect(self.move_pet)
        self.pos_timer.start(50)

    def update_state(self):
        if random.random() < 0.4:
            self.dx = random.choice([-2, -1, 1, 2])
            self.dy = random.choice([-2, -1, 1, 2])
        else:
            self.dx = 0
            self.dy = 0

    def move_pet(self):
        if (self.dx == 0 and self.dy == 0) or self.dragging or self.chat_box.isVisible():
            return
        g = self.screen_geo
        max_x = g.right() - self.width()
        max_y = g.bottom() - self.height()
        x, y = self.x() + self.dx, self.y() + self.dy
        if x < g.left() or x > max_x:
            self.dx *= -1
            x = min(max(x, g.left()), max_x)
        if y < g.top() or y > max_y:
            self.dy *= -1
            y = min(max(y, g.top()), max_y)
        self.move(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.press_pos = event.globalPos()
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            self.dragging = False

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if (event.globalPos() - self.press_pos).manhattanLength() > 5:
                self.dragging = True
            if self.dragging:
                self.move(event.globalPos() - self.drag_position)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            was_drag, self.dragging = self.dragging, False
            if not was_drag:
                self.toggle_chat()

    def toggle_chat(self):
        cb = self.chat_box
        if cb.isVisible():
            cb.hide()
            return
        g = self.screen_geo
        x = self.x() + self.width() + 10
        if x + cb.width() > g.right():
            x = self.x() - cb.width() - 10
        y = min(max(self.y(), g.top()), g.bottom() - cb.height())
        cb.move(max(x, g.left()), y)
        cb.show()
        cb.raise_()
        cb.input_box.setFocus()


if __name__ == "__main__":
    if SMOKE_TEST:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv)
    fairy = Lively3DFairy()
    fairy.move(300, 300)
    fairy.show()
    if SMOKE_TEST:
        QTimer.singleShot(800, app.quit)
    sys.exit(app.exec_())
