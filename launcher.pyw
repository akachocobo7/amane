"""
amane ランチャー — 設定画面つきGUI

ダブルクリックで起動。.envの編集と、パイプラインの起動・停止を1画面で行える。
Windows環境では .pyw 拡張子によりコンソールウィンドウが表示されない。
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
PIPELINE_SCRIPT = APP_DIR / "voice_pipeline.py"

# .envで管理する設定項目の定義
# (キー, ラベル, デフォルト値, 選択肢) — 選択肢ありならCombobox、なしならEntry
ENV_FIELDS: list[tuple[str, str, str, list[str] | None]] = [
    ("VOISONA_EMAIL", "VoiSona メールアドレス", "", None),
    ("VOISONA_API_PASSWORD", "API パスワード", "", None),
    ("VOISONA_VOICE_NAME", "ボイス名", "田中傘", None),
    ("INPUT_DEVICE_INDEX", "入力デバイス番号（空欄=既定）", "", None),
    ("WHISPER_DEVICE", "STTデバイス", "cuda", ["cuda", "cpu"]),
    ("SILENCE_MS", "無音判定 (ms)", "800", None),
    ("VAD_AGGRESSIVENESS", "VAD感度", "2", ["0", "1", "2", "3"]),
]


def parse_env(path: Path) -> dict[str, str]:
    """シンプルな .env パーサー。"""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        values[key] = val
    return values


def save_env(path: Path, values: dict[str, str]) -> None:
    """値が空でない項目だけ .env に書き出す。"""
    lines: list[str] = []
    for key, val in values.items():
        if val:
            lines.append(f"{key}={val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AmaneLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("amane — ランチャー")
        self.geometry("620x560")
        self.resizable(False, False)

        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None

        self._build_ui()
        self._load_env()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        # --- ヘッダー ---
        header = ttk.Frame(self, padding=(16, 12, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="amane", font=("", 18, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="リアルタイム音声変換パイプライン",
            foreground="gray",
        ).pack(anchor="w")

        # --- 設定フォーム ---
        form = ttk.LabelFrame(self, text="設定（.env）", padding=12)
        form.pack(fill="x", padx=16, pady=(8, 4))

        self._entries: dict[str, ttk.Entry | ttk.Combobox] = {}
        for key, label, _, choices in ENV_FIELDS:
            row = ttk.Frame(form)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
            if choices:
                widget = ttk.Combobox(row, values=choices, state="readonly", width=29)
                widget.pack(side="left", fill="x", expand=True)
            else:
                widget = ttk.Entry(row, width=32)
                widget.pack(side="left", fill="x", expand=True)
                if "パスワード" in label:
                    widget.configure(show="*")
            self._entries[key] = widget

        # --- ボタン ---
        btn_frame = ttk.Frame(self, padding=(16, 8))
        btn_frame.pack(fill="x")

        self._save_btn = ttk.Button(
            btn_frame, text="設定を保存", command=self._save_settings
        )
        self._save_btn.pack(side="left")

        self._stop_btn = ttk.Button(
            btn_frame, text="停止", command=self._stop_pipeline, state="disabled"
        )
        self._stop_btn.pack(side="right")

        self._start_btn = ttk.Button(
            btn_frame, text="起動", command=self._start_pipeline
        )
        self._start_btn.pack(side="right", padx=(0, 8))

        # --- ログ表示 ---
        log_frame = ttk.LabelFrame(self, text="ログ", padding=4)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(4, 12))

        self._log = scrolledtext.ScrolledText(
            log_frame, height=10, state="disabled", font=("Consolas", 9), wrap="word"
        )
        self._log.pack(fill="both", expand=True)

    def _load_env(self) -> None:
        values = parse_env(ENV_PATH)
        for key, _, default, choices in ENV_FIELDS:
            val = values.get(key, default)
            widget = self._entries[key]
            if choices:
                widget.set(val if val in choices else default)
            else:
                widget.delete(0, "end")
                widget.insert(0, val)

    def _save_settings(self) -> None:
        values = {key: self._entries[key].get().strip() for key, _, _, _ in ENV_FIELDS}
        save_env(ENV_PATH, values)
        self._append_log("[launcher] 設定を保存しました\n")

    def _append_log(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _start_pipeline(self) -> None:
        if self._process and self._process.poll() is None:
            self._append_log("[launcher] パイプラインは既に起動中です\n")
            return

        self._save_settings()
        self._append_log("[launcher] パイプラインを起動しています...\n")

        uv = self._find_uv()
        cmd = [uv, "run", "python", str(PIPELINE_SCRIPT)]

        device_idx = self._entries["INPUT_DEVICE_INDEX"].get().strip()
        if device_idx:
            cmd.extend(["--input-device", device_idx])

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=self._creation_flags(),
            )
        except FileNotFoundError:
            messagebox.showerror(
                "エラー",
                "uv が見つかりません。\n"
                "https://docs.astral.sh/uv/ からインストールしてください。",
            )
            return

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True
        )
        self._reader_thread.start()

    def _stop_pipeline(self) -> None:
        if self._process and self._process.poll() is None:
            self._append_log("[launcher] パイプラインを停止しています...\n")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._append_log("[launcher] 停止しました\n")

        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")

    def _read_output(self) -> None:
        """サブプロセスの出力を読み取り、ログ表示に反映する。"""
        proc = self._process
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            self.after(0, self._append_log, line)
        self.after(0, self._on_process_exit)

    def _on_process_exit(self) -> None:
        code = self._process.returncode if self._process else -1
        self._append_log(f"[launcher] パイプライン終了 (code={code})\n")
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")

    def _on_close(self) -> None:
        self._stop_pipeline()
        self.destroy()

    @staticmethod
    def _find_uv() -> str:
        """uvの実行パスを探す。"""
        if sys.platform == "win32":
            home = Path.home()
            candidates = [
                home / ".local" / "bin" / "uv.exe",
                home / ".cargo" / "bin" / "uv.exe",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)
        return "uv"

    @staticmethod
    def _creation_flags() -> int:
        if sys.platform == "win32":
            return subprocess.CREATE_NO_WINDOW
        return 0


if __name__ == "__main__":
    app = AmaneLauncher()
    app.mainloop()
