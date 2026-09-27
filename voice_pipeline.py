"""
poitto — リアルタイム音声変換パイプライン

マイク → STT(faster-whisper) → VoiSona Talk(TTS) → 音声出力

使い方:
  uv run voice_pipeline.py                 # 通常起動
  uv run voice_pipeline.py --list-devices   # オーディオデバイス一覧
  uv run voice_pipeline.py --input-device 3 # マイクを指定して起動
"""

import argparse
import logging
import os
import re
import sys
import queue
import threading
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# ログ設定（ログ出力は英語、UIメッセージは日本語）
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("poitto")

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000  # STT用サンプリングレート (Hz)
CHANNELS = 1  # モノラル
FRAME_DURATION_MS = 30  # VADフレーム長 (ms) — 10/20/30のいずれか
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples

MIN_SPEECH_MS = 300  # これより短い発話は無視 (ms)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / FRAME_DURATION_MS)

# Whisperハルシネーション対策: 無音や短い音に対して繰り返し出力される定型フレーズ
HALLUCINATION_PHRASES = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "お疲れ様でした",
    "おやすみなさい",
    "どうもありがとうございました",
    "ありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "Thanks for watching!",
    "Thank you for watching!",
    "Bye!",
    "Bye bye!",
}

# STT設定
WHISPER_MODEL_SIZE = "small"  # small / medium / large-v3
WHISPER_DEVICE = "cpu"  # cuda / cpu
WHISPER_COMPUTE_TYPE = "int8"  # float16 / int8（CPUではint8を使用）

# VoiSona Talk API
VOISONA_BASE_URL = "http://localhost:32766/api/talk/v1"

# フレーズ分割パターン: 句読点・感嘆符・疑問符・読点で区切る
PHRASE_SPLIT_PATTERN = re.compile(r"(?<=[。！？、．，!?])")


# ---------------------------------------------------------------------------
# 設定の読み込み
# ---------------------------------------------------------------------------
def load_settings() -> dict:
    """環境変数から各種設定を読み込む。"""
    load_dotenv()

    silence_ms = int(os.getenv("SILENCE_MS", "800"))
    # VAD aggressiveness: 0(最も緩い)〜3(最も厳しい)
    vad_aggressiveness = int(os.getenv("VAD_AGGRESSIVENESS", "2"))

    return {
        "email": os.getenv("VOISONA_EMAIL"),
        "password": os.getenv("VOISONA_API_PASSWORD"),
        "voice_name": os.getenv("VOISONA_VOICE_NAME", "田中傘"),
        "input_device_index": os.getenv("INPUT_DEVICE_INDEX"),
        "silence_ms": silence_ms,
        "silence_frames": int(silence_ms / FRAME_DURATION_MS),
        "vad_aggressiveness": vad_aggressiveness,
    }


# ---------------------------------------------------------------------------
# フレーズ分割
# ---------------------------------------------------------------------------
def split_phrases(text: str) -> list[str]:
    """テキストを句読点でフレーズに分割する。

    「こんにちは、今日はいい天気ですね。元気ですか？」
    → ["こんにちは、", "今日はいい天気ですね。", "元気ですか？"]
    """
    parts = PHRASE_SPLIT_PATTERN.split(text)
    # 空文字列を除去して返す
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# VoiSona Talk API
# ---------------------------------------------------------------------------
class VoiSonaTalk:
    """VoiSona Talk REST APIのラッパー。"""

    def __init__(self, email: str, password: str):
        self.auth = (email, password)
        self.base = VOISONA_BASE_URL
        self.voice_name: str | None = None
        self.voice_version: str | None = None
        # TTS再生用のスレッドとキュー
        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._tts_busy = threading.Event()  # TTSが再生中かどうか

    def fetch_voices(self) -> list[dict]:
        """利用可能なボイス一覧を取得する。"""
        url = f"{self.base}/voices"
        try:
            resp = requests.get(url, auth=self.auth, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            # APIは {"items": [...]} 形式で返す
            voices = data.get("items", data) if isinstance(data, dict) else data
            log.info("Available voices: %s", voices)
            return voices
        except requests.RequestException as e:
            log.error("Failed to fetch voices: %s", e)
            return []

    def select_voice(self, name: str) -> bool:
        """指定した名前のボイスを選択し、voice_name / voice_version を設定する。

        name には voice_name（例: "tanaka-san_ja_JP"）または
        display_names の日本語名（例: "田中傘"）のどちらでも指定できる。
        """
        voices = self.fetch_voices()
        for v in voices:
            # voice_name で直接一致
            if v.get("voice_name") == name:
                self.voice_name = v["voice_name"]
                self.voice_version = v.get("voice_version")
                log.info(
                    "Selected voice: %s (version=%s)",
                    self.voice_name,
                    self.voice_version,
                )
                return True
            # display_names の日本語名で一致
            for dn in v.get("display_names", []):
                if dn.get("name") == name:
                    self.voice_name = v["voice_name"]
                    self.voice_version = v.get("voice_version")
                    log.info(
                        "Selected voice: %s (version=%s) matched by display name '%s'",
                        self.voice_name,
                        self.voice_version,
                        name,
                    )
                    return True
        log.error("Voice '%s' not found", name)
        return False

    def _synthesize(self, text: str) -> bool:
        """テキストをVoiSona Talkで合成する（同期呼び出し）。"""
        url = f"{self.base}/speech-syntheses"
        payload = {
            "text": text,
            "language": "ja_JP",
            "voice_name": self.voice_name,
            "destination": "audio_device",
            "force_enqueue": True,  # チャンク方式: 前の発話の後に続けて再生
        }
        if self.voice_version:
            payload["voice_version"] = self.voice_version

        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, auth=self.auth, timeout=30)
            resp.raise_for_status()
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("TTS synthesized in %.0f ms: '%s'", elapsed, text)
            return True
        except requests.RequestException as e:
            log.error("TTS failed: %s", e)
            return False

    def warmup(self) -> None:
        """VoiSona Talkをウォーム状態にする（初回合成の遅延を排除）。"""
        if not self.voice_name:
            return
        log.info("Warming up VoiSona TTS engine...")
        # 極短テキストを合成して内部キャッシュを温める
        self._synthesize("。")
        log.info("Warmup complete")

    def _tts_worker(self) -> None:
        """TTSキューからテキストを取り出して順番に合成するワーカースレッド。"""
        while True:
            text = self._tts_queue.get()
            if text is None:
                break  # 終了シグナル
            self._tts_busy.set()
            self._synthesize(text)
            self._tts_queue.task_done()
            # キューが空になったら busy を解除
            if self._tts_queue.empty():
                self._tts_busy.clear()

    def start_worker(self) -> None:
        """TTSワーカースレッドを開始する。"""
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def speak(self, text: str) -> None:
        """テキストをフレーズに分割し、TTSキューに投入する。

        最初のフレーズから順に合成が始まるため、
        長い文でも先頭部分がすぐ再生される。
        """
        if not self.voice_name:
            log.error("No voice selected")
            return

        phrases = split_phrases(text)
        if not phrases:
            phrases = [text]  # 分割できなかった場合はそのまま

        log.info("TTS queued %d phrase(s): %s", len(phrases), phrases)
        for phrase in phrases:
            self._tts_queue.put(phrase)

    def wait_until_done(self) -> None:
        """キュー内の全TTSが完了するまで待つ。"""
        self._tts_queue.join()

    @property
    def is_busy(self) -> bool:
        """TTSが再生中かどうか。"""
        return self._tts_busy.is_set() or not self._tts_queue.empty()

    def shutdown(self) -> None:
        """ワーカースレッドを終了する。"""
        self._tts_queue.put(None)
        if self._tts_thread:
            self._tts_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# STT（faster-whisper）
# ---------------------------------------------------------------------------
def create_whisper_model() -> WhisperModel:
    """faster-whisperモデルを読み込む。GPU失敗時はCPUにフォールバック。"""
    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        log.info(
            "Whisper model loaded: size=%s device=%s compute=%s",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        return model
    except Exception as e:
        log.warning("GPU init failed (%s), falling back to CPU", e)
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        log.info("Whisper model loaded on CPU (int8)")
        return model


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    """音声データからテキストを認識する。"""
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="ja",
        vad_filter=False,  # 自前VADを使うのでオフ
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        condition_on_previous_text=False,
    )
    # セグメントを収集し、低信頼度のものを除外
    collected = []
    for seg in segments:
        if seg.no_speech_prob > 0.5:
            log.debug(
                "Segment skipped (no_speech_prob=%.2f): '%s'",
                seg.no_speech_prob,
                seg.text,
            )
            continue
        collected.append(seg.text)

    text = "".join(collected).strip()

    # リピート除去: 同じ文字列が2回以上繰り返されている場合は1回にする
    if text:
        half = len(text) // 2
        if len(text) % 2 == 0 and text[:half] == text[half:]:
            log.info("Repetition filtered: '%s' -> '%s'", text, text[:half])
            text = text[:half]

    # ハルシネーション（幻聴）フィルタ
    if text in HALLUCINATION_PHRASES:
        log.info("Hallucination filtered: '%s'", text)
        return ""

    if text:
        log.info(
            "STT result (lang=%s prob=%.2f): '%s'",
            info.language,
            info.language_probability,
            text,
        )
    return text


# ---------------------------------------------------------------------------
# オーディオデバイス
# ---------------------------------------------------------------------------
def list_audio_devices():
    """利用可能なオーディオデバイスの一覧を表示する。"""
    print("\n利用可能なオーディオデバイス:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        direction = []
        if dev["max_input_channels"] > 0:
            direction.append("入力")
        if dev["max_output_channels"] > 0:
            direction.append("出力")
        default_in = i == sd.default.device[0]
        default_out = i == sd.default.device[1]
        markers = []
        if default_in:
            markers.append("既定入力")
        if default_out:
            markers.append("既定出力")
        marker_str = f"  ◀ {', '.join(markers)}" if markers else ""
        print(f"  [{i:2d}] {dev['name']}  ({'/'.join(direction)}){marker_str}")
    print(f"\n  .env に INPUT_DEVICE_INDEX=<番号> を設定するとマイクを指定できます")
    print(
        f"     Discordルーティング時は、VoiSona Talkの出力先をVB-CABLEに設定してください\n"
    )


# ---------------------------------------------------------------------------
# VAD + 録音ループ
# ---------------------------------------------------------------------------
def run_pipeline(
    model: WhisperModel,
    tts: VoiSonaTalk,
    input_device: int | None = None,
    silence_frames: int = 27,
    vad_aggressiveness: int = 2,
) -> None:
    """メインの録音→STT→TTS パイプラインを実行する。"""
    vad = webrtcvad.Vad(vad_aggressiveness)
    log.info("VAD aggressiveness=%d", vad_aggressiveness)

    audio_queue: queue.Queue[bytes] = queue.Queue()
    mic_muted = threading.Event()  # TTS再生中にマイクをミュートするフラグ

    # 使用するマイクデバイスをログに記録
    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        log.info("Using input device [%d]: %s", input_device, dev_info["name"])
    else:
        dev_info = sd.query_devices(sd.default.device[0])
        log.info("Using default input device: %s", dev_info["name"])

    def audio_callback(indata, frames, time_info, status):
        """sounddeviceのコールバック: 生の音声データをキューに投入。"""
        if status:
            log.warning("Audio input status: %s", status)
        if mic_muted.is_set():
            return  # TTS再生中はマイク入力を捨てる
        # int16に変換してバイト列にする
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        audio_queue.put(pcm)

    def drain_audio_queue():
        """キューに溜まった音声データをすべて捨てる。"""
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

    print("\nマイク入力を開始します。話しかけてください。(Ctrl+C で終了)\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=FRAME_SIZE,
        device=input_device,
        callback=audio_callback,
    ):
        speech_frames: list[bytes] = []
        silence_count = 0
        is_speaking = False
        speech_start_time = 0.0  # 遅延計測用

        try:
            while True:
                frame = audio_queue.get()

                # VADで音声/無音を判定
                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                if is_speech:
                    speech_frames.append(frame)
                    silence_count = 0
                    if not is_speaking:
                        is_speaking = True
                        speech_start_time = time.perf_counter()
                        log.info("Speech started")
                elif is_speaking:
                    # 喋り中に無音フレームが来た
                    speech_frames.append(frame)  # 無音部分も含めて保持
                    silence_count += 1

                    if silence_count >= silence_frames:
                        # 喋り終わり判定
                        is_speaking = False

                        if len(speech_frames) >= MIN_SPEECH_FRAMES:
                            vad_elapsed = (
                                time.perf_counter() - speech_start_time
                            ) * 1000
                            log.info(
                                "Speech ended (%d frames, ~%d ms)",
                                len(speech_frames),
                                len(speech_frames) * FRAME_DURATION_MS,
                            )

                            # バイト列を numpy に戻す
                            pcm_bytes = b"".join(speech_frames)
                            audio_data = (
                                np.frombuffer(pcm_bytes, dtype=np.int16).astype(
                                    np.float32
                                )
                                / 32767.0
                            )

                            # STT
                            t0 = time.perf_counter()
                            text = transcribe(model, audio_data)
                            stt_ms = (time.perf_counter() - t0) * 1000
                            log.info("STT took %.0f ms", stt_ms)

                            # TTS（フレーズ分割して非同期でキューに投入）
                            if text:
                                mic_muted.set()
                                drain_audio_queue()

                                tts.speak(text)  # キューに投入（非同期）
                                tts.wait_until_done()  # 全フレーズの合成完了を待つ

                                total_ms = (
                                    time.perf_counter() - speech_start_time
                                ) * 1000
                                log.info(
                                    "Total latency: %.0f ms (VAD=%.0f, STT=%.0f, TTS=%.0f)",
                                    total_ms,
                                    vad_elapsed,
                                    stt_ms,
                                    total_ms - vad_elapsed - stt_ms,
                                )

                                # 再生後の残響を拾わないよう少し待つ
                                time.sleep(0.3)
                                drain_audio_queue()
                                mic_muted.clear()
                            else:
                                log.info("No text recognized, skipping TTS")
                        else:
                            log.debug(
                                "Speech too short (%d frames), ignored",
                                len(speech_frames),
                            )

                        speech_frames.clear()
                        silence_count = 0

        except KeyboardInterrupt:
            print("\n\n終了します。")
            tts.shutdown()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="poitto - リアルタイム音声変換パイプライン"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="利用可能なオーディオデバイスの一覧を表示して終了",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="入力デバイスのインデックス番号（--list-devices で確認）",
    )
    args = parser.parse_args()

    # デバイス一覧表示モード
    if args.list_devices:
        load_dotenv()
        list_audio_devices()
        return

    # 設定の読み込み
    settings = load_settings()

    email = settings["email"]
    password = settings["password"]
    if not email or not password:
        log.error("VOISONA_EMAIL / VOISONA_API_PASSWORD not set in .env")
        sys.exit(1)

    silence_ms = settings["silence_ms"]
    silence_frames = settings["silence_frames"]
    log.info("SILENCE_MS=%d (%d frames)", silence_ms, silence_frames)

    print("=" * 50)
    print("  poitto - リアルタイム音声変換パイプライン")
    print("=" * 50)

    # VoiSona Talk 初期化
    tts = VoiSonaTalk(email, password)

    # ボイスの選択
    if not tts.select_voice(settings["voice_name"]):
        log.error("Cannot continue without a valid voice")
        sys.exit(1)

    # TTSワーカースレッドを起動
    tts.start_worker()

    # ウォームアップ（初回合成の遅延を排除）
    tts.warmup()

    # 入力デバイスの決定（コマンドライン引数 > .env > システム既定）
    input_device = args.input_device
    if input_device is None and settings["input_device_index"] is not None:
        input_device = int(settings["input_device_index"])

    # Whisperモデルの読み込み
    log.info("Loading Whisper model...")
    model = create_whisper_model()

    # パイプライン実行
    run_pipeline(
        model,
        tts,
        input_device=input_device,
        silence_frames=silence_frames,
        vad_aggressiveness=settings["vad_aggressiveness"],
    )


if __name__ == "__main__":
    main()
