"""Synchronous WAV playback via sounddevice/PortAudio (worker threads only).

numpy-free: the PCM is decoded with the stdlib ``wave`` module and streamed
through a RawOutputStream callback, since sd.play()/sd.wait() hard-require
numpy, which the voice feature deliberately avoids.
"""

from __future__ import annotations

import io
import sys
import threading
import wave
from array import array

from ._sounddevice import _PORTAUDIO_LOCK, SOUNDDEVICE_AVAILABLE, VOICE_UNAVAILABLE_HINT
from .config import VoiceError

# Click-suppression constants for play_wav: short linear ramps at the
# audible edges of the phrase plus a tail of appended silence.
_PLAYBACK_FADE_IN_S = 0.008
_PLAYBACK_FADE_OUT_S = 0.030
_PLAYBACK_TAIL_S = 0.120


def _fade_edges(pcm: array[int], rate: int, channels: int) -> None:
    """Linear fade-in/out on an interleaved int16 PCM ``array`` (in place).

    Gain is computed per frame and applied to every channel of the frame,
    so an interleaved stereo phrase keeps its image. Both ramps start and
    end at zero gain: the phrase no longer jumps out of silence at the
    start nor into the zero padding of the final block at the end — either
    jump within one sample is an audible click."""
    ch = max(channels, 1)
    total_frames = len(pcm) // ch
    fade_in = min(total_frames, int(rate * _PLAYBACK_FADE_IN_S))
    fade_out = min(total_frames, int(rate * _PLAYBACK_FADE_OUT_S))
    for i in range(fade_in):
        gain = i / fade_in
        for c in range(ch):
            pcm[i * ch + c] = int(pcm[i * ch + c] * gain)
    for i in range(fade_out):
        gain = (fade_out - 1 - i) / fade_out
        base = (total_frames - fade_out + i) * ch
        for c in range(ch):
            pcm[base + c] = int(pcm[base + c] * gain)


def play_wav(data: bytes) -> None:
    """Decode PCM WAV bytes and play them synchronously (blocks until the
    sound finishes). Worker threads only."""
    if not SOUNDDEVICE_AVAILABLE:
        raise VoiceError(f"sounddevice is not available ({VOICE_UNAVAILABLE_HINT})")
    try:
        with wave.open(io.BytesIO(data)) as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise VoiceError(f"TTS answer is not a valid WAV: {e}") from e
    if sampwidth != 2:
        # play_wav interprets the frames as native int16 (array("h")); a
        # 24-bit or float32 answer from the gateway would be played as
        # shifted garbage (crackle over the whole phrase) — fail readable
        # instead.
        raise VoiceError(
            f"TTS answer is {sampwidth * 8}-bit PCM; only 16-bit is supported"
        )
    import sounddevice as sd

    # int16 framing: building array.array("h") from the raw frames doubles
    # as the odd-byte guard — a truncated final sample raises ValueError
    # (surfaced as a readable VoiceError) instead of corrupting playback.
    # This replaces np.frombuffer; sd.play()/sd.wait() hard-require numpy,
    # which the voice extra deliberately avoids.
    try:
        pcm = array("h", frames)
    except ValueError as e:
        raise VoiceError(f"TTS answer has a truncated PCM frame: {e}") from e
    if not pcm:
        return

    # Click suppression, applied to the decoded PCM before playback:
    # TTS answers rarely end at (or start from) zero amplitude, so first
    # fade the audible edges in-place ...
    _fade_edges(pcm, rate, channels)
    # ... then append tail silence AFTER the fade-out — any stop()/close
    # transient now happens once the audible content has already decayed
    # (many TTS backends also ship an unzeroed tail; the fade covers it).
    pcm.frombytes(bytes(2 * channels * int(rate * _PLAYBACK_TAIL_S)))

    raw = pcm.tobytes()
    total = len(raw)
    state = {"pos": 0, "underflows": 0, "filled": False}
    done = threading.Event()

    # Callback mode, not the blocking RawOutputStream.write() pair: with
    # blocking writes the ring buffer is empty when the stream starts, so
    # the first frames go out as zeros (an audible click at the head of
    # every phrase) and any writer-thread hiccup mid-phrase drops out.
    # (Priming via Pa_WriteStream before start() is NOT an option — on
    # current PortAudio it fails with paStreamIsStopped.) The callback
    # fills each block straight from memory on demand — nothing can run
    # dry. Replaces sd.play()/sd.wait(), which hard-require numpy.

    def _callback(outdata: object, frames: int, time: object, status: object) -> None:
        try:
            if status:
                # Count every callback that reports a flag (for an
                # output-only stream that is paOutputUnderflow) — a counter
                # shows how often the device ran dry, a boolean hides it.
                state["underflows"] += 1
            view = memoryview(outdata).cast("B")  # type: ignore[arg-type]
            end = min(state["pos"] + len(view), total)
            block = end - state["pos"]
            view[:block] = raw[state["pos"] : end]
            if block < len(view):
                # The final partial block is padded with silence — the
                # phrase's natural tail.
                view[block:] = bytes(len(view) - block)
            state["pos"] = end
            if end >= total:
                # Race guard: `done` must NOT be raised by the callback that
                # fills the last block — the main thread would call
                # stream.stop() while this callback is still running, and on
                # ALSA / partially CoreAudio that truncates the final block
                # (the phrase tail ends in a click). Raise it one callback
                # LATER, when the last block is already in the device.
                if state["filled"]:
                    done.set()
                else:
                    state["filled"] = True
        except Exception as e:  # noqa: BLE001 - a dead callback would hang play_wav
            print(f"drcode voice: playback callback failed: {e}", file=sys.stderr)
            done.set()

    def _output_latency() -> float:
        """Numeric output latency with generous slack against callback
        delays. A Python callback runs on the PortAudio audio thread and
        must grab the GIL on every invocation; under CPU load those grabs
        stall, so the real protection is a deep device ring buffer —
        "high" (~40–100 ms on most backends) is not enough. Ask the device
        for its reported high-output latency, but never below 0.3 s; any
        query failure or nonsensical value degrades to the same 0.3 s.
        The price — up to 300 ms before the phrase starts — is irrelevant
        for notifications, and the stop()-drain plus the appended tail
        silence keep the ending click-free."""
        fallback = 0.3
        try:
            device = sd.default.device[1]
            high = float(sd.query_devices(device)["default_high_output_latency"])
        except Exception:  # noqa: BLE001 - a broken query must not kill playback
            return fallback
        if high <= 0:
            return fallback
        return max(high, fallback)

    # The whole playback (including the drain in stop()) stays under the
    # lock so a concurrent recording open/close can never interleave with it.
    with _PORTAUDIO_LOCK:
        latency = _output_latency()
        stream = sd.RawOutputStream(
            samplerate=rate,
            channels=channels,
            dtype="int16",
            latency=latency,
            # Explicit block size instead of the variable default (0): fewer
            # callback invocations per second means fewer GIL acquisitions
            # on the audio thread, and each call copies a larger chunk.
            blocksize=2048,
            callback=_callback,
        )
        try:
            stream.start()
            # `done` fires one callback after the last block (see the race
            # guard above), so the wait normally ends ~one block late —
            # before stop() touches anything. The timeout is only a hang
            # guard for a device that stops calling the callback.
            duration = len(pcm) / max(channels, 1) / rate
            if not done.wait(timeout=duration + 30.0):
                print("drcode voice: playback timed out", file=sys.stderr)
        finally:
            stream.stop()  # drains the pending audio, like sd.wait()
            stream.close()
    if state["underflows"]:
        print(
            f"drcode voice: playback buffer underflowed "
            f"{state['underflows']} times (latency={latency}s)",
            file=sys.stderr,
        )
