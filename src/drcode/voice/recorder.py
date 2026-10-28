"""Microphone capture (push-to-talk) via sounddevice/PortAudio.

16 kHz mono PCM16 from the default input device into an in-memory buffer,
returned as a complete WAV file on stop.
"""

from __future__ import annotations

import io
import sys
import threading
import time
import wave
from array import array
from collections.abc import Callable
from typing import Protocol

from ._sounddevice import _PORTAUDIO_LOCK, SOUNDDEVICE_AVAILABLE
from .config import VoiceError


class _AudioStream(Protocol):
    """The subset of sounddevice.InputStream the recorder touches (tests
    fake the whole stream)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def _capture_problem(errors: int, overflows: int) -> str | None:
    """A readable one-line description of what the capture lost, or None
    when the recording is complete (both counters are zero)."""
    parts = []
    if overflows:
        parts.append(f"the input buffer overflowed {overflows} time(s)")
    if errors:
        parts.append(f"the recording callback failed {errors} time(s)")
    if not parts:
        return None
    return " and ".join(parts)


# The input stream mirrors the playback protection: a numeric latency with
# generous slack plus a fixed block size — the price is a deeper device
# ring buffer, which is exactly what absorbs GIL stalls on the audio thread.
_INPUT_LATENCY_FLOOR = 0.3
_INPUT_BLOCKSIZE = 2048


def _input_latency() -> float:
    """Numeric input latency with generous slack against callback delays
    (the recording counterpart of play_wav's ``_output_latency``). A Python
    callback runs on the PortAudio audio thread and must grab the GIL on
    every invocation; under CPU load (agents, tests) those grabs stall, and
    with the default short ring buffer PortAudio's only option is to DROP
    the frames — silent speech loss. Ask the device for its reported
    high-input latency, but never below :data:`_INPUT_LATENCY_FLOOR`; any
    query failure or nonsensical value degrades to the same floor."""
    try:
        import sounddevice as sd

        device = sd.default.device[0]
        high = float(sd.query_devices(device)["default_high_input_latency"])
    except Exception:  # noqa: BLE001 - a broken query must not kill recording
        return _INPUT_LATENCY_FLOOR
    if high <= 0:
        return _INPUT_LATENCY_FLOOR
    return max(high, _INPUT_LATENCY_FLOOR)


def _close_stream(stream: _AudioStream | None) -> None:
    """Stop and close a PortAudio stream under the module-level lock, from
    whatever thread owns the recorder at that moment (never the UI thread —
    the controller offloads cancel/stop to workers)."""
    if stream is None:
        return
    try:
        with _PORTAUDIO_LOCK:
            stream.stop()
            stream.close()
    except Exception as e:  # noqa: BLE001 - closing must never mask the result
        print(f"drcode voice: stream close failed: {e}", file=sys.stderr)


class MicrophoneRecorder:
    """Push-to-talk recorder: 16 kHz mono PCM16 from the default input
    device into memory, delivered as a complete WAV file on stop.

    The callback runs on PortAudio's audio thread — it only appends copies
    of the incoming chunks (list.append is atomic enough under the GIL);
    everything else happens on the caller's thread.
    """

    SAMPLE_RATE = 16_000

    # A zero-frame recording of a stream that ran at least this long is a
    # dead recording (the callback never delivered anything): stop() raises
    # instead of shipping a silent WAV to the ASR endpoint.
    MIN_AUDIBLE_DURATION = 0.5

    def __init__(self) -> None:
        self._chunks: list[array[int]] = []
        self._stream: _AudioStream | None = None
        self._lock = threading.Lock()
        # Failures of the audio-thread callback (sounddevice swallows
        # exceptions raised there — "Exception ignored from cffi callback"
        # — so they must be counted here to be visible at all).
        self._callback_errors = 0
        # Input-buffer overflows reported by PortAudio through the
        # callback's ``status`` argument (paInputOverflow on an input-only
        # stream): the device ring buffer ran dry because the audio thread
        # could not deliver frames in time (GIL stalls under CPU load) and
        # PortAudio DROPPED them — speech is silently missing from the
        # recording. Counted for the same reason as callback failures:
        # without a counter this loss is completely invisible.
        self._overflows = 0
        self._opened_at = time.monotonic()
        # Strong reference to the bound callback for as long as the stream
        # lives: sounddevice wraps it into a cffi trampoline, and letting the
        # wrapper be garbage-collected while PortAudio still calls it is a
        # crash (released only after the stream is fully closed).
        self._cb: Callable[..., object] | None = None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def overflows(self) -> int:
        """How many input-buffer overflows PortAudio reported so far
        (frames DROPPED — speech silently missing). Thread-safe: read from
        the UI thread while the audio thread increments."""
        with self._lock:
            return self._overflows

    @property
    def callback_errors(self) -> int:
        """How many times the audio-thread callback failed so far.
        Thread-safe: read from the UI thread."""
        with self._lock:
            return self._callback_errors

    def start(self) -> None:
        """Open the input stream. Raises VoiceError when the microphone
        cannot be opened (device busy/absent, PortAudio trouble)."""
        if not SOUNDDEVICE_AVAILABLE or self._stream is not None:
            return
        import sounddevice as sd

        self._cb = self._callback
        try:
            with _PORTAUDIO_LOCK:
                # RawInputStream (not InputStream): its callback hands over
                # a plain cffi buffer, so the audio path needs no numpy —
                # the regular InputStream hard-requires numpy to wrap the
                # callback data into ndarrays.
                stream: _AudioStream = sd.RawInputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    latency=_input_latency(),
                    # Explicit block size instead of the variable default
                    # (0), same protection as the playback stream: fewer
                    # callback invocations per second means fewer GIL
                    # acquisitions on the audio thread, and each call
                    # copies a larger chunk.
                    blocksize=_INPUT_BLOCKSIZE,
                    callback=self._cb,
                )
                with self._lock:
                    # Reset + publish BEFORE stream.start(): the callback
                    # may begin firing the moment start() runs (or even
                    # during it), and any chunk it delivers must land in
                    # the fresh buffer — frames between start() and the
                    # old post-start assignment were silently dropped.
                    self._chunks = []
                    self._callback_errors = 0
                    self._overflows = 0
                    self._opened_at = time.monotonic()
                    self._stream = stream
                stream.start()
        except Exception as e:
            # Never started (or start() itself failed): unpublish, so the
            # callback drops everything and `recording` stays honest.
            with self._lock:
                self._stream = None
            self._cb = None
            raise VoiceError(f"Microphone unavailable: {e}") from e

    def _callback(
        self, indata: object, frames: int, time: object, status: object
    ) -> None:
        # Audio-thread callback: a plain append of a copy. RawInputStream
        # hands over a byte-typed cffi buffer, which array.array("h", ...)
        # rejects outright (TypeError) — convert through bytes() first.
        # Cheap and non-blocking under the GIL — all assembly happens
        # later, on the caller's thread in stop().
        if status:
            # For an input-only stream a truthy status is paInputOverflow:
            # PortAudio dropped frames it could not deliver. Counted, never
            # printed per-occurrence (a GIL stall bursts dozens of these) —
            # the counter is surfaced by the recorder's properties and by
            # stop().
            with self._lock:
                self._overflows += 1
        try:
            chunk = array("h", bytes(indata))  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 - a dead callback is invisible otherwise
            with self._lock:
                self._callback_errors += 1
            print(f"drcode voice: microphone callback failed: {e}", file=sys.stderr)
            return
        with self._lock:
            if self._stream is not None:
                self._chunks.append(chunk)

    def stop(self) -> bytes:
        """Close the stream and return the recording as WAV bytes.

        A recording that ended up with zero frames despite the stream
        having been open longer than a moment (or whose callback failed)
        raises a distinct VoiceError instead — shipping a silent WAV to
        the ASR endpoint only surfaces later as the misleading
        "ASR returned no text". Same for a non-empty recording made of
        nothing but digital zeros: the device delivers frames but
        captures no sound (wrong input device selected, muted input) —
        that too is caught here. A genuinely brief tap still yields a
        valid, silent WAV.

        An INCOMPLETE recording — the callback failed partway through, or
        PortAudio reported input-buffer overflows (dropped frames) — also
        raises, even though the PCM is non-empty: a truncated utterance
        must never travel to the ASR endpoint silently (the user would
        read confidently wrong text with no hint that speech was lost).

        The stream is stopped BEFORE the chunk buffer is snapshotted:
        PortAudio guarantees no callbacks after stop() returns, so the
        final block (up to ``blocksize`` frames of tail speech) is
        captured too — the old clear-then-close order silently dropped
        it."""
        with self._lock:
            stream = self._stream
        _close_stream(stream)
        self._cb = None
        with self._lock:
            self._stream = None
            chunks = self._chunks
            self._chunks = []
            errors = self._callback_errors
            overflows = self._overflows
        problem = _capture_problem(errors, overflows)
        pcm = array("h")
        for chunk in chunks:
            pcm.extend(chunk)
        if not pcm:
            if problem:
                raise VoiceError(f"Microphone returned no audio: {problem}")
            duration = time.monotonic() - self._opened_at
            if stream is not None and duration >= self.MIN_AUDIBLE_DURATION:
                raise VoiceError(
                    f"Microphone returned no audio in {duration:.1f} s of recording"
                )
        elif problem:
            raise VoiceError(
                f"Recording incomplete: {problem} — parts of the speech "
                f"are missing; record the utterance again"
            )
        elif not any(pcm):
            duration = time.monotonic() - self._opened_at
            raise VoiceError(
                f"Microphone recorded only silence in {duration:.1f} s "
                f"(digital zeros) — check that the right input device is "
                f"selected and not muted"
            )
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def discard(self) -> None:
        """Drop the recording and close the stream (cancel path)."""
        with self._lock:
            stream, self._stream = self._stream, None
            self._chunks = []
        _close_stream(stream)
        self._cb = None
