"""Fire-and-forget TTS announcements: the :class:`TtsSpeaker` daemon.

One daemon worker thread synthesizes and plays the queued phrases one at a
time (at most :data:`TtsSpeaker.MAX_PENDING` phrases wait in the queue — a
burst of pipeline events can never stack up minutes of speech). The phrase
following the one being played is synthesized in parallel on a short-lived
helper thread, so consecutive announcements follow each other without a
network-synthesis gap between them (the playback itself stays strictly
serialized).
"""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from .client import VoiceClient
from .config import VoiceConfig, config_from_settings
from .playback import play_wav

if TYPE_CHECKING:
    from ..settings import ProjectSettings


class _VoiceSynth(Protocol):
    """Structural subset of VoiceClient used by TtsSpeaker's factory seam
    (tests inject stubs with just this method)."""

    def speech(self, text: str) -> bytes: ...


class TtsSpeaker:
    """Fire-and-forget TTS announcer: one daemon worker thread synthesizes
    and plays the queued phrases strictly one at a time.

    The phrase following the one being played is synthesized in parallel on
    a short-lived helper thread, so consecutive announcements follow each
    other without a network-synthesis gap between them (the playback itself
    stays strictly serialized).

    ``speak`` never raises and never touches the network on the caller's
    thread; when more than MAX_PENDING phrases are already waiting, new ones
    are dropped (a burst of events must not stack up minutes of speech).
    A disabled/empty voice config in the project settings silences the
    speaker entirely.
    """

    MAX_PENDING = 2

    def __init__(
        self,
        settings_getter: Callable[[], ProjectSettings],
        *,
        on_error: Callable[[str], None] | None = None,
        client_factory: Callable[[VoiceConfig], _VoiceSynth] | None = None,
        play: Callable[[bytes], None] | None = None,
    ) -> None:
        # Test seams: `client_factory` builds the synthesizing client, `play`
        # consumes the WAV bytes. Defaults are the real VoiceClient + the
        # local sounddevice playback.
        self._settings_getter = settings_getter
        self._on_error = on_error
        self._client_factory = client_factory or VoiceClient
        self._play = play or play_wav
        self._queue: queue.Queue[tuple[str, VoiceConfig]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Phrases currently held out of the queue by the prefetch helper
        # (0 or 1): they are already being synthesized but sit in no queue,
        # so speak()'s cap must count them on top of qsize.
        self._prefetching = 0

    def speak(self, text: str) -> None:
        """Queue a phrase for synthesis+playback. Safe from any thread;
        silently no-op when voice is off/unconfigured."""
        text = text.strip()
        if not text:
            return
        try:
            config = config_from_settings(self._settings_getter())
        except Exception:  # noqa: BLE001 - settings must never break the caller
            return
        if config is None or not config.tts_model:
            return
        with self._lock:
            # The worker is persistent (blocks on the queue forever, exits
            # with the process as a daemon): starting it lazily avoids a
            # "thread exited between the alive-check and the put" race.
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker, daemon=True, name="drcode-tts"
                )
                self._thread.start()
            if self._queue.qsize() + self._prefetching >= self.MAX_PENDING:
                return
        self._queue.put((text, config))

    def _worker(self) -> None:
        # ready = the next phrase, already synthesized by the prefetch
        # helper while the previous one was playing.
        ready: tuple[bytes | None, str | None] | None = None
        while True:
            if ready is None:
                text, config = self._queue.get()
                wav, error = self._synthesize(text, config)
            else:
                wav, error = ready
                ready = None
            if wav is None:
                self._report_error(error or "TTS synthesis failed")
                continue
            finish = self._start_prefetch()
            try:
                self._play(wav)
            except Exception as e:  # noqa: BLE001 - announcements never break anything
                self._report_error(str(e))
            finally:
                if finish is not None:
                    ready = finish()

    def _synthesize(
        self, text: str, config: VoiceConfig
    ) -> tuple[bytes | None, str | None]:
        """Network synthesis, never raising: (wav, None) on success,
        (None, readable message) on failure."""
        try:
            return self._client_factory(config).speech(text), None
        except Exception as e:  # noqa: BLE001 - announcements never break anything
            return None, str(e)

    def _synthesize_into(
        self,
        text: str,
        config: VoiceConfig,
        out: list[tuple[bytes | None, str | None]],
    ) -> None:
        """Helper-thread body: synthesize one phrase into ``out``."""
        out.append(self._synthesize(text, config))

    def _start_prefetch(self) -> Callable[[], tuple[bytes | None, str | None]] | None:
        """Pop the next queued phrase and start synthesizing it on a helper
        thread while the current one plays. Returns a callable that joins
        the helper and yields its result (None when the queue was empty)."""
        try:
            text, config = self._queue.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._prefetching += 1
        box: list[tuple[bytes | None, str | None]] = []
        thread = threading.Thread(
            target=self._synthesize_into,
            args=(text, config, box),
            daemon=True,
            name="drcode-tts-prefetch",
        )
        thread.start()

        def finish() -> tuple[bytes | None, str | None]:
            thread.join()
            with self._lock:
                self._prefetching -= 1
            return box[0]

        return finish

    def _report_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception as e:  # noqa: BLE001 - the callback is best-effort
                print(f"drcode voice: error callback failed: {e}", file=sys.stderr)
