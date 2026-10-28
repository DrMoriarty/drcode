"""The ASR/TTS HTTP client: an OpenAI-compatible audio API over httpx.

Synchronous, meant to be called from worker threads — never from the UI
event loop. ``POST /audio/transcriptions`` (multipart WAV → text),
``POST /audio/speech`` (text → WAV bytes) and a cheap ``GET /models``
availability probe.
"""

from __future__ import annotations

import io
import ssl
import wave

import httpx

from .config import VoiceConfig, VoiceError

# Network timeouts: transcription of a long utterance can take a while;
# the calls run in worker threads, so a generous ceiling costs nothing.
ASR_TIMEOUT = 60.0
TTS_TIMEOUT = 60.0
CHECK_TIMEOUT = 10.0


def _wav_upload_details(wav: bytes) -> str:
    """A compact human summary of the recorded upload — ``(12.3s, 456 KB)``
    or ``(34.2s, 1.1 MB)``. The duration comes from the WAV header (any
    parse failure degrades to 0.0s rather than masking the real error the
    message is attached to); the size is simply the payload length."""
    try:
        with wave.open(io.BytesIO(wav)) as handle:
            duration = handle.getnframes() / max(handle.getframerate(), 1)
    except Exception:  # noqa: BLE001 - a broken header must not mask the error
        duration = 0.0
    size = len(wav)
    if size >= 1_048_576:
        size_s = f"{size / 1_048_576:.1f} MB"
    else:
        size_s = f"{size / 1024:.0f} KB"
    return f"{duration:.1f}s, {size_s}"


class VoiceClient:
    """Synchronous OpenAI-compatible audio-API client (worker threads only)."""

    def __init__(
        self, config: VoiceConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        # `transport` is a test seam (httpx.MockTransport): production code
        # always leaves it None and gets a fresh real connection per call.
        self.config = config
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _require(self, model: str, what: str) -> None:
        if not model:
            raise VoiceError(f"Voice is not configured: no {what} model set")

    def _verify(self) -> ssl.SSLContext | bool:
        """Trust-store-backed TLS verification, falling back to httpx
        defaults. Системное хранилище сертификатов: корпоративные CA
        отсутствуют в certifi, из-за чего httpx даёт
        CERTIFICATE_VERIFY_FAILED, который оборачивается в ConnectError
        (тот же паттерн, что в ``agent_model._ssl_verify``)."""
        try:
            import truststore

            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            return True

    def _post(self, url: str, **kwargs: object) -> httpx.Response:
        if self._transport is not None:
            with httpx.Client(transport=self._transport) as client:
                return client.post(url, **kwargs)  # type: ignore[arg-type]
        return httpx.post(url, verify=self._verify(), **kwargs)  # type: ignore[arg-type]

    def _get(self, url: str, **kwargs: object) -> httpx.Response:
        if self._transport is not None:
            with httpx.Client(transport=self._transport) as client:
                return client.get(url, **kwargs)  # type: ignore[arg-type]
        return httpx.get(url, verify=self._verify(), **kwargs)  # type: ignore[arg-type]

    def transcribe(self, wav: bytes) -> str:
        """Speech → text (``POST /audio/transcriptions``): multipart WAV
        upload, the JSON answer carries the recognized text with
        punctuation and capitalization already in place.

        Every failure message carries the recording's duration and upload
        size ("(12.3s, 456 KB)"): the user can tell at a glance whether a
        full recording reached the endpoint or the request died with most
        of the speech never sent (transmission) — versus a capture-side
        loss, which the recorder reports separately."""
        self._require(self.config.asr_model, "ASR")
        details = _wav_upload_details(wav)
        try:
            response = self._post(
                f"{self.config._base()}/audio/transcriptions",
                headers=self._headers(),
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": self.config.asr_model},
                timeout=ASR_TIMEOUT,
            )
            response.raise_for_status()
            text = str(response.json().get("text") or "").strip()
        except httpx.HTTPStatusError as e:
            raise VoiceError(
                f"ASR failed: HTTP {e.response.status_code} ({details})"
            ) from e
        except (httpx.HTTPError, ValueError) as e:
            raise VoiceError(f"ASR failed: {type(e).__name__}: {e} ({details})") from e
        if not text:
            raise VoiceError(f"ASR returned no text ({details})")
        return text

    def speech(self, text: str) -> bytes:
        """Text → WAV bytes (``POST /audio/speech``)."""
        self._require(self.config.tts_model, "TTS")
        try:
            response = self._post(
                f"{self.config._base()}/audio/speech",
                headers=self._headers(),
                json={
                    "model": self.config.tts_model,
                    "input": text,
                    "voice": self.config.tts_voice,
                    "response_format": "wav",
                },
                timeout=TTS_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise VoiceError(f"TTS failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise VoiceError(f"TTS failed: {type(e).__name__}: {e}") from e
        return response.content

    def check(self) -> str:
        """Cheap availability probe (``GET /models``): raises VoiceError
        with a readable reason, returns a short human summary on success."""
        try:
            response = self._get(
                f"{self.config._base()}/models",
                headers=self._headers(),
                timeout=CHECK_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as e:
            raise VoiceError(f"HTTP {e.response.status_code}") from e
        except (httpx.HTTPError, ValueError) as e:
            raise VoiceError(f"{type(e).__name__}: {e}") from e
        models = payload.get("data") if isinstance(payload, dict) else None
        count = len(models) if isinstance(models, list) else 0
        if count:
            return f"OK — {count} models available"
        return "OK — endpoint answered"
