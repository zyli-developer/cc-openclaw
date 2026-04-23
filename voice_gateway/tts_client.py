"""TTS client — Volcengine Bigmodel Bidirectional Streaming TTS v3 (true split mode).

Uses the standalone Volcengine TTS product at
`wss://openspeech.bytedance.com/api/v3/tts/bidirection`. This product does TTS
*only* — no LLM, no ASR. Each ASR-turn / CC-reply text gets synthesized here.

Key property: one WebSocket connection handles *multiple* synthesize() calls —
no per-utterance reconnect like the E2E-dialogue workaround we had before.
Comfort text + CC reply + subsequent turns all share the same session.

Auth: X-Api-App-Key + X-Api-Access-Key + X-Api-Resource-Id=seed-tts-1.0 (uses
the same DOUBAO_APP_ID / DOUBAO_ACCESS_TOKEN account credentials).

Binary protocol generated and parsed by the `volcengine-audio` SDK helpers.
"""
import asyncio
import logging
import os
import uuid
from typing import AsyncGenerator

import websockets

from volcengine_audio import (
    EventReceive,
    EventSend,
    MessageType,
    VolcengineTTSFunctions,
)

log = logging.getLogger(__name__)

TTS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
# Resource_id is bound to the TTS instance opened on Volcengine. seed-tts-2.0
# uses model "seed-tts-2.0-standard" (or "-expressive") and a distinct speaker
# catalog (*_uranus_bigtts / saturn_*_tob / etc.) — see your console's
# "音色详情" for the exact names your instance provides.
RESOURCE_ID = os.environ.get("DOUBAO_TTS_RESOURCE_ID", "seed-tts-2.0")
DEFAULT_MODEL = os.environ.get("DOUBAO_TTS_MODEL", "seed-tts-2.0-standard")
DEFAULT_SPEAKER = os.environ.get("DOUBAO_TTS_SPEAKER", "zh_female_vv_uranus_bigtts")
DEFAULT_SAMPLE_RATE = 24000


class TTSClient:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self._connect_id = str(uuid.uuid4())
        self._ws = None
        self._connection_started = False
        self._session_started = False

    async def connect(self) -> None:
        app_id = os.environ.get("DOUBAO_APP_ID", "")
        token = os.environ.get("DOUBAO_ACCESS_TOKEN", "")
        if not app_id or not token:
            raise RuntimeError(
                "DOUBAO_APP_ID / DOUBAO_ACCESS_TOKEN missing from environment"
            )

        headers = {
            "X-Api-App-Key": app_id,
            "X-Api-Access-Key": token,
            "X-Api-Resource-Id": RESOURCE_ID,
            "X-Api-Connect-Id": self._connect_id,
        }
        self._ws = await websockets.connect(
            TTS_URL,
            additional_headers=headers,
            ping_interval=None,
        )

        # StartConnection → wait ConnectionStarted
        await self._ws.send(VolcengineTTSFunctions.start_connection_payload())
        await self._expect_event(EventReceive.ConnectionStarted, timeout=5.0)
        self._connection_started = True

        # StartSession req_params — speaker must match the voice catalog
        # enabled on the TTS instance (see console "音色详情" tab).
        start_req_params = {
            "model": DEFAULT_MODEL,
            "speaker": DEFAULT_SPEAKER,
            "audio_params": {
                "format": "pcm",
                "sample_rate": DEFAULT_SAMPLE_RATE,
            },
        }
        await self._ws.send(
            VolcengineTTSFunctions.start_session_payload(
                session_id=self.session_id,
                req_params=start_req_params,
                user_info={"uid": self.session_id},
            )
        )
        server_sid, _ = await self._expect_event(EventReceive.SessionStarted, timeout=5.0)
        # Server may re-assign the session_id — use its value going forward.
        if server_sid:
            self.session_id = server_sid
        self._session_started = True
        log.info("TTS connected (Volcengine bigmodel bidirection, server session=%s)", self.session_id[:12])

    async def _expect_event(self, target: EventReceive, timeout: float = 5.0):
        """Drain frames until we see `target` or hit timeout/error.

        Returns (session_id_from_server, payload) — the server may re-assign
        session_id at ConnectionStarted/SessionStarted; subsequent frames must
        use the server-issued value.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {target.name}")
            msg = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            if not isinstance(msg, (bytes, bytearray)):
                continue
            event, sid, payload = VolcengineTTSFunctions.extract_response_payload(bytes(msg))
            log.debug("TTS recv event=%s sid=%s payload=%s",
                      event.name if hasattr(event, 'name') else event,
                      sid, str(payload)[:120])
            if event == target:
                return sid, payload
            if event in (
                EventReceive.ConnectionFailed,
                EventReceive.SessionFailed,
                EventReceive.DialogCommonError,
                EventReceive.INVALID_MODEL,
                EventReceive.SERVER_PROCESSING_ERROR,
                EventReceive.SERVICE_UNAVAILABLE,
                EventReceive.AUDIO_FLOW_ERROR,
            ):
                raise RuntimeError(f"TTS upstream error event={event.name} payload={payload}")
            # Other events (e.g. USAGE) — ignore

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """Ask Doubao to synthesize `text`. Yield PCM chunks as they stream back.

        Works for arbitrary number of calls on the same WebSocket connection —
        there's no per-utterance reconnect, which is why v3 bidirection beats
        the E2E-dialogue workaround.
        """
        if not self._ws or not self._session_started:
            raise RuntimeError("TTSClient not connected")

        audio_params = {
            "format": "pcm",
            "sample_rate": DEFAULT_SAMPLE_RATE,
        }
        frame = VolcengineTTSFunctions.task_request_payload(
            session_id=self.session_id,
            text=text,
            speaker=DEFAULT_SPEAKER,
            audio_params=audio_params,
        )
        await self._ws.send(bytes(frame))

        log.debug("TTS sent TaskRequest: text=%r speaker=%s", text[:60], DEFAULT_SPEAKER)

        # The v3 bidirection product is designed for continuous use — the
        # server emits audio chunks but does NOT fire a per-utterance end
        # marker (TTSSentenceEnd/TTSEnded only fire at FinishSession time on
        # many configs). So we detect per-text completion heuristically:
        # after the first audio chunk, any silence ≥ SILENCE_TIMEOUT means
        # "this synthesis is done, ready for the next TaskRequest".
        FIRST_CHUNK_TIMEOUT = 15.0   # Generous: Doubao can take a few seconds to start
        SILENCE_TIMEOUT = 0.8        # After first audio, short silence = done
        audio_bytes = 0
        got_first_audio = False

        while True:
            try:
                timeout = SILENCE_TIMEOUT if got_first_audio else FIRST_CHUNK_TIMEOUT
                msg = await asyncio.wait_for(self._ws.recv(), timeout)
            except asyncio.TimeoutError:
                if got_first_audio:
                    log.info("TTS synthesize done via silence (%d bytes)", audio_bytes)
                    return
                raise RuntimeError(
                    f"TTS timed out waiting for first audio chunk ({FIRST_CHUNK_TIMEOUT}s)"
                )

            if not isinstance(msg, (bytes, bytearray)):
                continue
            event, _sid, payload = VolcengineTTSFunctions.extract_response_payload(bytes(msg))
            log.debug("TTS event=%s payload_len=%s",
                      event.name if hasattr(event, 'name') else event,
                      len(payload) if isinstance(payload, (bytes, bytearray, str, dict, list)) else '?')

            if event == EventReceive.TTSResponse:
                if isinstance(payload, (bytes, bytearray)) and payload:
                    audio_bytes += len(payload)
                    got_first_audio = True
                    yield bytes(payload)

            elif event in (EventReceive.TTSSentenceEnd, EventReceive.TTSEnded):
                # Explicit end — great, use it.
                log.info("TTS synthesize done via %s (%d bytes)", event.name, audio_bytes)
                return

            elif event in (
                EventReceive.SessionFailed,
                EventReceive.SessionCanceled,
                EventReceive.DialogCommonError,
                EventReceive.SERVER_PROCESSING_ERROR,
                EventReceive.SERVICE_UNAVAILABLE,
                EventReceive.AUDIO_FLOW_ERROR,
            ):
                raise RuntimeError(f"TTS synthesize error event={event.name} payload={payload}")

            # TTSSentenceStart / TTSSubtitle / USAGE — ignore

    async def close(self) -> None:
        if not self._ws:
            return
        try:
            if self._session_started:
                await self._ws.send(
                    VolcengineTTSFunctions.finish_session_payload(self.session_id)
                )
        except Exception:
            pass
        try:
            if self._connection_started:
                await self._ws.send(VolcengineTTSFunctions.finish_connection_payload())
        except Exception:
            pass
        try:
            await self._ws.close()
        except Exception:
            pass
        self._ws = None
        log.info("TTS closed (session=%s)", self.session_id[:8])
