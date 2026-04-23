"""ASR client — Volcengine Bigmodel Streaming ASR v3 (true split mode).

Uses the standalone Volcengine streaming ASR product at
`wss://openspeech.bytedance.com/api/v3/sauc/bigmodel`. This product does ASR
*only* — no LLM, no TTS. AutoService CC is our brain; we just want transcripts
out of the user's mic audio.

Auth: X-Api-App-Key + X-Api-Access-Key + X-Api-Resource-Id=volc.bigasr.sauc.duration
(same DOUBAO_APP_ID / DOUBAO_ACCESS_TOKEN creds we already use for the E2E
dialogue product — these are account-level credentials).

Binary protocol is generated and parsed by the `volcengine-audio` SDK helpers;
we just drive the high-level flow (config → audio frames → transcript events).

Event stream yielded by receive():
  {"type": "input_audio_buffer.speech_started"}                                         - first partial of an utterance
  {"type": "conversation.item.input_audio_transcription.result",    "transcript": str}  - interim
  {"type": "conversation.item.input_audio_transcription.completed", "transcript": str}  - final
  {"type": "error", "error": {"message": str}}
"""
import logging
import os
import uuid
from typing import AsyncGenerator

import websockets

from volcengine_audio import (
    AudioCodec,
    STTAudioFormatV3,
    VolcengineAsrFunctionsV3,
    VolcengineAsrRequestV3,
)

log = logging.getLogger(__name__)

ASR_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
RESOURCE_ID = "volc.bigasr.sauc.duration"


class ASRClient:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self._connect_id = str(uuid.uuid4())
        self._ws = None
        self._seq = 0
        # Utterance index up to which we've already emitted 'completed' events.
        # Every time a new utterance in `result.utterances` becomes `definite: true`,
        # we emit a final for it once and advance this counter.
        self._finalized_utterances = 0
        # Track whether we've emitted speech_started for the current (not-yet-final)
        # utterance. Reset each time we emit a final.
        self._speech_started_sent = False

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

        # Retry the WS handshake on transient TLS/TCP resets. Volcengine's
        # edge occasionally drops a new connection mid-TLS; the same creds
        # and URL succeed 200-500ms later.
        import asyncio
        last_exc = None
        for attempt in range(3):
            try:
                self._ws = await websockets.connect(
                    ASR_URL,
                    additional_headers=headers,
                    ping_interval=None,
                    open_timeout=8,
                )
                break
            except (ConnectionResetError, OSError, TimeoutError) as e:
                last_exc = e
                log.warning("ASR connect attempt %d/3 failed: %s", attempt + 1, e)
                await asyncio.sleep(0.3 * (attempt + 1))
        else:
            raise RuntimeError(f"ASR upstream unreachable after 3 attempts: {last_exc}")

        # First frame: full client request with audio + request config.
        req = VolcengineAsrRequestV3(
            user=VolcengineAsrRequestV3.User(uid=self.session_id),
            audio=VolcengineAsrRequestV3.Audio(
                format=STTAudioFormatV3.pcm,
                codec=AudioCodec.raw,
                rate=16000,
                bits=16,
                channel=1,
            ),
            request=VolcengineAsrRequestV3.Request(
                model_name="bigmodel",
                enable_itn=True,
                enable_punc=True,
                show_utterances=True,  # needed for utterance-level 'definite' flag
                # Aggressive VAD for responsive turn-taking. Defaults are tuned
                # for long-form dictation (3000ms / 800ms) which feels sluggish
                # for conversational voice — users wait 3 seconds after they
                # stop talking before the system responds. Bring both down.
                vad_segment_duration=800,
                end_window_size=400,
            ),
        )
        self._seq = 1
        frame = VolcengineAsrFunctionsV3.generate_asr_full_client_request(
            sequence=self._seq,
            request_params=req.model_dump(exclude_none=True, mode="json"),
            compression=True,
        )
        await self._ws.send(bytes(frame))
        log.info("ASR connected (Volcengine bigmodel sauc, session=%s)", self.session_id[:8])

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if not self._ws:
            return
        self._seq += 1
        frame = VolcengineAsrFunctionsV3.generate_asr_audio_only_request(
            sequence=self._seq,
            audio=pcm_bytes,
            compress=True,
            keep_sequence=True,
        )
        await self._ws.send(bytes(frame))

    async def commit(self) -> None:
        # Bigmodel sauc ASR uses server-side VAD driven by vad_segment_duration
        # and end_window_size — no explicit commit needed mid-session. Kept as
        # a no-op for interface parity with the old Volcengine-AI-Gateway client.
        pass

    async def receive(self) -> AsyncGenerator[dict, None]:
        if not self._ws:
            return
        async for msg in self._ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            try:
                parsed = VolcengineAsrFunctionsV3.parse_response(bytes(msg))
            except Exception as e:
                yield {"type": "error", "error": {"message": f"parse error: {e}"}}
                continue

            # Error response (code != 0 typically means issue; sauc uses >1000 for errors)
            code = parsed.get("code")
            if code is not None and code != 0 and code < 1000:
                # Non-error ack — ignore.
                pass
            if code is not None and code >= 20000000:
                yield {
                    "type": "error",
                    "error": {"message": f"ASR server error code={code}: {parsed.get('message')}"},
                }
                continue

            message = parsed.get("message")
            if not isinstance(message, dict):
                continue

            results = message.get("result")
            if not results:
                continue
            # Bigmodel streaming emits either a single dict or a list with one entry
            res = results[0] if isinstance(results, list) else results
            if not isinstance(res, dict):
                continue

            text = res.get("text") or ""
            utterances = res.get("utterances") or []

            # 1. Emit finals for newly-definite utterances
            for i, u in enumerate(utterances):
                if u.get("definite") and i >= self._finalized_utterances:
                    utext = (u.get("text") or "").strip()
                    if utext:
                        yield {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "transcript": utext,
                        }
                    self._finalized_utterances = i + 1
                    self._speech_started_sent = False

            # 2. Emit speech_started + partial for the in-flight utterance (if any)
            pending_text = ""
            if utterances:
                # Take the first non-definite utterance after the finalized window
                for u in utterances[self._finalized_utterances:]:
                    if not u.get("definite"):
                        pending_text = (u.get("text") or "").strip()
                        break
            elif text:
                # No utterance breakdown — fall back to cumulative text
                pending_text = text.strip()

            if pending_text:
                if not self._speech_started_sent:
                    self._speech_started_sent = True
                    yield {"type": "input_audio_buffer.speech_started"}
                yield {
                    "type": "conversation.item.input_audio_transcription.result",
                    "transcript": pending_text,
                }

    async def close(self) -> None:
        if self._ws:
            try:
                # Negative-sequence empty audio frame signals end-of-stream.
                end_frame = VolcengineAsrFunctionsV3.generate_asr_audio_only_request(
                    sequence=self._seq,
                    audio=b"",
                    compress=False,
                    keep_sequence=False,
                )
                await self._ws.send(bytes(end_frame))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            log.info("ASR closed (session=%s)", self.session_id[:8])
