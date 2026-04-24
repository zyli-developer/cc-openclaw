"""Benchmark TTS first-byte latency across speakers, models, and endpoints.

Not part of the gateway — standalone diagnostic script. Loads creds from
voice-web/.env.local, hits Volcengine directly (bypassing our route), and
reports:

  - TaskRequest sent → TTSSentenceStart  (server planning/preprocessing)
  - TTSSentenceStart → first TTSResponse (first audio byte)
  - Total time to silence (fully synthesized)
  - Total audio bytes

Run: ./.venv/Scripts/python.exe voice_gateway/bench_tts.py
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

import websockets

from volcengine_audio import EventReceive, VolcengineTTSFunctions


def load_env():
    root = os.path.dirname(__file__)
    for path in [
        os.path.join(root, "..", "voice-web", ".env.local"),
        os.path.join(root, "..", ".env.local"),
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
            return


async def bench_one(
    label: str,
    url: str,
    resource_id: str,
    model: str,
    speaker: str,
    text: str = "你好，这是一次延迟测试",
) -> dict:
    """Run one synthesis against a specific endpoint+config and return timings."""
    app_id = os.environ["DOUBAO_APP_ID"]
    token = os.environ["DOUBAO_ACCESS_TOKEN"]
    headers = {
        "X-Api-App-Key": app_id,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    result: dict = {"label": label, "speaker": speaker, "model": model}

    try:
        async with websockets.connect(url, additional_headers=headers, open_timeout=10) as ws:
            # StartConnection
            await ws.send(VolcengineTTSFunctions.start_connection_payload())
            msg = await asyncio.wait_for(ws.recv(), 5)
            ev, _, _ = VolcengineTTSFunctions.extract_response_payload(bytes(msg))
            if ev != EventReceive.ConnectionStarted:
                result["error"] = f"conn_failed ev={ev}"
                return result

            # StartSession
            sid = str(uuid.uuid4())
            start_params = {
                "model": model,
                "speaker": speaker,
                "audio_params": {"format": "pcm", "sample_rate": 24000},
            }
            await ws.send(
                VolcengineTTSFunctions.start_session_payload(sid, start_params, {"uid": sid})
            )
            msg = await asyncio.wait_for(ws.recv(), 5)
            ev, sid2, _ = VolcengineTTSFunctions.extract_response_payload(bytes(msg))
            if ev != EventReceive.SessionStarted:
                result["error"] = f"sess_failed ev={ev}"
                return result
            sid = sid2 or sid

            # TaskRequest — START TIMING
            t0 = time.perf_counter()
            frame = VolcengineTTSFunctions.task_request_payload(
                sid, text, speaker,
                {"format": "pcm", "sample_rate": 24000},
            )
            await ws.send(frame)

            t_sent_sent_start = t_first_audio = None
            audio_bytes = 0
            last_audio_time = None

            while True:
                try:
                    # Longer first-chunk timeout, then short silence-based done detection
                    timeout = 0.6 if audio_bytes > 0 else 20.0
                    msg = await asyncio.wait_for(ws.recv(), timeout)
                except asyncio.TimeoutError:
                    if audio_bytes > 0:
                        t_done = time.perf_counter()
                        break
                    result["error"] = "first_chunk_timeout"
                    return result

                if not isinstance(msg, (bytes, bytearray)):
                    continue
                ev, _, payload = VolcengineTTSFunctions.extract_response_payload(bytes(msg))

                if ev == EventReceive.TTSSentenceStart and t_sent_sent_start is None:
                    t_sent_sent_start = time.perf_counter()
                elif ev == EventReceive.TTSResponse and isinstance(payload, (bytes, bytearray)):
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                    audio_bytes += len(payload)
                    last_audio_time = time.perf_counter()
                elif ev in (EventReceive.TTSSentenceEnd, EventReceive.TTSEnded):
                    t_done = time.perf_counter()
                    break
                elif hasattr(ev, "name") and "ERROR" in ev.name:
                    result["error"] = f"upstream {ev.name}"
                    return result

            result["to_sent_start_s"] = (t_sent_sent_start - t0) if t_sent_sent_start else None
            result["to_first_audio_s"] = (t_first_audio - t0) if t_first_audio else None
            result["to_done_s"] = t_done - t0
            result["audio_bytes"] = audio_bytes
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:100]}"

    return result


async def main():
    load_env()
    TEXT = "你好，这是一次延迟测试"

    cases = [
        # Label, url, resource_id, model, speaker
        ("bidir + vv_uranus + standard",
         "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
         "seed-tts-2.0", "seed-tts-2.0-standard",
         "zh_female_vv_uranus_bigtts"),

        ("bidir + vv_uranus + expressive",
         "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
         "seed-tts-2.0", "seed-tts-2.0-expressive",
         "zh_female_vv_uranus_bigtts"),

        ("bidir + xiaohe + standard",
         "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
         "seed-tts-2.0", "seed-tts-2.0-standard",
         "zh_female_xiaohe_uranus_bigtts"),

        ("bidir + cancan + standard",
         "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
         "seed-tts-2.0", "seed-tts-2.0-standard",
         "saturn_zh_female_cancan_tob"),

        ("unidir + vv_uranus + standard",
         "wss://openspeech.bytedance.com/api/v3/tts/unidirectional",
         "seed-tts-2.0", "seed-tts-2.0-standard",
         "zh_female_vv_uranus_bigtts"),

        ("unidir-stream + vv_uranus + standard",
         "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream",
         "seed-tts-2.0", "seed-tts-2.0-standard",
         "zh_female_vv_uranus_bigtts"),
    ]

    print(f"{'label':45s} | {'sent_start':11s} | {'first_audio':11s} | {'done':9s} | {'bytes':9s} | notes")
    print("-" * 110)
    for label, url, rid, model, sp in cases:
        r = await bench_one(label, url, rid, model, sp, TEXT)
        if "error" in r:
            print(f"{label:45s} | {'':11s} | {'':11s} | {'':9s} | {'':9s} | ERR {r['error']}")
        else:
            ss = f"{r['to_sent_start_s']:.2f}s" if r['to_sent_start_s'] else "-"
            fa = f"{r['to_first_audio_s']:.2f}s" if r['to_first_audio_s'] else "-"
            d = f"{r['to_done_s']:.2f}s"
            b = f"{r['audio_bytes']/1024:.0f}K"
            print(f"{label:45s} | {ss:11s} | {fa:11s} | {d:9s} | {b:9s} |")
        # Brief pause between tests to avoid rate limiting
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
