"""
SIP Voice Agent — Sara with ElevenLabs Human-Like Voice
========================================================

Asterisk AudioSocket → Deepgram STT → Groq LLM → ElevenLabs TTS
ElevenLabs gives much more natural, human-like voice than Deepgram TTS.
"""

import os
import json
import struct
import asyncio
import logging
import uuid
import audioop
import random

import httpx
import websockets
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────



# ElevenLabs voice - Sarah (natural female)
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
ELEVENLABS_MODEL = "eleven_turbo_v2"

AUDIOSOCKET_HOST = os.getenv("AUDIOSOCKET_HOST", "0.0.0.0")
AUDIOSOCKET_PORT = int(os.getenv("AUDIOSOCKET_PORT", 9092))

ASTERISK_SAMPLE_RATE = 8000
FRAME_SIZE = 320

# Deepgram STT
DG_STT_URL = "wss://api.deepgram.com/v1/listen"
DG_STT_PARAMS = (
    "?encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&model=nova-3"
    "&punctuate=true"
    "&interim_results=true"
    "&endpointing=300"
    "&utterance_end_ms=1000"
)

# Groq LLM
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DOCUMENT_TEXT = """
Our business hours are 9 AM to 6 PM Monday to Friday.
We offer appointment booking and customer support services.
Refund policy: Refunds are processed within 5 business days.
"""

SYSTEM_PROMPT = (
    "You are Sara, a friendly and warm phone booking agent. You sound natural, like a real person.\n\n"
    "PERSONALITY:\n"
    "- Be conversational and warm. Use words like 'sure', 'absolutely', 'of course'.\n"
    "- Show empathy. If someone sounds unsure, reassure them.\n"
    "- Use short, natural sentences. Maximum 1-2 sentences per response.\n"
    "- Never sound robotic or scripted.\n\n"
    "BOOKING WORKFLOW:\n"
    "Collect these one at a time:\n"
    "1) full_name - Ask naturally like 'Can I get your name please?'\n"
    "2) appointment_date - Ask like 'What date works best for you?'\n"
    "3) appointment_time - Ask like 'And what time would you prefer?'\n\n"
    "RULES:\n"
    "- Ask ONE question at a time.\n"
    "- If user says a weekday, ask 'Do you mean this coming Tuesday?'\n"
    "- NEVER invent dates.\n"
    "- Once done, confirm: 'Great! So I have you down for [date] at [time]. Is that correct?'\n"
    "- If user says bye: 'Thanks for calling! Have a wonderful day.'\n\n"
    f"Company Info:\n{DOCUMENT_TEXT}"
)

GREETING = "Hi there! This is Sara. How can I help you today?"

FILLER_PHRASES = ["Sure.", "Okay.", "Of course.", "Absolutely."]

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── AUDIOSOCKET ──────────────────────────────────────────────────────────────

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xFF


async def read_audiosocket_message(reader):
    header = await reader.readexactly(3)
    msg_type = header[0]
    payload_len = struct.unpack(">H", header[1:3])[0]
    payload = b""
    if payload_len > 0:
        payload = await reader.readexactly(payload_len)
    return msg_type, payload


def build_audiosocket_message(msg_type, payload=b""):
    header = struct.pack(">BH", msg_type, len(payload))
    return header + payload


async def send_audio(writer, audio_bytes):
    offset = 0
    while offset < len(audio_bytes):
        chunk = audio_bytes[offset: offset + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))
        frame = build_audiosocket_message(MSG_AUDIO, chunk)
        writer.write(frame)
        await writer.drain()
        offset += FRAME_SIZE
        await asyncio.sleep(0.005)


# ─── SESSION ──────────────────────────────────────────────────────────────────

class CallSession:
    def __init__(self, call_uuid):
        self.call_uuid = call_uuid
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.is_speaking = False

    def add_user(self, text):
        self.conversation.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.conversation.append({"role": "assistant", "content": text})


# ─── GROQ LLM ────────────────────────────────────────────────────────────────

async def get_llm_response(session):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": session.conversation,
        "max_tokens": 150,
        "temperature": 0.8,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(GROQ_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    session.add_assistant(text)
    logger.info(f"LLM -> {text}")
    return text


# ─── ELEVENLABS TTS ───────────────────────────────────────────────────────────

async def synthesize_speech(text):
    """Convert text to speech using ElevenLabs, return raw PCM 8kHz audio."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "output_format": "pcm_22050",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        pcm_22050 = resp.content

    # Downsample from 22050Hz to 8000Hz for Asterisk
    pcm_8000 = audioop.ratecv(pcm_22050, 2, 1, 22050, 8000, None)[0]
    logger.info(f"TTS: {len(pcm_8000)} bytes audio")
    return pcm_8000


# ─── HANDLE ONE CALL ─────────────────────────────────────────────────────────

async def handle_call(reader, writer):
    peer = writer.get_extra_info("peername")
    logger.info(f"New AudioSocket connection from {peer}")

    session = None
    dg_ws = None
    dg_task = None
    orch_task = None
    transcript_queue = asyncio.Queue()

    async def connect_deepgram():
        url = f"{DG_STT_URL}{DG_STT_PARAMS}"
        try:
            ws = await websockets.connect(url, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"})
        except TypeError:
            ws = await websockets.connect(url, extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"})
        logger.info("Deepgram STT connected")
        return ws

    async def deepgram_listener():
        try:
            async for msg in dg_ws:
                data = json.loads(msg)
                if data.get("type") == "Results":
                    is_final = data.get("is_final", False)
                    transcript = (
                        data.get("channel", {})
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                        .strip()
                    )
                    if transcript and is_final:
                        logger.info(f'STT -> "{transcript}"')
                        await transcript_queue.put(transcript)
        except ConnectionClosed:
            logger.info("Deepgram STT closed")
        except Exception as e:
            logger.error(f"STT error: {e}")

    async def orchestrator():
        # Send greeting
        try:
            logger.info(f"Greeting: {GREETING}")
            session.add_assistant(GREETING)
            audio = await synthesize_speech(GREETING)
            session.is_speaking = True
            await send_audio(writer, audio)
            session.is_speaking = False
        except Exception as e:
            logger.error(f"Greeting error: {e}")

        while True:
            text = await transcript_queue.get()
            await asyncio.sleep(0.1)
            while not transcript_queue.empty():
                text += " " + transcript_queue.get_nowait()

            if not text.strip():
                continue

            logger.info(f'User: "{text}"')
            session.add_user(text)

            try:
                # Say filler and get LLM response in parallel
                filler = random.choice(FILLER_PHRASES)
                filler_task = asyncio.create_task(synthesize_speech(filler))
                llm_task = asyncio.create_task(get_llm_response(session))

                # Play filler while LLM thinks
                filler_audio = await filler_task
                await send_audio(writer, filler_audio)

                # Get LLM response
                reply = await llm_task

                # Speak response
                audio = await synthesize_speech(reply)
                session.is_speaking = True
                await send_audio(writer, audio)
                session.is_speaking = False

            except Exception as e:
                logger.error(f"Pipeline error: {e}")

    try:
        dg_ws = await connect_deepgram()
        dg_task = asyncio.create_task(deepgram_listener())

        while True:
            msg_type, payload = await read_audiosocket_message(reader)

            if msg_type == MSG_UUID:
                call_uuid = str(uuid.UUID(bytes=payload[:16]))
                session = CallSession(call_uuid)
                logger.info(f"Call UUID: {call_uuid}")
                orch_task = asyncio.create_task(orchestrator())

            elif msg_type == MSG_AUDIO:
                if dg_ws:
                    upsampled = audioop.ratecv(payload, 2, 1, ASTERISK_SAMPLE_RATE, 16000, None)[0]
                    await dg_ws.send(upsampled)

            elif msg_type == MSG_DTMF:
                logger.info(f"DTMF: {payload.decode('ascii') if payload else ''}")

            elif msg_type == MSG_HANGUP:
                logger.info("Call hung up")
                break

            elif msg_type == MSG_ERROR:
                logger.error("AudioSocket error")
                break

    except (asyncio.IncompleteReadError, ConnectionResetError):
        logger.info("AudioSocket closed")
    except Exception as e:
        logger.error(f"Call error: {e}")
    finally:
        if dg_ws:
            try:
                await dg_ws.close()
            except Exception:
                pass
        if dg_task:
            dg_task.cancel()
        if orch_task:
            orch_task.cancel()
        try:
            writer.close()
        except Exception:
            pass
        logger.info(f"Call ended: {session.call_uuid if session else 'unknown'}")


# ─── SERVER ───────────────────────────────────────────────────────────────────

async def main():
    server = await asyncio.start_server(handle_call, AUDIOSOCKET_HOST, AUDIOSOCKET_PORT)
    addr = server.sockets[0].getsockname()
    logger.info(f"AudioSocket server listening on {addr[0]}:{addr[1]}")
    logger.info("Waiting for Asterisk connections...")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")