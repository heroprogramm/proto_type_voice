"""
SIP Voice Agent — Asterisk AudioSocket + Deepgram Agent API
============================================================

Zoiper/Phone → Asterisk → AudioSocket → Python Server → Deepgram Agent
Deepgram handles STT + LLM + TTS internally.
"""

import os
import json
import struct
import asyncio
import logging
import uuid
import audioop

import websockets
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────



AUDIOSOCKET_HOST = os.getenv("AUDIOSOCKET_HOST", "0.0.0.0")
AUDIOSOCKET_PORT = int(os.getenv("AUDIOSOCKET_PORT", 9092))

ASTERISK_SAMPLE_RATE = 8000
FRAME_SIZE = 320

DG_INPUT_RATE = 16000
DG_OUTPUT_RATE = 8000

DOCUMENT_TEXT = """
Our business hours are 9 AM to 6 PM Monday to Friday.
We offer appointment booking and customer support services.
Refund policy: Refunds are processed within 5 business days.
"""

AGENT_PROMPT = (
    "You are Sara, a friendly and warm phone booking agent. You sound natural, like a real person — not robotic.\n\n"
    "PERSONALITY:\n"
    "- Be conversational and warm. Use filler words occasionally like 'sure', 'absolutely', 'of course'.\n"
    "- Show empathy. If someone sounds unsure, reassure them.\n"
    "- Use short, natural sentences. Never give long responses.\n"
    "- Maximum 1-2 sentences per response.\n\n"
    "BOOKING WORKFLOW:\n"
    "If the user wants an appointment, collect these one at a time:\n"
    "1) full_name - Ask naturally like 'Can I get your name please?'\n"
    "2) appointment_date - Ask like 'What date works best for you?'\n"
    "3) appointment_time - Ask like 'And what time would you prefer?'\n\n"
    "RULES:\n"
    "- Ask ONE question at a time. Wait for the answer before asking the next.\n"
    "- If the user says a weekday like 'Tuesday', ask 'Do you mean this coming Tuesday?'\n"
    "- NEVER invent or assume dates.\n"
    "- Once you have all three, confirm naturally: 'Great! So I have you down for [date] at [time]. Is that correct?'\n"
    "- If user says something unrelated, gently guide back: 'I would be happy to help with that! Would you also like to book an appointment?'\n"
    "- If user says bye, respond warmly: 'Thanks for calling! Have a wonderful day.'\n\n"
    f"Company Info:\n{DOCUMENT_TEXT}"
)

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── AUDIOSOCKET MESSAGE TYPES ───────────────────────────────────────────────

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xFF

# ─── AUDIOSOCKET HELPERS ─────────────────────────────────────────────────────

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

# ─── HANDLE ONE CALL ─────────────────────────────────────────────────────────

async def handle_call(reader, writer):
    peer = writer.get_extra_info("peername")
    logger.info(f"New AudioSocket connection from {peer}")

    dg_ws = None
    call_uuid = None

    async def connect_deepgram():
        url = "wss://agent.deepgram.com/v1/agent/converse"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        try:
            ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=5,
                ping_timeout=20,
            )
        except TypeError:
            ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=5,
                ping_timeout=20,
            )

        settings = {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": DG_INPUT_RATE},
                "output": {
                    "encoding": "linear16",
                    "sample_rate": DG_OUTPUT_RATE,
                    "container": "none",
                },
            },
            "agent": {
                "language": "en",
                "listen": {"provider": {"type": "deepgram", "model": "nova-3"}},
                "think": {
                    "provider": {"type": "groq", "model": "llama-3.1-8b-instant"},
                    "endpoint": {
                        "url": "https://api.groq.com/openai/v1/chat/completions",
                        "headers": {"Authorization": f"Bearer {GROQ_API_KEY}"},
                    },
                    "prompt": AGENT_PROMPT,
                },
                "speak": {"provider": {"type": "deepgram", "model": "aura-2-asteria-en"}},
                "greeting": "Hi there! This is Sara. How can I help you today?",
            },
        }

        await ws.send(json.dumps(settings))
        logger.info("Deepgram Agent connected")
        return ws

    async def deepgram_to_asterisk():
        buffer = bytearray()

        try:
            async for msg in dg_ws:
                if isinstance(msg, bytes):
                    buffer.extend(msg)

                    while len(buffer) >= FRAME_SIZE:
                        chunk = bytes(buffer[:FRAME_SIZE])
                        del buffer[:FRAME_SIZE]

                        frame = build_audiosocket_message(MSG_AUDIO, chunk)
                        writer.write(frame)
                        await writer.drain()
                        await asyncio.sleep(0.01)

                    continue

                data = json.loads(msg)
                logger.info(f"Deepgram event: {json.dumps(data)}")

        except ConnectionClosed:
            logger.info("Deepgram connection closed")
        except Exception as e:
            logger.error(f"Deepgram receiver error: {e}")

    async def keepalive():
        try:
            while True:
                await asyncio.sleep(5)
                if dg_ws:
                    await dg_ws.send(json.dumps({"type": "KeepAlive"}))
        except Exception:
            pass

    dg_task = None
    ka_task = None

    try:
        dg_ws = await connect_deepgram()

        if not dg_ws:
            return

        dg_task = asyncio.create_task(deepgram_to_asterisk())
        ka_task = asyncio.create_task(keepalive())

        while True:
            msg_type, payload = await read_audiosocket_message(reader)

            if msg_type == MSG_UUID:
                call_uuid = str(uuid.UUID(bytes=payload[:16]))
                logger.info(f"Call UUID: {call_uuid}")

            elif msg_type == MSG_AUDIO:
                if dg_ws:
                    upsampled = audioop.ratecv(
                        payload, 2, 1, ASTERISK_SAMPLE_RATE, DG_INPUT_RATE, None
                    )[0]
                    await dg_ws.send(upsampled)

            elif msg_type == MSG_DTMF:
                digit = payload.decode("ascii") if payload else ""
                logger.info(f"DTMF: {digit}")

            elif msg_type == MSG_HANGUP:
                logger.info("Call hung up")
                break

            elif msg_type == MSG_ERROR:
                logger.error(f"AudioSocket error: {payload.hex()}")
                break

    except (asyncio.IncompleteReadError, ConnectionResetError):
        logger.info("AudioSocket connection closed")
    except Exception as e:
        logger.error(f"Call handler error: {e}")
    finally:
        if dg_ws:
            try:
                await dg_ws.close()
            except Exception:
                pass
        if dg_task:
            dg_task.cancel()
        if ka_task:
            ka_task.cancel()
        try:
            writer.close()
        except Exception:
            pass
        logger.info(f"Call session ended: {call_uuid or 'unknown'}")

# ─── TCP SERVER ──────────────────────────────────────────────────────────────

async def main():
    server = await asyncio.start_server(
        handle_call,
        AUDIOSOCKET_HOST,
        AUDIOSOCKET_PORT,
    )
    addr = server.sockets[0].getsockname()
    logger.info(f"AudioSocket server listening on {addr[0]}:{addr[1]}")
    logger.info("Waiting for Asterisk connections...")

    async with server:
        await server.serve_forever()

# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")