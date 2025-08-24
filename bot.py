# Copyright (c) 2024–2025, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Pipecat Voice AI Bot — export raw transcript only on disconnect.

Realtime loop: STT -> LLM -> TTS.
On disconnect: POST the raw context (all messages) to n8n.
"""

import os
import aiohttp

import requests
from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (first run may take ~20s)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import DailyParams

from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.services.heygen.api import NewSessionRequest


logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# ------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

def post_to_n8n(transcript):
    """Send raw transcript to n8n webhook."""
    r = requests.post(N8N_WEBHOOK_URL, json={"transcript": transcript})
    if r.status_code == 200:
        logger.info("📤 n8n POST ok (200)")
    else:
        logger.error(f"❌ n8n POST failed ({r.status_code}): {r.text}")


# ------------------------------------------------------------------------
# Bot
# ------------------------------------------------------------------------
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    async with aiohttp.ClientSession() as session:
        logger.info("Starting bot")

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [{
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        }]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        heygen = HeyGenVideoService(
            api_key=os.getenv("HEYGEN_API_KEY"),
            session=session,
            session_request=NewSessionRequest(
                avatar_id="Shawn_Therapist_public"  # Or your custom avatar ID
            ),
        )

        pipeline = Pipeline([
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            heygen,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_t, _c):
            logger.info("Client connected")
            # Kick off conversation (you can keep or remove this extra system nudge)
            messages.append({"role": "system", "content": "Say hello and briefly introduce yourself. Make sure to ask the user for their name"})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        async def end_and_export(reason: str):
            logger.info(f"Conversation ended: {reason}")
            transcript = context.get_messages()  # raw: system/user/assistant
            post_to_n8n(transcript)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_t, _c):
            await end_and_export("disconnect")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)

async def bot(runner_args: RunnerArguments):
    '''"""Main bot entry point for the bot starter."""
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,

            video_out_enabled=True,         # Enable video output
            video_out_is_live=True,         # Real-time video streaming
            video_out_width=1280,
            video_out_height=720,

            vad_analyzer=SileroVADAnalyzer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)'''

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
