from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from http_api.controllers import audio, image, text, video


def load_services(app: FastAPI):
    from service.audio import SpeechRecognizer
    from service.image import ImageClassifier
    from service.text import SentimentService
    from service.video import VideoDetector

    app.state.text_service = SentimentService()
    app.state.audio_service = SpeechRecognizer()
    app.state.image_service = ImageClassifier()
    app.state.video_service = VideoDetector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(load_services, app)
    try:
        yield
    finally:
        for name in ("text", "audio", "image", "video"):
            delattr(app.state, f"{name}_service")


def create_app():
    app = FastAPI(title="ML Applications API", version="1.0.0", lifespan=lifespan)
    for controller in (text, audio, image, video):
        app.include_router(controller.router)
    return app


app = create_app()
