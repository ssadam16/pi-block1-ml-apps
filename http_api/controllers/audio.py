from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from http_api.dependencies import get_audio_service, uploaded_file
from dto.response import TranscriptionResponse

router = APIRouter(prefix="/audio", tags=["Аудио"])


@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(path: Path = Depends(uploaded_file), service=Depends(get_audio_service)):
    try:
        return {"text": service.recognize_speech(path)}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
