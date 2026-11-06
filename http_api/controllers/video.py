from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from http_api.dependencies import get_video_service, uploaded_file

router = APIRouter(prefix="/video", tags=["Видео"])


@router.post("/detect", response_class=FileResponse, responses={200: {"content": {"video/mp4": {}}}})
def detect(
    confidence: float = Query(0.5, gt=0, le=1),
    path: Path = Depends(uploaded_file),
    service=Depends(get_video_service),
):
    output = path.parent / "processed.mp4"
    try:
        service.process_video(path, output, minimum_confidence=confidence)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return FileResponse(output, media_type="video/mp4", filename="processed.mp4")
