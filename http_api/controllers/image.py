from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from http_api.dependencies import get_image_service, uploaded_file
from dto.response import ImageResponse

router = APIRouter(prefix="/image", tags=["Изображения"])


@router.post("/classify", response_model=ImageResponse)
def classify(
    top_k: int = Query(5, ge=1, le=1000),
    path: Path = Depends(uploaded_file),
    service=Depends(get_image_service),
):
    try:
        return {"predictions": service.classify_image(path, top_k=top_k)}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
