from fastapi import APIRouter, Depends

from http_api.dependencies import get_text_service
from dto.request import SentimentRequest
from dto.response import SentimentResponse

router = APIRouter(prefix="/text", tags=["Текст"])


@router.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(body: SentimentRequest, service=Depends(get_text_service)):
    return service.analyze_sentiment(body.text)
