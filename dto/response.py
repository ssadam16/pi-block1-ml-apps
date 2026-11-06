from pydantic import BaseModel, Field


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float = Field(ge=0, le=1)
    scores: dict[str, float]
    device: str


class TranscriptionResponse(BaseModel):
    text: str


class ImagePrediction(BaseModel):
    class_id: int
    name: str
    probability: float = Field(ge=0, le=1)


class ImageResponse(BaseModel):
    predictions: list[ImagePrediction]
