from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import http_api.app as application
from http_api import dependencies


@pytest.fixture
def services():
    return {name: Mock() for name in ("text", "audio", "image", "video")}


@pytest.fixture
def client(monkeypatch, services):
    def load(app):
        for name, service in services.items():
            setattr(app.state, f"{name}_service", service)

    monkeypatch.setattr(application, "load_services", load)
    with TestClient(application.create_app()) as test_client:
        yield test_client


def test_sentiment(client, services):
    expected = {
        "sentiment": "положительная", "confidence": 0.9,
        "scores": {"положительная": 0.9, "нейтральная": 0.08, "отрицательная": 0.02},
        "device": "cpu",
    }
    services["text"].analyze_sentiment.return_value = expected
    response = client.post("/text/sentiment", json={"text": "  Отличный фильм!  "})
    assert response.status_code == 200
    assert response.json() == expected
    services["text"].analyze_sentiment.assert_called_once_with("Отличный фильм!")


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": " \n "}, {"text": 123}, {"text": "x" * 10001}])
def test_invalid_text(client, services, body):
    assert client.post("/text/sentiment", json=body).status_code == 422
    services["text"].analyze_sentiment.assert_not_called()


def test_audio_and_cleanup(client, services):
    paths = []

    def recognize(path):
        assert path.read_bytes() == b"audio bytes"
        paths.append(path)
        return "Привет мир"

    services["audio"].recognize_speech.side_effect = recognize
    response = client.post("/audio/transcribe", files={"file": ("../../record.wav", b"audio bytes")})
    assert response.status_code == 200
    assert response.json() == {"text": "Привет мир"}
    assert not paths[0].parent.exists()


def test_image(client, services):
    predictions = [{"class_id": 1, "name": "goldfish", "probability": 0.8}]
    services["image"].classify_image.return_value = predictions
    response = client.post("/image/classify?top_k=1", files={"file": ("photo.jpg", b"image")})
    assert response.status_code == 200
    assert response.json() == {"predictions": predictions}
    assert services["image"].classify_image.call_args.kwargs == {"top_k": 1}


def test_video_response_and_cleanup(client, services):
    paths = []

    def process(path, output, minimum_confidence):
        assert path.read_bytes() == b"video input"
        assert minimum_confidence == 0.7
        output.write_bytes(b"processed video")
        paths.append(output)

    services["video"].process_video.side_effect = process
    response = client.post("/video/detect?confidence=0.7", files={"file": ("clip.mp4", b"video input")})
    assert response.status_code == 200
    assert response.content == b"processed video"
    assert response.headers["content-type"] == "video/mp4"
    assert 'filename="processed.mp4"' in response.headers["content-disposition"]
    assert not paths[0].parent.exists()


@pytest.mark.parametrize("endpoint", ["/audio/transcribe", "/image/classify", "/video/detect"])
def test_missing_and_empty_file(client, endpoint):
    assert client.post(endpoint).status_code == 422
    assert client.post(endpoint, files={"file": ("empty", b"")}).status_code == 422


@pytest.mark.parametrize("endpoint", ["/image/classify?top_k=0", "/image/classify?top_k=1001", "/video/detect?confidence=0", "/video/detect?confidence=1.1", "/video/detect?confidence=nan"])
def test_invalid_parameters(client, endpoint):
    assert client.post(endpoint, files={"file": ("input", b"data")}).status_code == 422


@pytest.mark.parametrize("endpoint,name,method", [
    ("/audio/transcribe", "audio", "recognize_speech"),
    ("/image/classify", "image", "classify_image"),
    ("/video/detect", "video", "process_video"),
])
def test_unreadable_file_and_cleanup(client, services, endpoint, name, method):
    paths = []

    def reject(path, *args, **kwargs):
        paths.append(path)
        raise ValueError("Файл повреждён")

    getattr(services[name], method).side_effect = reject
    response = client.post(endpoint, files={"file": ("broken", b"invalid data")})
    assert response.status_code == 422
    assert response.json() == {"detail": "Файл повреждён"}
    assert not paths[0].parent.exists()


def test_upload_limit(client, services, monkeypatch):
    monkeypatch.setattr(dependencies, "MAX_UPLOAD_BYTES", 3)
    response = client.post("/audio/transcribe", files={"file": ("large.wav", b"1234")})
    assert response.status_code == 413
    services["audio"].recognize_speech.assert_not_called()


def test_unexpected_failure_is_not_validation_error(client, services):
    services["audio"].recognize_speech.side_effect = RuntimeError("Inference failure")
    with pytest.raises(RuntimeError, match="Inference failure"):
        client.post("/audio/transcribe", files={"file": ("audio.wav", b"data")})


def test_openapi(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/text/sentiment", "/audio/transcribe", "/image/classify", "/video/detect",
    }
    for endpoint in ("/audio/transcribe", "/image/classify", "/video/detect"):
        assert "multipart/form-data" in schema["paths"][endpoint]["post"]["requestBody"]["content"]


def test_services_loaded_once_and_released(monkeypatch, services):
    def load(app):
        for name, service in services.items():
            setattr(app.state, f"{name}_service", service)

    loader = Mock(side_effect=load)
    monkeypatch.setattr(application, "load_services", loader)
    app = application.create_app()
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
        loader.assert_called_once_with(app)
        assert app.state.text_service is services["text"]
    for name in services:
        assert not hasattr(app.state, f"{name}_service")
