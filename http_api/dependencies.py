from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException, Request, UploadFile


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def get_text_service(request: Request):
    return request.app.state.text_service


def get_audio_service(request: Request):
    return request.app.state.audio_service


def get_image_service(request: Request):
    return request.app.state.image_service


def get_video_service(request: Request):
    return request.app.state.video_service


def uploaded_file(file: UploadFile):
    with TemporaryDirectory(prefix="ml-api-") as directory:
        path = Path(directory) / "input.bin"
        size = 0
        with path.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Размер файла превышает 50 МиБ")
                target.write(chunk)
        if not size:
            raise HTTPException(422, "Загружен пустой файл")
        yield path
