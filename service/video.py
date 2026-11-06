import math
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)


class VideoDetector:
    def __init__(self, requested_device="auto"):
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна")

        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(requested_device)
        self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def draw_objects_on_frame(self, frame, minimum_confidence):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image_tensor = self.preprocess(image).to(self.device)

        with torch.inference_mode():
            detected_objects = self.model([image_tensor])[0]

        height, width = frame.shape[:2]
        boxes = detected_objects["boxes"].detach().cpu().tolist()
        scores = detected_objects["scores"].detach().cpu().tolist()
        labels = detected_objects["labels"].detach().cpu().tolist()

        for box, score, label_index in zip(boxes, scores, labels):
            if score < minimum_confidence:
                continue

            left, top, right, bottom = [int(value) for value in box]
            left = max(0, min(width - 1, left))
            right = max(0, min(width - 1, right))
            top = max(0, min(height - 1, top))
            bottom = max(0, min(height - 1, bottom))
            label = self.categories[label_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (40, 210, 40), 2)
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 210, 40),
                2,
                cv2.LINE_AA,
            )

        return frame

    def process_video(self, input_path, output_path, minimum_confidence=0.5):
        input_video = Path(input_path)
        output_video = Path(output_path)

        if not input_video.is_file():
            raise FileNotFoundError(f"Видео не найдено: {input_video}")
        if input_video.resolve() == output_video.resolve():
            raise ValueError("Выходной файл должен отличаться от исходного видео")
        if output_video.suffix.lower() != ".mp4":
            raise ValueError("Выходной файл должен иметь расширение .mp4.")
        if not 0 < minimum_confidence <= 1:
            raise ValueError("Порог уверенности должен быть больше 0 и не больше 1")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        video_reader = cv2.VideoCapture(str(input_video))
        if not video_reader.isOpened():
            video_reader.release()
            raise ValueError("Не удалось открыть видео")

        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        video_writer = None
        processed_frames = 0
        started = time.perf_counter()

        try:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_video), codec, fps, (width, height))
            if not video_writer.isOpened():
                raise RuntimeError("Не удалось создать файл для результата")

            while True:
                frame_available, frame = video_reader.read()
                if not frame_available:
                    break

                video_writer.write(self.draw_objects_on_frame(frame, minimum_confidence))
                processed_frames += 1
        finally:
            video_reader.release()
            if video_writer is not None:
                video_writer.release()

        if processed_frames == 0:
            raise ValueError("В видео нет кадров для обработки")

        elapsed = time.perf_counter() - started
        return {
            "output": output_video,
            "frames": processed_frames,
            "seconds": elapsed,
            "device": str(self.device),
        }
