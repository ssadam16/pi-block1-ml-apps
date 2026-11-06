from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_NAME = "openai/whisper-tiny"
SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30


class SpeechRecognizer:
    def __init__(self, model_name=MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def read_audio_file(audio_file_path):
        try:
            audio_samples, original_sample_rate = sf.read(audio_file_path, dtype="float32", always_2d=True)
        except sf.LibsndfileError as error:
            raise ValueError("Не удалось прочитать аудиофайл") from error
        audio_samples = audio_samples.mean(axis=1)

        if audio_samples.size == 0:
            raise ValueError("В аудиофайле нет звука.")

        if original_sample_rate != SAMPLE_RATE:
            divisor = np.gcd(original_sample_rate, SAMPLE_RATE)
            audio_samples = resample_poly(
                audio_samples,
                SAMPLE_RATE // divisor,
                original_sample_rate // divisor,
            ).astype(np.float32)

        duration = len(audio_samples) / SAMPLE_RATE
        if duration > MAX_DURATION_SECONDS:
            raise ValueError("Запись не должна превышать 30 секунд.")

        return audio_samples

    def recognize_speech(self, audio_file_path):
        path = Path(audio_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")

        audio_samples = self.read_audio_file(path)
        audio_inputs = self.processor(
            audio_samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        audio_features = audio_inputs.input_features.to(self.device)
        audio_attention_mask = audio_inputs.attention_mask.to(self.device)

        with torch.inference_mode():
            transcribed_tokens = self.model.generate(
                audio_features,
                attention_mask=audio_attention_mask,
                language="russian",
                task="transcribe",
                max_new_tokens=225,
            )

        return self.processor.batch_decode(
            transcribed_tokens,
            skip_special_tokens=True,
        )[0].strip()
