from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def classify_image(self, image_path, top_k=5):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден:{path}")

        try:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("Не удалось прочитать изображение") from error
        processed_image = self.preprocess(rgb_image).unsqueeze(0)
        processed_image = processed_image.to(self.device)

        with torch.inference_mode():
            class_logits = self.model(processed_image)[0]
            class_probabilities = torch.softmax(class_logits, dim=0)
            top_probabilities, class_indexes = torch.topk(class_probabilities, k=min(top_k, len(self.categories)))

        return [
            {
                "class_id": int(rank),
                "name": self.categories[int(rank)],
                "probability": float(score),
            }
            for score, rank in zip(top_probabilities.cpu().tolist(), class_indexes.cpu().tolist())
        ]
