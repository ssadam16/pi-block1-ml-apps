import argparse
from pathlib import Path

from service.image import ImageClassifier


def main():
    parser = argparse.ArgumentParser(description="Классификация изображения с помощью MobileNetV2")
    parser.add_argument("image", help="Путь к изображению")
    args = parser.parse_args()

    if not Path(args.image).is_file():
        parser.error(f"Файл не найден: {args.image}")

    predictions = ImageClassifier().classify_image(args.image)
    for rank, image_prediction in enumerate(predictions, start=1):
        probability = image_prediction["probability"] * 100
        print(f"{rank}. {image_prediction['name']} — {probability:.1f}%")


if __name__ == "__main__":
    main()
