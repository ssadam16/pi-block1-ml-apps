import argparse
from pathlib import Path

from service.video import VideoDetector


def main():
    parser = argparse.ArgumentParser(description="Поиск объектов в видео")
    parser.add_argument("video", help="Путь к исходному видео")
    parser.add_argument("--output", help="Путь к выходному файлу")
    parser.add_argument("--confidence", type=float, default=0.5, help="Минимальная уверенность обнаружение")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    input_video = Path(args.video)
    output = Path(args.output) if args.output else input_video.with_name(f"{input_video.stem}_processed.mp4")

    try:
        detector = VideoDetector(args.device)
        result = detector.process_video(input_video, output, args.confidence)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Результат сохранён: {result['output']}")
    print(f"Количество обработанных кадров: {result['frames']}")
    print(f"Время: {result['seconds']:.1f} с; устройство: {result['device']}")


if __name__ == "__main__":
    main()
