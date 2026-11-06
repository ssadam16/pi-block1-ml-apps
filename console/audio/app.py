import argparse
from pathlib import Path

from service.audio import SpeechRecognizer


def main():
    parser = argparse.ArgumentParser(description="Распознавание русской речи из аудиофайла")
    parser.add_argument("audio_file_path", help="Путь к WAV, FLAC или OGG-файлу длительностью до 30 секунд")
    parser.add_argument("--output", help="Путь к TXT-файлу для сохранения расшифровки")
    args = parser.parse_args()

    try:
        transcript = SpeechRecognizer().recognize_speech(args.audio_file_path)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    except Exception as error:
        parser.error(f"Не удалось обработать аудиофайл: {error}")

    print(transcript or "Речь не распознана.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript + "\n", encoding="utf-8")
        print(f"Расшифровка сохранена в: {output_path}")


if __name__ == "__main__":
    main()

