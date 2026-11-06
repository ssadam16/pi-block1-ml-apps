import argparse
from pathlib import Path
import json

from service.text import SentimentService


def main():
    parser = argparse.ArgumentParser(description="Анализ тональности русского текста")
    parser.add_argument("--text", required=True, help="Текст")
    args = parser.parse_args()

    try:
        result = SentimentService().analyze_sentiment(args.text)
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
