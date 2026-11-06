import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"
DISPLAY_NAMES = {
    "negative": "отрицательная",
    "neutral": "нейтральная",
    "positive": "положительная",
}


class SentimentService:
    def __init__(self, model_name=MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def analyze_sentiment(self, text):
        if not text or not text.strip():
            raise ValueError("Введите текст для анализа: ")

        encoded_text = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded_text = {name: value.to(self.device) for name, value in encoded_text.items()}

        with torch.inference_mode():
            sentiment_logits = self.model(**encoded_text).logits[0]
            class_probabilities = torch.softmax(sentiment_logits, dim=-1).cpu().tolist()

        class_labels = self.model.config.id2label
        sentiment_scores = {}
        for class_index, confidence in enumerate(class_probabilities):
            label = class_labels.get(class_index, class_labels.get(str(class_index), f"LABEL_{class_index}"))
            display_name = DISPLAY_NAMES.get(str(label).lower(), str(label))
            sentiment_scores[display_name] = round(float(confidence), 4)

        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        return {
            "sentiment": predicted_sentiment,
            "confidence": sentiment_scores[predicted_sentiment],
            "scores": sentiment_scores,
            "device": str(self.device),
        }
