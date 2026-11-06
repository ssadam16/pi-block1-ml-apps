# Распознавание речи

Распознаёт русскую речь в WAV, FLAC или OGG-файлах длительностью до 30 секунд с помощью `openai/whisper-tiny`.

Из корня проекта:

```bash
python -m console.audio.app "запись.wav"
python -m console.audio.app "запись.wav" --output "расшифровка.txt"
```
