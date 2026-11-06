# Набор ML-приложений

В репозитории собраны четыре небольшие программы на Python. Каждая использует готовую модель и выполняет одну задачу: определяет тональность текста, переводит речь в текст, классифицирует изображение или находит объекты в видео.

Модели не обучаются. Их веса загружаются автоматически при первом запуске, поэтому для первого запуска нужен интернет.

## Состав проекта

| Папка | Что делает | Библиотека и модель |
|---|---|---|
| `text` | Определяет положительную, нейтральную или отрицательную тональность русского текста | Hugging Face Transformers, `cointegrated/rubert-tiny-sentiment-balanced` |
| `audio` | Распознаёт русскую речь в аудиофайле | Hugging Face Transformers, `openai/whisper-tiny` |
| `image` | Показывает пять наиболее вероятных классов изображения | PyTorch / TorchVision, MobileNetV2 с весами ImageNet |
| `video` | Отмечает рамками найденные объекты и сохраняет обработанное видео | PyTorch / TorchVision, Faster R-CNN MobileNetV3 на COCO |

В проекте применяются два варианта из задания: Hugging Face Transformers для текста и аудио, а также PyTorch с TorchVision для изображений и видео.

## Активируем окружение

Открываем терминал в корневой папке проекта:
`python -m venv .venv`

В Windows PowerShell:
`.\.venv\Scripts\Activate.ps1`

Если в PowerShell запрещает выполнение скрипта:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Устанавливаем зависимости:
`pip install -r requirements.txt`

Сохранить текущие зависимости:
`pip freeze > requirements.txt`

Если хотите использовать GPU с поддержкой CUDA, вместо CPU:
```PowerShell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```


## Структура

```text
ml-applications/
├── audio/
│   ├── app.py
│   └── README.md
├── docs/
│   └── task-specs.md
├── image/
│   ├── app.py
│   └── README.md
├── text/
│   ├── app.py
│   └── README.md
├── video/
│   ├── app.py
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```