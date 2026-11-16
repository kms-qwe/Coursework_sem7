# Прунинг U-Net для сегментации кожных образований

Курсовая работа, 7 семестр. Тема: оптимизация моделей сегментации изображений для устройств с ограниченными ресурсами.

Модель: `smp.Unet("resnet34", classes=1)`, вход 256x256.
Данные: ISIC 2016 Part 1 (train/val/test), PH2 как внешний тест.

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Данные

Данных и весов в репозитории нет.

```bash
scripts/download_data.sh             # ISIC 2016, PH2, веса ImageNet для resnet34
.venv/bin/python src/prepare_data.py # кэш 256x256 в data/cache/
```

Откуда качается:
- ISIC 2016 Part 1: https://isic-archive.s3.amazonaws.com/challenges/2016/
- PH2 (256x256): https://huggingface.co/datasets/ahishamm/ph2_masks

## Запуск

```bash
scripts/run_all.sh
```

Или по шагам:

```bash
.venv/bin/python src/train_base.py      # исходная модель -> runs/models/base.pth
.venv/bin/python src/prune_baseline.py  # бейзлайн прунинга
.venv/bin/python src/benchmark.py       # размер и скорость, экспорт в ONNX
.venv/bin/python src/report.py          # runs/results/summary.md
.venv/bin/python src/own_method_mini.py # свой метод (черновая версия, один уровень)
```

У всех скриптов есть флаг `--smoke` для быстрой проверки на 16 картинках (вывод в `runs_smoke/`).

## Свой метод

Обучение идет в Google Colab, замер скорости — локально.

1. Выполните `scripts/pack_for_colab.sh` и загрузите `colab_upload/coursework_sem7.zip` в корень «Мой диск».
2. Откройте в Colab `notebooks/own_method_colab.ipynb` (среда с GPU) и выполните ячейки по порядку.
3. Скачайте с Диска папку `coursework_sem7/runs`, распакуйте ее в корень проекта и выполните:

```bash
.venv/bin/python src/own_method.py --bench-only  # время и размер -> runs/results/own/own_benchmark.csv
.venv/bin/python src/own_report.py               # runs/results/own/own_summary.md и графики
```

## Результаты

Таблицы и логи лежат в `runs/results/` и `runs/logs/`, сводка в `runs/results/summary.md`.
Веса (`runs/models/`) и ONNX (`runs/onnx/`) в git не кладутся, они получаются запуском скриптов выше.

Загрузка модели:

```python
import sys; sys.path.insert(0, "src")
from common import load_model
model = load_model("runs/models/l1_50.pth")
```
