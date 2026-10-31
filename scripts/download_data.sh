#!/usr/bin/env bash
# ISIC 2016 Part 1, PH2 (256x256 с HF) и веса imagenet для resnet34
set -euo pipefail
cd "$(dirname "$0")/.."

ISIC_URL=https://isic-archive.s3.amazonaws.com/challenges/2016
PH2_URL=https://huggingface.co/datasets/ahishamm/ph2_masks/resolve/main/data/train-00000-of-00001-9b48dfef44b35212.parquet

mkdir -p data/raw/isic2016 data/raw/ph2

for f in ISBI2016_ISIC_Part1_Training_Data ISBI2016_ISIC_Part1_Training_GroundTruth \
         ISBI2016_ISIC_Part1_Test_Data ISBI2016_ISIC_Part1_Test_GroundTruth; do
  if [ -d "data/raw/isic2016/$f" ]; then
    echo "уже есть: $f"
    continue
  fi
  echo "скачиваю $f.zip"
  curl -fL -C - --retry 5 -o "data/raw/isic2016/$f.zip" "$ISIC_URL/$f.zip"
  unzip -q -o "data/raw/isic2016/$f.zip" -d data/raw/isic2016
  rm "data/raw/isic2016/$f.zip"
done

if [ ! -f data/raw/ph2/ph2_256.parquet ]; then
  echo "скачиваю PH2"
  curl -fL --retry 5 -o data/raw/ph2/ph2_256.parquet "$PH2_URL"
fi

echo "скачиваю веса ImageNet для ResNet34"
.venv/bin/python -c "import segmentation_models_pytorch as smp; smp.Unet('resnet34', encoder_weights='imagenet')"

du -sh data/raw/*
