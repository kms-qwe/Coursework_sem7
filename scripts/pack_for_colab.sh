#!/usr/bin/env bash
# архив для Google Colab: код, кэш данных, base.pth, unstructured_l1_90.pth
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=colab_upload
STAGE="$OUT/coursework_sem7"
rm -rf "$OUT"
mkdir -p "$STAGE/src" "$STAGE/data/cache" "$STAGE/runs/models" "$STAGE/notebooks"
cp src/*.py "$STAGE/src/"
cp requirements.txt "$STAGE/"
cp data/cache/{train,val,test,ph2}_256.npz "$STAGE/data/cache/"
cp runs/models/base.pth runs/models/unstructured_l1_90.pth "$STAGE/runs/models/"
cp notebooks/own_method_colab.ipynb "$STAGE/notebooks/"
(cd "$OUT" && zip -qr coursework_sem7.zip coursework_sem7)
rm -rf "$STAGE"
ls -lh "$OUT"
