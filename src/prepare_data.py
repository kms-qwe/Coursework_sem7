"""Кэш 256x256 в data/cache/. ISIC train делится на train 810 / val 90, PH2 только для теста."""
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import CACHE_DIR, IMG_SIZE, RAW_DIR

VAL_SIZE = 90
SEED = 42


def isic_pairs(part):
    img_dir = RAW_DIR / "isic2016" / f"ISBI2016_ISIC_Part1_{part}_Data"
    gt_dir = RAW_DIR / "isic2016" / f"ISBI2016_ISIC_Part1_{part}_GroundTruth"
    pairs = []
    for img in sorted(img_dir.glob("*.jpg")):
        masks = list(gt_dir.glob(f"{img.stem}_*egmentation.png"))
        if len(masks) != 1:
            raise FileNotFoundError(f"нет маски для {img.name}")
        pairs.append((img, masks[0]))
    if not pairs:
        raise FileNotFoundError(f"пусто в {img_dir}, сначала download_data.sh")
    return pairs


def resize_pair(img, mask):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    mask = mask.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8), np.asarray(mask) > 127


def load_isic(pair):
    img = Image.open(pair[0])
    img.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))  # так быстрее грузятся большие jpg
    return resize_pair(img, Image.open(pair[1]))


def save(name, items, ids):
    X = np.stack([a for a, _ in items])
    Y = np.stack([b for _, b in items])
    np.savez(CACHE_DIR / f"{name}_{IMG_SIZE}.npz", X=X, Y=Y, ids=np.array(ids))
    print(f"{name:5s}: {len(ids):4d} изображений, доля пикселей очага {Y.mean():.3f}")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        train_pairs = isic_pairs("Training")
        train_items = list(ex.map(load_isic, train_pairs))
        test_pairs = isic_pairs("Test")
        test_items = list(ex.map(load_isic, test_pairs))

    order = np.random.default_rng(SEED).permutation(len(train_pairs))
    val_idx, tr_idx = sorted(order[:VAL_SIZE]), sorted(order[VAL_SIZE:])
    save("train", [train_items[i] for i in tr_idx], [train_pairs[i][0].stem for i in tr_idx])
    save("val", [train_items[i] for i in val_idx], [train_pairs[i][0].stem for i in val_idx])
    save("test", test_items, [p[0].stem for p in test_pairs])

    table = pq.read_table(RAW_DIR / "ph2" / "ph2_256.parquet").to_pylist()
    ph2 = [resize_pair(Image.open(io.BytesIO(r["image"]["bytes"])), Image.open(io.BytesIO(r["label"]["bytes"])))
           for r in table]
    save("ph2", ph2, [r["image"].get("path") or f"ph2_{i}" for i, r in enumerate(table)])


if __name__ == "__main__":
    main()
