"""
data_loader.py
Reads the dataset automatically, detects class subfolders, validates
images, and produces train/validation/test splits (80/10/10) with
augmentation and normalization applied only to the training split.

Expected (and auto-detected) layout -- any of these are handled:

    dataset/
        drowsy/          *.jpg / *.png ...
        not_drowsy/       *.jpg / *.png ...

or

    dataset/
        train/
            drowsy/...
            not_drowsy/...
        test/
            drowsy/...
            not_drowsy/...

Class folder names are matched case-insensitively against common synonyms
(e.g. "drowsy"/"fatigue"/"sleepy" -> Drowsy, "alert"/"not_drowsy"/
"non_drowsy"/"awake" -> Not Drowsy) so the code adapts to the dataset
instead of requiring the user to rename folders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config
from utils.exceptions import CorruptedImageError, DatasetError
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm"}

_DROWSY_SYNONYMS = {"drowsy", "fatigue", "fatigued", "sleepy", "tired", "drowsiness"}
_ALERT_SYNONYMS = {
    "not_drowsy", "notdrowsy", "non_drowsy", "nondrowsy", "alert", "awake",
    "active", "non-drowsy", "not-drowsy",
}


def _classify_folder_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    if normalized in _DROWSY_SYNONYMS:
        return "Drowsy"
    if normalized in _ALERT_SYNONYMS:
        return "Not Drowsy"
    # Fall back to substring matching for names like "0_drowsy" or "closed_eyes"
    if "drowsy" in normalized or "sleep" in normalized or "closed" in normalized or "fatig" in normalized:
        return "Drowsy"
    if "alert" in normalized or "awake" in normalized or "open" in normalized or "active" in normalized:
        return "Not Drowsy"
    return name  # unrecognised; caller decides what to do


@dataclass
class DatasetSample:
    path: Path
    label: str  # "Drowsy" or "Not Drowsy"


def discover_class_folders(root: Path) -> Dict[str, Path]:
    """Finds class subfolders anywhere up to two levels deep and maps them
    to the canonical labels 'Drowsy' / 'Not Drowsy'."""
    if not root.exists():
        raise DatasetError(
            f"Dataset directory '{root}' does not exist. Create it and add "
            f"your images -- see dataset/README.md for the expected layout."
        )

    candidates: Dict[str, Path] = {}

    # Depth 1: dataset/<class>/
    for child in sorted(root.iterdir()):
        if child.is_dir():
            label = _classify_folder_name(child.name)
            if label in ("Drowsy", "Not Drowsy") and _has_images(child):
                candidates[label] = child

    if len(candidates) == 2:
        return candidates

    # Depth 2: dataset/<split>/<class>/  -- merge splits, treat as one pool
    # (train.py performs its own 80/10/10 split regardless of any existing
    # train/test folders the dataset ships with).
    merged: Dict[str, List[Path]] = {"Drowsy": [], "Not Drowsy": []}
    found_any = False
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        for child in sorted(split_dir.iterdir()):
            if child.is_dir():
                label = _classify_folder_name(child.name)
                if label in ("Drowsy", "Not Drowsy") and _has_images(child):
                    merged[label].append(child)
                    found_any = True

    if found_any and merged["Drowsy"] and merged["Not Drowsy"]:
        # Represent as pseudo-single dirs by returning the list wrapped later;
        # simplest is to raise a marker so caller uses list-based collection.
        return {"__multi__": root}  # signal handled in collect_samples

    raise DatasetError(
        f"Could not find two class folders (Drowsy / Not Drowsy and their "
        f"common synonyms) under '{root}'. Found top-level entries: "
        f"{[p.name for p in root.iterdir() if p.is_dir()]}. Please place "
        f"images under dataset/drowsy/ and dataset/not_drowsy/ (or similarly "
        f"named folders) -- see dataset/README.md."
    )


def _has_images(folder: Path) -> bool:
    for f in folder.rglob("*"):
        if f.suffix.lower() in VALID_EXTENSIONS:
            return True
    return False


def collect_samples(root: Path = config.DATASET_DIR) -> List[DatasetSample]:
    class_map = discover_class_folders(root)
    samples: List[DatasetSample] = []

    if "__multi__" in class_map:
        for split_dir in sorted(root.iterdir()):
            if not split_dir.is_dir():
                continue
            for child in sorted(split_dir.iterdir()):
                if not child.is_dir():
                    continue
                label = _classify_folder_name(child.name)
                if label not in ("Drowsy", "Not Drowsy"):
                    continue
                samples.extend(_collect_from_folder(child, label))
    else:
        for label, folder in class_map.items():
            samples.extend(_collect_from_folder(folder, label))

    if not samples:
        raise DatasetError(f"No valid images found under '{root}'.")

    drowsy_n = sum(1 for s in samples if s.label == "Drowsy")
    not_drowsy_n = len(samples) - drowsy_n
    logger.info(
        "Dataset discovered: %d Drowsy, %d Not Drowsy (total %d)",
        drowsy_n, not_drowsy_n, len(samples),
    )
    if drowsy_n == 0 or not_drowsy_n == 0:
        raise DatasetError(
            "One of the two classes has zero images -- cannot train a "
            "binary classifier. Check dataset/README.md for the expected "
            "folder layout."
        )
    return samples


def _collect_from_folder(folder: Path, label: str) -> List[DatasetSample]:
    out = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() in VALID_EXTENSIONS:
            out.append(DatasetSample(path=f, label=label))
    return out


def _load_and_preprocess(path: Path, size: int = config.IMG_SIZE) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CorruptedImageError(f"Could not decode image: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=-1)


def load_dataset_arrays(
    root: Path = config.DATASET_DIR,
    seed: int = config.RANDOM_SEED,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Loads every valid image into memory as (X, y) arrays and returns
    train/val/test splits (80/10/10), stratified by class, with corrupted
    images skipped (and logged) rather than crashing the run."""
    samples = collect_samples(root)
    random.Random(seed).shuffle(samples)

    class_names = list(config.CLASS_NAMES)  # ("Not Drowsy", "Drowsy")
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    by_class: Dict[str, List[DatasetSample]] = {"Drowsy": [], "Not Drowsy": []}
    for s in samples:
        by_class[s.label].append(s)

    train_samples: List[DatasetSample] = []
    val_samples: List[DatasetSample] = []
    test_samples: List[DatasetSample] = []

    for label, items in by_class.items():
        n = len(items)
        n_train = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)
        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    random.Random(seed).shuffle(train_samples)
    random.Random(seed + 1).shuffle(val_samples)
    random.Random(seed + 2).shuffle(test_samples)

    def _build(split: List[DatasetSample]) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        skipped = 0
        for s in split:
            try:
                X_list.append(_load_and_preprocess(s.path))
                y_list.append(label_to_index[s.label])
            except CorruptedImageError as exc:
                skipped += 1
                logger.warning(str(exc))
        if skipped:
            logger.warning("Skipped %d corrupted/unreadable images.", skipped)
        if not X_list:
            raise DatasetError("A dataset split ended up empty after filtering corrupted images.")
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        y_onehot = np.eye(len(class_names), dtype=np.float32)[y]
        return X, y_onehot

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_samples), len(val_samples), len(test_samples),
    )

    return _build(train_samples), _build(val_samples), _build(test_samples)


def build_augmentation_generator():
    """Keras ImageDataGenerator applying the augmentation strategy from the
    implementation strategy (rotation, shift, zoom, brightness, flip) --
    used only on the training split."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=config.AUGMENTATION_ROTATION_RANGE,
        width_shift_range=config.AUGMENTATION_WIDTH_SHIFT,
        height_shift_range=config.AUGMENTATION_HEIGHT_SHIFT,
        zoom_range=config.AUGMENTATION_ZOOM_RANGE,
        brightness_range=config.AUGMENTATION_BRIGHTNESS_RANGE,
        horizontal_flip=config.AUGMENTATION_HORIZONTAL_FLIP,
        fill_mode="nearest",
    )
