# OBB test / validation data layout

This folder documents the **directory layout** expected by [`post_processing_obb.py`](../post_processing_obb.py) (`--test-data` / `--split`) and by [`evaluate_obb.py`](../evaluate_obb.py) when combined with a path list such as `val2017.txt` or `test2017.txt`.

## Recommended COCO-like tree

Use a single dataset root (example name: `coco_data_obb`):

```
coco_data_obb/
├── images/
│   ├── train2017/
│   ├── val2017/
│   └── test2017/
├── labels/
│   ├── train2017/
│   ├── val2017/
│   └── test2017/
├── train2017.txt
├── val2017.txt
└── test2017.txt
```

- **`images/<split>/`**: RGB images (`.jpg`, `.png`, …).
- **`labels/<split>/`**: One `.txt` per image, YOLO OBB format (class and box; corner or `xywhr` as supported by `evaluate_obb.py`’s `load_gt_labels`).
- **`<split>.txt`**: Optional listing of images for that split. Paths can be relative to the dataset root, e.g. `./images/test2017/img001.jpg`, or basenames only (see `--images-subdir` in `evaluate_obb.py`).

Generate the split lists from your dataset root with the repo root helper:

```bash
python format_dataset.py --base-dir /path/to/coco_data_obb
```

That writes `train2017.txt`, `val2017.txt`, and `test2017.txt` next to `images/` and `labels/`.

## `post_processing_obb.py`

- **`--test-data`**: Dataset root (e.g. `coco_data_obb`) or a folder that contains `images/<split>/`.
- **`--split`**: e.g. `test2017` — restricts inference to `.../images/test2017/` (or the first matching subfolder found).

## `evaluate_obb.py`

- **`--data-dir`**: Dataset root (`coco_data_obb`).
- **`--val-list`**: Path to a text file listing validation images (often `val2017.txt` or `test2017.txt` from above).
- If list lines are **basenames only**, set **`--images-subdir`** (e.g. `val2017`) so full paths resolve to `data_dir/images/val2017/<name>`.

This folder may hold small samples for smoke tests; full datasets are usually stored outside the repository and pointed to with absolute paths.
