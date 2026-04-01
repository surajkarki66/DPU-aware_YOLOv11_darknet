# Evaluation (quantized / deployment outputs)

Utilities for scoring **axis-aligned detection** (COCO-style) and **oriented bounding boxes (OBB)** after quantization. These scripts live under the repo’s `vitis-ai/evaluation/` tree; many are meant to run with the working directory set to the script’s folder or with explicit paths as documented below.

## Detection (`Detect/`)

| Script | Purpose |
|--------|---------|
| [`post_processing.py`](Detect/post_processing.py) | Decode raw ONNX outputs (DFL, anchors) to boxes. Usage: `python post_processing.py <path_to_output> <model_name> <iou_threshold> <img_height> <img_width>` |
| [`coco_prep.py`](Detect/coco_prep.py) | Convert a text dump `output_boxes_<model_name>.txt` into a JSON dict keyed by image id/filename: `coco-preped-<model_name>.json`. Args: `python coco_prep.py <model_name>` |
| [`evaluate.py`](Detect/evaluate.py) | COCO metrics via `pycocotools`. Usage: `python evaluate.py <detection_results.json>`. Expects **`gt_eval.json`** (COCO ground truth) in the **current working directory** and detection JSON in the format produced by the prep step (filename → list of `[x, y, w, h, score]`). |

**Typical sequence** (from `vitis-ai/evaluation/Detect/` or with adjusted paths):

1. Produce `output_boxes_<model_name>.txt` from your runtime (DPU dump or ONNX run).
2. `python coco_prep.py <model_name>`
3. Ensure `gt_eval.json` is present in the directory from which you run `evaluate.py`.
4. `python evaluate.py coco-preped-<model_name>.json` (or your merged detections file in the same structure).

## OBB (`OBB/`)

| Script | Purpose |
|--------|---------|
| [`post_processing_obb.py`](OBB/post_processing_obb.py) | Run quantized **OBB** ONNX over a folder of images; writes `*_obb_predictions.pkl`. Requires `--onnx`, `--config` (quantization `*_config.pkl`), `--test-data`, optional `--split`. |
| [`evaluate_obb.py`](OBB/evaluate_obb.py) | Compute precision/recall/mAP against YOLO-format OBB labels. Requires `--data-dir`, `--val-list`, and either `--predictions` (pkl from post-processing) or `--onnx` + `--config`, or `--quantize-dir` + `--model-name`. |

Example evaluation (after generating a pickle):

```bash
python evaluate_obb.py \
  --data-dir /path/to/dataset \
  --val-list /path/to/val2017.txt \
  --predictions /path/to/YOLO_int_obb_predictions.pkl \
  --input-size 416
```

Class names default from `data/hyps/args.yaml` unless you pass `--hyp`.

### OBB test data layout

See [`OBB/test_data/README.md`](OBB/test_data/README.md) for the expected dataset directory structure used by `--test-data` and evaluation lists.

## Dependencies

- **Detect `evaluate.py`**: `pycocotools`, COCO-format `gt_eval.json`.
- **OBB**: `torch`, `opencv-python`, `numpy`, `pyyaml`; ONNX path needs `onnxruntime`.
