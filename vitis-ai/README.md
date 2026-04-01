# Vitis-AI deployment (HW-aware YOLOv11)

This directory holds **DPU compilation** and **post-quantization evaluation** helpers used after you quantize a model at the repository root with `vai_quantize.py`.

## Layout

| Path | Role |
|------|------|
| [`compilation/`](compilation/) | Copy exported `.xmodel` files here and run `vai_c_xir` via `run_compile.sh` for a chosen DPU RAM (`arch_B*.json`). |
| [`evaluation/Detect/`](evaluation/Detect/) | Axis-aligned detection: text dump → COCO-style JSON → `pycocotools` metrics (needs `gt_eval.json` in the working directory). |
| [`evaluation/OBB/`](evaluation/OBB/) | Oriented boxes: ONNX + config → `.pkl` predictions → `evaluate_obb.py` against YOLO OBB labels. |

## Prerequisites

- **Host / GPU**: Training, `vai_quantize.py` calibration and test, and ONNX-based evaluation run from the repo root with the Python environment described in the [root README](../README.md).
- **Vitis-AI tools**: Quantization uses `pytorch_nndct`; compilation uses `vai_c_xir` inside a Vitis-AI release that matches your target platform.
- **Evaluation extras**: `pycocotools` for Detect COCO eval; OpenCV / ONNX Runtime where noted in [`evaluation/README.md`](evaluation/README.md).

## Typical flow

1. Train or obtain a checkpoint (`.pt`).
2. From the repo root, run calibration → test → deploy export with `vai_quantize.py` (see [Quantization & deployment](../README.md#-quantization--deployment)) or `run_compression.sh`.
3. Copy the exported `.xmodel` into `vitis-ai/compilation/models/` and follow [`compilation/README.md`](compilation/README.md).
4. Optionally run scripts under `evaluation/` to score quantized ONNX or exported outputs on validation data.

Scripts under `vitis-ai/` assume paths relative to their own directory unless documented otherwise; evaluation scripts that import `models/` or `utils/` resolve the repository root automatically where implemented.
