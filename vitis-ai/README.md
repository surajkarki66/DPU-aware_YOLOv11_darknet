# Vitis-AI deployment (HW-aware YOLOv11)

This directory holds **DPU compilation** and **post-quantization evaluation** used after you quantize a model at the repository root with `vai_quantize.py`.

## Layout

| Path | Role |
|------|------|
| [`compilation/`](compilation/) | Copy exported `.xmodel` files into `models/` and run `vai_c_xir` via `run_compile.sh` |
| [`evaluation/Quantized_Model/`](evaluation/Quantized_Model/) | ONNX eval for detect or OBB (`eval_onnx.py`) |
| [`evaluation/Compiled_Model/`](evaluation/Compiled_Model/) | FPGA NPZ inference + eval (`fpga_inference*.py`, `eval_npz.py`) |

## Prerequisites

- **Host / GPU**: Training and `vai_quantize.py` run from the repo root (see [root README](../README.md)).
- **Vitis-AI tools**: Quantization uses `pytorch_nndct`; compilation uses `vai_c_xir` in a Vitis-AI release matching your target platform.
- **Evaluation**: OpenCV, ONNX Runtime; config pickle from `vai_quantize.py` calib export.

## Typical flow

1. Train or obtain a checkpoint (`runs/best_state_dict.pt`).
2. From the repo root, run calibration → test → deploy with `vai_quantize.py` or `run_compression.sh` / `run_compression_obb.sh`.
3. Copy the exported `.xmodel` into `compilation/models/` and follow [`compilation/README.md`](compilation/README.md).
4. Score outputs with scripts under [`evaluation/README.md`](evaluation/README.md).

Scripts under `vitis-ai/` assume paths relative to their directory unless documented otherwise.
