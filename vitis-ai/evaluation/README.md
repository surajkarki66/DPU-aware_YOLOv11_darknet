# Vitis AI evaluation (HW-aware YOLOv11)

Post-processing and metrics for **Detect** and **OBB** after quantization or FPGA inference.

Two deployment paths:

| Path | Folder | When to use |
|------|--------|-------------|
| **Quantized (ONNX)** | `Quantized_Model/` | PC-side ONNX from quantization deploy |
| **Compiled (FPGA)** | `Compiled_Model/` | DPU after `vai_c_xir` compilation |

Both paths use the **config pickle** exported during `vai_quantize.py` calibration:

`(tensor_no, tensor_stride, tensor_ch, tensor_nc, task)` — pass via `--quant-meta`.

Use `--task detect` or `--task obb`.

## Quantized (ONNX)

```bash
cd vitis-ai/evaluation/Quantized_Model
python eval_onnx.py \
  --task detect \
  --model path/to/model.onnx \
  --data path/to/data.yaml \
  --quant-meta path/to/config.pkl \
  --imgsz 416
```

For OBB, set `--task obb`. Writes `metrics.json` and optional visualizations under `--save-dir`.

## Compiled (FPGA NPZ)

**Step 1 — run inference on device or Vitis runtime:**

```bash
cd vitis-ai/evaluation/Compiled_Model
python fpga_inference.py models/yolov11n.xmodel ./images predictions.npz 416
# OBB: use fpga_inference_obb.py
```

**Step 2 — evaluate NPZ predictions:**

```bash
python eval_npz.py \
  --predictions-npz predictions.npz \
  --task detect \
  --data path/to/data.yaml \
  --quant-meta path/to/config.pkl \
  --imgsz 416
```

## Layout

| File | Role |
|------|------|
| `Quantized_Model/eval_onnx.py` | ONNX Runtime eval for detect or OBB |
| `Quantized_Model/eval_core.py` | Shared decode + mAP logic |
| `Quantized_Model/utils.py` | Metrics and helper utilities |
| `Compiled_Model/fpga_inference.py` | Detect model → NPZ on FPGA |
| `Compiled_Model/fpga_inference_obb.py` | OBB model → NPZ on FPGA |
| `Compiled_Model/eval_npz.py` | Decode NPZ + mAP for detect or OBB |

## Notes

- Match `--imgsz` to training, quantization, and compilation input size (e.g. 416).
- Detect uses 3 outputs; OBB uses 6 outputs (3 box + 3 angle heads).
- `eval_npz.py` reuses decoders from `Quantized_Model/eval_core.py`.
