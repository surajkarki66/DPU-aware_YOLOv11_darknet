# Compilation model inputs

Place quantized `.xmodel` files here before running `vai_c_xir` (see `../run_compile.sh`).

Typical filenames (must match the line you uncomment in `run_compile.sh`):

| File | Model type |
|------|------------|
| `YOLO_int_B4096.xmodel` | YOLOv11 detection (example for B4096 target) |
| `YOLO_OBB_int_B4096.xmodel` | YOLOv11-OBB |

Typical source: `quantize_result/` at the repo root after `vai_quantize.py --quant_mode test --deploy`, or a renamed folder from `run_compression.sh`.

Keep Detect and OBB exports separate; compilation uses different output names (`yolov11n_*` vs `yolov11n_obb_*`).
