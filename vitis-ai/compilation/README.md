# DPU compilation (`vai_c_xir`)

After quantization at the repository root, Vitis-AI produces an **INT8 `.xmodel`** (and related artifacts under `quantize_result/` or a renamed folder from `run_compression.sh`). This folder compiles that XIR model for a **specific DPU RAM size** using Xilinx `vai_c_xir`.

## What you need

1. **Exported XIR model**: e.g. `YOLO_int_<target_suffix>.xmodel` or `YOLO_OBB_int_<target_suffix>.xmodel`, depending on whether you used `--task detect` or `--task obb` in `vai_quantize.py`. Place it under:

   ```
   vitis-ai/compilation/models/
   ```

   The exact filename must match what you reference in `run_compile.sh` (see below).

2. **Architecture JSON**: One of the files in [`Architectures/`](Architectures/) that matches your DPU’s on-chip RAM (B512 … B4096). Larger values correspond to more BRAM and typically allow larger feature maps.

3. **Environment**: Shell with `vai_c_xir` on `PATH` (Vitis-AI compiler for your card / platform).

## `run_compile.sh`

The script contains **commented** example `vai_c_xir` invocations. For each DPU size block, two lines are provided:

- **Detection**: `YOLO_int_B*.xmodel` → output under `zynq_output/yolov11n_B*/`
- **OBB**: `YOLO_OBB_int_B*.xmodel` → output under `zynq_output/yolov11n_obb_B*/`

### Steps

1. `cd` to this directory:

   ```bash
   cd vitis-ai/compilation
   ```

2. Copy or symlink your compiled `.xmodel` into `models/` with the name expected by the line you will uncomment (e.g. `models/YOLO_int_B4096.xmodel`).

3. Uncomment **one** pair of lines for your target RAM (or edit paths/names to match your files).

4. Run:

   ```bash
   bash run_compile.sh
   ```

### `vai_c_xir` arguments (as used in the script)

| Flag | Meaning |
|------|---------|
| `-x` | Input `.xmodel` file |
| `-a` | Architecture JSON (DPU RAM / topology) |
| `-o` | Output directory for the compiled DPU model |
| `-n` | Name prefix for generated artifacts |

Outputs are written under `zynq_output/` (created relative to the current working directory when you run the script).

## Architecture files

JSON files in [`Architectures/`](Architectures/) describe DPU configurations (e.g. `arch_B4096.json`). Choose the file that matches your deployment platform documentation. A summary table is also in the [root README](../../README.md#-dpu-architectures).

## Related repository scripts

- Quantization and `.xmodel` export: [`vai_quantize.py`](../../vai_quantize.py) from the repo root (`--quant_mode test` with `--deploy`).
- End-to-end quant + export helper: [`run_compression.sh`](../../run_compression.sh).

If compilation fails, verify the `.xmodel` target string used during quantization matches your hardware, and that the architecture JSON matches your DPU binary.
