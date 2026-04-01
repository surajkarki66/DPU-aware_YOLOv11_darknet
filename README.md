# Hardware-Aware YOLOv11 for Xilinx DPU

A PyTorch implementation of YOLOv11 optimized for deployment on Xilinx FPGA Deep Learning Processing Units (DPU) using Vitis-AI. This project provides a complete workflow from training to deployment, including DPU-aware model design, quantization, and compilation.

## 🌟 Features

- **Multiple YOLOv11 Variants**: Support for YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, and YOLOv11x
- **DPU-Optimized Architecture**: Hardware-aware design compatible with Xilinx DPU constraints
- **Flexible Activation Functions**: Support for SiLU (default) and ReLU activation functions
- **Vitis-AI Integration**: Complete quantization and compilation workflow
- **Multiple DPU Targets**: Pre-configured architectures for various DPU configurations (B512 to B4096)
- **Training & Inference**: Full pipeline from training to deployment
- **COCO Dataset Support**: Built-in support for COCO dataset format

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Profiling](#model-profiling)
- [Quantization & Deployment](#quantization--deployment)
- [Project Structure](#project-structure)
- [DPU Architectures](#dpu-architectures)

## 🔧 Requirements

- Python 3.7+
- CUDA-capable GPU (for training)
- Vitis-AI environment (for quantization and compilation)

## 📦 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd HW-aware-YOLOv11
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch 1.13.1
- torchvision 0.14.1
- OpenCV 4.7.0
- NumPy < 2.0.0
- PyYAML 6.0
- TQDM 4.65.0
- THOP (for FLOPs calculation)
- Matplotlib & Seaborn (for visualization)
- TensorBoard 2.13.0

## 📊 Dataset Preparation

### COCO Dataset Format

1. Organize your dataset in COCO format:

```
coco_data/
├── images/
│   ├── train2017/
│   ├── val2017/
│   └── test2017/
└── labels/
    ├── train2017/
    ├── val2017/
    └── test2017/
```

1. Generate dataset file lists (writes `train2017.txt`, `val2017.txt`, `test2017.txt` next to `images/`):

```bash
python format_dataset.py --base-dir /path/to/coco_data
```

The default base directory is `./kitti_COCO` if `--base-dir` is omitted.

### Prepare Calibration Data

For quantization, prepare calibration data:

```bash
python prepare_calibration_data.py
```

## 🎯 Training

Train a YOLOv11 model with custom configuration:

```bash
python train.py \
    --version n \
    --activation relu \
    --img_size 416 \
    --batch_size 32 \
    --epochs 300 \
    --data_config data/hyps/args.yaml
```

### Training Arguments

- `--version`: YOLOv11 variant (`n`, `s`, `m`, `l`, `x`)
- `--activation`: Activation function (`silu` or `relu`)
- `--img_size`: Input image size (default: 640)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--data_config`: Path to dataset/hyperparameter configuration

### Configuration Files

Hyperparameters and dataset settings are defined in YAML files:

- `[data/hyps/args.yaml](data/hyps/args.yaml)`: Full COCO dataset (80 classes)
- `[data/hyps/args_hpd.yaml](data/hyps/args_hpd.yaml)`: Custom dataset configuration

Training outputs are saved to `runs/` directory.

## 🔍 Inference

Run inference on images or videos:

### Single Image

```bash
python inference.py \
    --model_path runs/best_state_dict.pt \
    --version n \
    --activation relu \
    --source image \
    --source_path path/to/image.jpg \
    --input_size 416
```

### Video/Camera

```bash
python inference.py \
    --model_path runs/best_state_dict.pt \
    --version n \
    --activation relu \
    --source video \
    --source_path path/to/video.mp4 \
    --input_size 416
```

### Inference Arguments

- `--model_path`: Path to trained model checkpoint
- `--version`: YOLOv11 variant
- `--num_classes`: Number of object classes
- `--activation`: Activation function used in the model
- `--source`: Input source type (`image`, `video`, or `camera`)
- `--source_path`: Path to input file
- `--input_size`: Model input size
- `--conf_threshold`: Confidence threshold (default: 0.25)
- `--iou_threshold`: NMS IoU threshold (default: 0.45)

## 📈 Model Profiling

Analyze model complexity and performance:

```bash
python model_profile.py \
    --version n \
    --activation relu \
    --num_classes 80 \
    --img_size 416
```

This tool provides:

- Parameter count
- FLOPs (Floating Point Operations)
- Model architecture summary
- Layer-wise analysis

## 🚀 Quantization & Deployment

### Automated Workflow

Use the provided shell script for complete quantization workflow:

```bash
bash run_compression.sh
```

This script performs:

1. **Calibration** (`--quant_mode calib`): Run calibration on your dataset
2. **Testing** (`--quant_mode test`): Validate the quantized model
3. **Deploy export** (`--quant_mode test --deploy`): Export a DPU-ready `.xmodel` (and related artifacts) under `quantize_result/` (renamed by the script)

DPU **bitstream-side compilation** (`vai_c_xir`) is a separate step; see Step 4 below and `[vitis-ai/compilation/README.md](vitis-ai/compilation/README.md)`.

### Manual Quantization

#### Step 1: Calibration

```bash
python vai_quantize.py \
    --model_path runs/best_state_dict.pt \
    --version n \
    --num_classes 1 \
    --activation relu \
    --batch_size 16 \
    --img_height 416 \
    --img_width 416 \
    --target DPUCZDX8G_ISA1_B4096 \
    --quant_mode calib
```

#### Step 2: Test Quantized Model

```bash
python vai_quantize.py \
    --model_path runs/best_state_dict.pt \
    --version n \
    --num_classes 1 \
    --activation relu \
    --batch_size 16 \
    --img_height 416 \
    --img_width 416 \
    --target DPUCZDX8G_ISA1_B4096 \
    --quant_mode test
```

#### Step 3: Export `.xmodel` (deploy)

From the repo root, run test mode with `--deploy` (typically `batch_size 1`, `subset_len 1` as in `run_compression.sh`):

```bash
python vai_quantize.py \
    --model_path runs/best_state_dict.pt \
    --version n \
    --num_classes 1 \
    --activation relu \
    --batch_size 1 \
    --subset_len 1 \
    --img_height 416 \
    --img_width 416 \
    --target DPUCZDX8G_ISA1_B4096 \
    --quant_mode test \
    --deploy
```

This produces the quantized `.xmodel` and supporting files (e.g. under `quantize_result/`).

#### Step 4: DPU compilation (vai_c_xir)

Copy the exported `.xmodel` into `vitis-ai/compilation/models/` (name must match what you pass to `vai_c_xir`), pick the matching `Architectures/arch_B*.json`, uncomment the right command in `run_compile.sh`, then:

```bash
cd vitis-ai/compilation
bash run_compile.sh
```

This runs the Xilinx compiler and writes platform-specific outputs under `vitis-ai/compilation/zynq_output/` (see `[vitis-ai/compilation/README.md](vitis-ai/compilation/README.md)`). For OBB models, use `run_compression_obb.sh` for the quant workflow and the OBB lines in `run_compile.sh`.

## 🏗️ Project Structure

```
HW-aware-YOLOv11/
├── models/                      # Model architectures
│   ├── common.py               # Common building blocks
│   └── yolo.py                 # YOLOv11 variants
├── utils/                       # Utility functions
│   ├── dataset.py              # Dataset loader
│   ├── plotting.py             # Visualization tools
│   └── util.py                 # Helper functions
├── data/                        # Data configurations
│   └── hyps/                   # Hyperparameter configs
├── vitis-ai/                    # Vitis-AI deployment (see vitis-ai/README.md)
│   ├── README.md
│   ├── compilation/           # vai_c_xir: .xmodel → DPU package
│   │   ├── Architectures/       # DPU architecture JSONs (B512–B4096)
│   │   ├── models/              # Place exported .xmodels for run_compile.sh
│   │   ├── zynq_output/         # Compiler output (after run_compile.sh)
│   │   ├── run_compile.sh
│   │   └── README.md
│   └── evaluation/            # Post-quantization metrics
│       ├── README.md
│       ├── Detect/            # Axis-aligned: post_processing, coco_prep, evaluate
│       └── OBB/               # Oriented boxes: post_processing_obb, evaluate_obb
├── train.py                     # Training script
├── inference.py                 # Inference script
├── vai_quantize.py             # Vitis-AI quantization
├── model_profile.py            # Model profiling
├── format_dataset.py           # Dataset path lists (train/val/test .txt)
├── prepare_calibration_data.py # Calibration data prep
├── run_compression.sh          # Detect: calib → test → deploy export
├── run_compression_obb.sh      # OBB: same pipeline for oriented model
└── requirements.txt            # Python dependencies
```

## 🖥️ DPU Architectures

Pre-configured DPU architectures are available in [`vitis-ai/compilation/Architectures/`](vitis-ai/compilation/Architectures/):


| Architecture    | RAM Size |
| --------------- | -------- |
| arch_B512.json  | 512 KB   |
| arch_B800.json  | 800 KB   |
| arch_B1024.json | 1 MB     |
| arch_B1152.json | 1.125 MB |
| arch_B1600.json | 1.56 MB  |
| arch_B2304.json | 2.25 MB  |
| arch_B3136.json | 3.06 MB  |
| arch_B4096.json | 4 MB     |


Select the appropriate architecture based on your FPGA resources and performance requirements.

## 🎓 Model Variants


| Variant  | Width Multiplier | Depth Multiplier | Parameters |
| -------- | ---------------- | ---------------- | ---------- |
| YOLOv11n | Smallest         | Minimal          | Lowest     |
| YOLOv11s | Small            | Minimal          | Low        |
| YOLOv11m | Medium           | Medium           | Medium     |
| YOLOv11l | Large            | Deep             | High       |
| YOLOv11x | Largest          | Deep             | Highest    |


## 📝 Notes

- **Activation Functions**: ReLU is recommended for DPU deployment as it's more hardware-friendly than SiLU
- **Input Size**: Common sizes are 416x416 or 640x640. Smaller sizes improve inference speed
- **Batch Size**: Adjust based on GPU memory during training
- **Quantization**: Post-training quantization converts FP32 models to INT8 for DPU deployment

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Ultralytics YOLOv11 architecture
- Xilinx Vitis-AI toolkit
- PyTorch framework

## 📧 Contact

For questions or issues, please open an issue in the repository.