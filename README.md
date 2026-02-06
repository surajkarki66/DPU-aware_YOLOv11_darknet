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
- [Quantization &amp; Deployment](#quantization--deployment)
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

2. Install dependencies:

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
    └── val2017/
```

2. Generate dataset file lists:

```bash
python format_dataset.py
```

This creates text files with image paths for train/val/test splits.

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

- [`data/hyps/args.yaml`](data/hyps/args.yaml): Full COCO dataset (80 classes)
- [`data/hyps/args_hpd.yaml`](data/hyps/args_hpd.yaml): Custom dataset configuration

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

1. **Calibration**: Quantize model using calibration dataset
2. **Testing**: Validate quantized model accuracy
3. **Compilation**: Generate DPU-ready `.xmodel` file

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

#### Step 3: Compilation

Navigate to compilation directory:

```bash
cd vitis-ai/compilation
bash run_compile.sh
```

The compilation script:

- Converts quantized model to DPU-compatible format
- Optimizes for specific DPU architecture
- Generates `.xmodel` file for deployment

Compiled models are saved in `vitis-ai/compilation/models/`

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
├── vitis-ai/                    # Vitis-AI deployment
│   ├── compilation/            # Model compilation
│   │   ├── Architectures/      # DPU architecture JSONs
│   │   ├── models/             # Compiled models
│   │   └── run_compile.sh      # Compilation script
│   └── evaluation/             # Post-deployment evaluation
│       ├── coco_prep.py        # COCO evaluation prep
│       ├── evaluate.py         # Accuracy evaluation
│       └── post_processing.py  # DPU output processing
├── train.py                     # Training script
├── inference.py                 # Inference script
├── vai_quantize.py             # Vitis-AI quantization
├── model_profile.py            # Model profiling
├── format_dataset.py           # Dataset preparation
├── prepare_calibration_data.py # Calibration data prep
├── run_compression.sh          # Automated workflow
└── requirements.txt            # Python dependencies
```

## 🖥️ DPU Architectures

Pre-configured DPU architectures are available in [`vitis-ai/compilation/Architectures/`](vitis-ai/compilation/Architectures/):

| Architecture    | Description      | RAM Size |
| --------------- | ---------------- | -------- |
| arch_B512.json  | Smallest DPU     | 512 KB   |
| arch_B800.json  | Small DPU        | 800 KB   |
| arch_B1024.json | Medium-Small DPU | 1 MB     |
| arch_B1152.json | Medium DPU       | 1.125 MB |
| arch_B1600.json | Medium-Large DPU | 1.56 MB  |
| arch_B2304.json | Large DPU        | 2.25 MB  |
| arch_B3136.json | Very Large DPU   | 3.06 MB  |
| arch_B4096.json | Largest DPU      | 4 MB     |

Select the appropriate architecture based on your FPGA resources and performance requirements.

## 🎓 Model Variants

| Variant  | Width Multiplier | Depth Multiplier | Parameters | Use Case                |
| -------- | ---------------- | ---------------- | ---------- | ----------------------- |
| YOLOv11n | Smallest         | Minimal          | Lowest     | Edge devices, real-time |
| YOLOv11s | Small            | Minimal          | Low        | Balanced speed/accuracy |
| YOLOv11m | Medium           | Medium           | Medium     | General purpose         |
| YOLOv11l | Large            | Deep             | High       | High accuracy           |
| YOLOv11x | Largest          | Deep             | Highest    | Maximum accuracy        |

## 📝 Notes

- **Activation Functions**: ReLU is recommended for DPU deployment as it's more hardware-friendly than SiLU
- **Input Size**: Common sizes are 416x416 or 640x640. Smaller sizes improve inference speed
- **Batch Size**: Adjust based on GPU memory during training
- **Quantization**: Post-training quantization converts FP32 models to INT8 for DPU deployment

## 🐛 Troubleshooting

### Training Issues

- Ensure dataset paths in YAML config files are correct
- Check CUDA availability for GPU training
- Adjust batch size if running out of memory

### Quantization Issues

- Verify calibration dataset is properly prepared
- Ensure model checkpoint loads correctly
- Check DPU target architecture matches your hardware

### Compilation Issues

- Confirm Vitis-AI environment is properly set up
- Verify `.xmodel` output path permissions
- Check DPU architecture JSON file is valid

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Ultralytics YOLOv11 architecture
- Xilinx Vitis-AI toolkit
- PyTorch framework

## 📧 Contact

For questions or issues, please open an issue in the repository.
