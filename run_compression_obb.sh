#!/bin/bash

# Configuration
MODEL_PATH="./best.pt"
VERSION="n"
NUM_CLASSES=1
ACTIVATION="relu"
TASK="obb"
TARGET="DPUCZDX8G_ISA1_B4096"
IMG_SIZE=416

echo "========================================"
echo "YOLOv11-OBB DPU Quantization Workflow"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Version: YOLOv11-$VERSION ($TASK)"
echo "Classes: $NUM_CLASSES"
echo "Activation: $ACTIVATION"
echo "Target DPU: $TARGET"
echo "Input Size: ${IMG_SIZE}x${IMG_SIZE}"
echo "========================================"
echo ""

# Step 1: Calibration
echo "Step 1/3: Running calibration..."
python vai_quantize.py \
    --model_path "$MODEL_PATH" \
    --version "$VERSION" \
    --num_classes $NUM_CLASSES \
    --activation "$ACTIVATION" \
    --task "$TASK" \
    --batch_size 16 \
    --img_height $IMG_SIZE \
    --img_width $IMG_SIZE \
    --target "$TARGET" \
    --quant_mode calib

if [ $? -ne 0 ]; then
    echo "❌ Calibration failed!"
    exit 1
fi

echo "✓ Calibration complete"
sleep 5

# Step 2: Test quantized model
echo ""
echo "Step 2/3: Testing quantized model..."
python vai_quantize.py \
    --model_path "$MODEL_PATH" \
    --version "$VERSION" \
    --num_classes $NUM_CLASSES \
    --activation "$ACTIVATION" \
    --task "$TASK" \
    --batch_size 1 \
    --img_height $IMG_SIZE \
    --img_width $IMG_SIZE \
    --target "$TARGET" \
    --quant_mode test

if [ $? -ne 0 ]; then
    echo "❌ Testing failed!"
    exit 1
fi

echo "✓ Testing complete"
sleep 5

# Step 3: Export xmodel
echo ""
echo "Step 3/3: Exporting xmodel for deployment..."
python vai_quantize.py \
    --model_path "$MODEL_PATH" \
    --version "$VERSION" \
    --num_classes $NUM_CLASSES \
    --activation "$ACTIVATION" \
    --task "$TASK" \
    --batch_size 1 \
    --subset_len 1 \
    --img_height $IMG_SIZE \
    --img_width $IMG_SIZE \
    --target "$TARGET" \
    --quant_mode test \
    --deploy

if [ $? -ne 0 ]; then
    echo "❌ Deployment export failed!"
    exit 1
fi

echo "✓ Deployment export complete"
sleep 2

# Rename quantize_result directory with model info
RESULT_DIR="quantize_result_yolov11${VERSION}_obb_${IMG_SIZE}_${TARGET##*_}"
echo ""
echo "Organizing results..."
if [ -d "quantize_result" ]; then
    if [ -d "$RESULT_DIR" ]; then
        echo "⚠ Directory $RESULT_DIR exists, removing old version..."
        rm -rf "$RESULT_DIR"
    fi
    mv quantize_result "$RESULT_DIR"
    echo "✓ Results saved to: $RESULT_DIR"
else
    echo "⚠ quantize_result directory not found"
fi
