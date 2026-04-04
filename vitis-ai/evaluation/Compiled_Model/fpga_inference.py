from __future__ import annotations

import os
import sys

from fpga_inference_obb import run_fpga_inference

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python fpga_inference.py <model_path> <test_data_path> <output_npz_path> <img_size>")
        print("\nArguments:")
        print("  model_path      : Path to .xmodel file")
        print("  test_data_path  : Path to folder containing test images")
        print("  output_npz_path : Output NPZ file path (e.g., predictions.npz)")
        print("  img_size        : Image size (single value for square images, e.g., 416)")
        print("\nExample:")
        print("  python fpga_inference.py model/yolov8n.xmodel ./images predictions.npz 416")
        sys.exit(1)

    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_npz_path = sys.argv[3]
    img_size = int(sys.argv[4])

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data path not found: {test_data_path}")
        sys.exit(1)

    run_fpga_inference(
        model_path,
        test_data_path,
        output_npz_path,
        img_height=img_size,
        img_width=img_size,
        obb=False,
    )
