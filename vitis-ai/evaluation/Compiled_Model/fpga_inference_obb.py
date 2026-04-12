import cv2
import json
import numpy as np
import vart
import xir
import os
import sys
import time
from glob import glob


def get_child_subgraph_dpu(graph: "Graph"):
    """Extract DPU subgraph from XIR graph."""
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def preprocess_image(image, input_scale, width=416, height=416):
    """
    Preprocess frame for YOLO input using letterbox:
      - Aspect-ratio preserving resize
      - Gray (114, 114, 114) padding to reach exact target dimensions
      - BGR -> RGB, normalize to [0, 1], quantize to int8 for DPU

    Returns:
        padded (np.int8) : preprocessed image ready for DPU
        meta   (dict)    : gain, pad_w, pad_h, h0, w0 — needed for
                           inverse transform in eval_npz.py
    """
    h0, w0 = image.shape[:2]

    # Uniform scale that fits the image inside (width x height) without cropping
    r = min(width / w0, height / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

    # Aspect-ratio preserving resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Symmetric gray padding to reach target size
    pad_w = (width - new_w) / 2.0
    pad_h = (height - new_h) / 2.0
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # BGR -> RGB, normalize, quantize
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32) / 255.0
    padded = (padded * input_scale).astype(np.int8)

    meta = {
        "h0": h0,
        "w0": w0,
        "gain": r,
        "pad_w": left,
        "pad_h": top,
    }
    return padded, meta


# YOLOv11-OBB: 6 outputs (3 detection + 3 angle)
OBB_NUM_OUTPUTS = 6
OBB_REG_MAX = 16
OBB_STRIDES = [8, 16, 32]


def run_fpga_inference(model_path, test_data_path, output_npz_path, img_height=416, img_width=416, obb=False):
    """
    Run Vitis AI model on all test images and save raw outputs.
    
    For YOLOv11-OBB: model must have 6 outputs — outputs 0,1,2 = detection heads,
    outputs 3,4,5 = angle heads (same order as training export in test.py / inference.py).
    
    Args:
        model_path: Path to .xmodel file
        test_data_path: Path to folder containing test images
        output_npz_path: Path to save compressed NPZ file with predictions
        img_height: Input image height
        img_width: Input image width
        obb: If True, use YOLOv11-OBB format (6 outputs: 3 detect + 3 angle)
    """
    
    print(f"Loading model from: {model_path}")
    
    # Setup model
    g = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(g)
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")
    
    # Get input tensor info
    inputTensors = dpu_runner.get_input_tensors()
    input_shape = tuple(inputTensors[0].dims)
    input_fixpos = inputTensors[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos
    
    # Get output tensor info
    outputTensors = dpu_runner.get_output_tensors()
    output_shapes = [tuple(tensor.dims) for tensor in outputTensors]
    output_fixpoints = [tensor.get_attr("fix_point") for tensor in outputTensors]
    
    if obb and len(output_shapes) != OBB_NUM_OUTPUTS:
        print(f"ERROR: YOLOv11-OBB expects {OBB_NUM_OUTPUTS} outputs (3 detect + 3 angle), got {len(output_shapes)}")
        sys.exit(1)
    
    print(f"Input shape: {input_shape}")
    print(f"Input fix_point: {input_fixpos}")
    print(f"Output shapes: {output_shapes}")
    print(f"Output fix_points: {output_fixpoints}")
    if obb:
        print(f"Model type: YOLOv11-OBB (outputs 0,1,2=detect, 3,4,5=angle)")
    print(f"Preprocessing: Letterbox (aspect-ratio preserving + gray pad 114)")
    
    # Get all image files
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    all_files = sorted(glob(os.path.join(test_data_path, "*.*")))
    image_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
    
    if len(image_paths) == 0:
        print(f"ERROR: No images found in {test_data_path}")
        return
    
    print(f"\nFound {len(image_paths)} images to process")
    print(f"Processing images at {img_width}x{img_height}...")
    
    # Storage for all predictions
    all_image_names = []
    all_predictions = [] 
    all_metas = []  # Store letterbox metadata for each image
    
    total_inference_time = 0
    overall_start_time = time.time()
    
    # Process each image
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: Failed to load {img_name}, skipping...")
            continue
        
        # Preprocess with letterbox
        processed_image, meta = preprocess_image(image, input_scale, width=img_width, height=img_height)
        
        # Prepare input data
        input_data = [np.empty(input_shape, dtype=np.int8, order="C")]
        input_data[0][0, ...] = processed_image.reshape(input_shape[1:])
        
        # Prepare output data
        output_data = []
        for outputTensor in outputTensors:
            output_shape = tuple(outputTensor.dims)
            output_data.append(np.empty(output_shape, dtype=np.int8, order="C"))
        
        # Run inference
        start_time = time.time()
        job_id = dpu_runner.execute_async(input_data, output_data)
        dpu_runner.wait(job_id)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Store results (keep as int8 to save space)
        all_image_names.append(img_name)
        all_predictions.append([out.copy() for out in output_data])
        all_metas.append(meta)
        
        # Progress update
        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            avg_time = total_inference_time / (idx + 1)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Processed {idx + 1}/{len(image_paths)} images | "
                  f"Avg: {avg_time*1000:.2f}ms | FPS: {fps:.1f}")
    
    total_processing_time = time.time() - overall_start_time
    num_images = len(all_predictions)
    
    if num_images == 0:
        print("ERROR: No images were successfully processed.")
        return
    
    # Save predictions to compressed NPZ file
    # Format aligned with OBB export (test.py / inference.py): output_0,1,2 = detect, output_3,4,5 = angle
    print(f"\nSaving predictions to {output_npz_path}...")
    save_dict = {
        'image_names': np.array(all_image_names),
        'output_shapes': np.array(output_shapes),
        'output_fixpoints': np.array(output_fixpoints),
        'num_outputs': len(output_shapes),
        'model_type': 'obb' if obb else 'detect',
        'img_height': np.int32(img_height),
        'img_width': np.int32(img_width),
        # Store letterbox metadata as JSON strings for each image
        'image_metas': np.array([json.dumps(m) for m in all_metas]),
    }
    if obb:
        save_dict['reg_max'] = np.int32(OBB_REG_MAX)
        save_dict['strides'] = np.array(OBB_STRIDES, dtype=np.int32)

    # Add predictions for each image (same key layout as before for compatibility)
    for img_idx, pred_list in enumerate(all_predictions):
        for out_idx, pred in enumerate(pred_list):
            save_dict[f'pred_{img_idx}_output_{out_idx}'] = pred

    np.savez_compressed(output_npz_path, **save_dict)
    
    # Get file size
    file_size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FPGA INFERENCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {num_images}")
    print(f"")
    print(f"Preprocessing: Letterbox (aspect-ratio preserving + gray pad 114)")
    print(f"")
    print(f"🚀 DPU INFERENCE:")
    print(f"   Total time: {total_inference_time:.2f}s")
    print(f"   Average per image: {total_inference_time/num_images*1000:.2f}ms")
    print(f"   DPU FPS: {num_images/total_inference_time:.1f}")
    print(f"")
    print(f"⏱️  OVERALL PERFORMANCE:")
    print(f"   Total time: {total_processing_time:.2f}s")
    print(f"   Average per image: {total_processing_time/num_images*1000:.2f}ms")
    print(f"   Overall FPS: {num_images/total_processing_time:.1f}")
    print(f"")
    print(f"💾 OUTPUT FILE:")
    print(f"   File: {output_npz_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Format: Compressed NPZ (int8 predictions + letterbox metadata)")
    if obb:
        print(f"   Model: YOLOv11-OBB (outputs 0,1,2=detect, 3,4,5=angle)")
    print(f"{'='*70}\n")

    # Save FPS statistics to JSON (FPGA/DPU only)
    fps_stats_path = output_npz_path.replace(".npz", "_fps.json") if output_npz_path.endswith(".npz") else output_npz_path + "_fps.json"
    fps_stats = {
        "num_images": num_images,
        "preprocessing": "letterbox",
        "dpu_inference": {
            "total_time_s": round(total_inference_time, 4),
            "avg_per_image_ms": round(total_inference_time / num_images * 1000, 2),
            "fps": round(num_images / total_inference_time, 2),
        },
    }
    with open(fps_stats_path, "w") as f:
        json.dump(fps_stats, f, indent=2)
    print(f"FPS statistics saved to: {fps_stats_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run YOLOv11 / YOLOv11-OBB on FPGA and save raw outputs to NPZ with letterbox preprocessing")
    parser.add_argument("model_path", help="Path to .xmodel file")
    parser.add_argument("test_data_path", help="Path to folder containing test images")
    parser.add_argument("output_npz_path", help="Output NPZ file path (e.g., predictions.npz)")
    parser.add_argument("img_size", type=int, help="Image size (single value for square images, e.g., 416)")
    parser.add_argument("--obb", action="store_true", help="YOLOv11-OBB model (6 outputs: 3 detect + 3 angle)")
    args = parser.parse_args()

    model_path = args.model_path
    test_data_path = args.test_data_path
    output_npz_path = args.output_npz_path
    img_size = args.img_size

    # Validate inputs
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data path not found: {test_data_path}")
        sys.exit(1)

    # Run inference
    run_fpga_inference(
        model_path,
        test_data_path,
        output_npz_path,
        img_height=img_size,
        img_width=img_size,
        obb=args.obb,
    )
