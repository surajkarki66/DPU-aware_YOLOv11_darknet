"""
Post-process YOLOv11 ONNX model outputs to bounding boxes and class scores.
Usage: python post_processing.py <path_to_output> <model_name> <iou_threshold> <img_height> <img_width>
"""
import onnxruntime as ort
import torch
import numpy as np
import pickle
import re
import os
import sys
import time

from glob import glob
from PIL import Image


def apply_dfl(box_preds, reg_max=16):
    """
    Apply Distribution Focal Loss operation for YOLOv11.
    
    Args:
        box_preds: Tensor of shape [batch, channels, anchors]
        reg_max: DFL channels (ch parameter from model config)
    
    Returns:
        Tensor of shape [batch, 4, anchors] with box coordinates
    """
    b, _, a = box_preds.shape  # batch, channels, anchors
    # Reshape to [b, 4, reg_max, a]
    box_preds = box_preds.view(b, 4, reg_max, a)
    # Transpose to [b, 4, a, reg_max] for softmax
    box_preds = box_preds.transpose(2, 3)
    # Apply softmax over reg_max dimension
    box_preds = box_preds.softmax(3)
    # Create weight tensor [0, 1, 2, ..., reg_max-1]
    weights = torch.arange(reg_max, dtype=box_preds.dtype, device=box_preds.device)
    weights = weights.view(1, 1, 1, reg_max)
    # Weighted sum over the last dimension
    box_preds = (box_preds * weights).sum(3)  # [b, 4, a]
    return box_preds


def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))
    except Exception as e:
        print(f"WARNING !! failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def check_version(current: str = "0.0.0", required: str = "0.0.0", name: str = "version", 
                  hard: bool = False, verbose: bool = False, msg: str = "") -> bool:
    from importlib import metadata
    if not current:
        print(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError((f"WARNING !! {current} package is required but not installed")) from e
            else:
                return False

    if not required:
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()
        v = parse_version(version)
        if op == "==" and c != v: result = False
        elif op == "!=" and c == v: result = False
        elif op in {">=", ""} and not (c >= v): result = False
        elif op == "<=" and not (c <= v): result = False
        elif op == ">" and not (c > v): result = False
        elif op == "<" and not (c < v): result = False
    if not result:
        warning = f"WARNING !! {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard: raise ModuleNotFoundError(warning)
        if verbose: print(warning)
    return result


TORCH_1_10 = check_version(torch.__version__, "1.10.0")


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points for YOLOv11."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert distance predictions to bounding boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def decode_bboxes(bboxes, anchors):
    """Decode bounding boxes from distance format."""
    return dist2bbox(bboxes, anchors, xywh=True, dim=1)


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2).
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=1,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
):
    """Perform non-maximum suppression (NMS) on YOLOv11 predictions."""
    import torchvision

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    prediction = prediction.transpose(-1, -2)  # [B, C, N] → [B, N, C]
    if in_place:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    else:
        prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = x[:, :4] + c
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output


def load_model_config(config_path):
    """
    Load YOLOv11 model configuration from pkl file.
    
    Args:
        config_path: Path to the config pkl file
        
    Returns:
        tuple: (tensor_no, tensor_stride, tensor_ch, tensor_nc)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        tensor_no, tensor_stride, tensor_ch, tensor_nc = pickle.load(f)
    
    return tensor_no, tensor_stride, tensor_ch, tensor_nc


def run_model_config(x, config_path):
    """
    Apply YOLOv11 post-processing to raw model outputs.
    
    Args:
        x: List of output tensors from the model (multi-scale predictions)
        config_path: Path to model config pkl file
        
    Returns:
        Processed predictions ready for NMS
    """
    shape = x[0].shape
    tensor_no, tensor_stride, tensor_ch, tensor_nc = load_model_config(config_path)

    # Concatenate multi-scale outputs
    x_cat = torch.cat([xi.view(shape[0], tensor_no, -1) for xi in x], 2)
    
    # Generate anchors and strides
    tensor_anchors, tensor_strides = (x.transpose(0, 1) for x in make_anchors(x, tensor_stride, 0.5))
    
    # Split into box predictions and class predictions
    box, cls = x_cat.split((tensor_ch * 4, tensor_nc), 1)
    
    # Apply DFL to box predictions and decode
    dbox = decode_bboxes(apply_dfl(box, tensor_ch), tensor_anchors.unsqueeze(0)) * tensor_strides
    
    # Combine boxes and class scores
    y = torch.cat((dbox, cls.sigmoid()), 1)
    
    return y


def res_exp(pred, config_path, iou):
    """Process predictions and write results."""
    global f, timing
    start = time.time()

    # Apply post-processing
    predi = run_model_config(pred, config_path)
    
    # Get number of classes from config
    with open(config_path, 'rb') as cf:
        _, _, _, tensor_nc = pickle.load(cf)
    
    # Apply NMS
    box_conf = non_max_suppression(prediction=predi, conf_thres=0.001, iou_thres=iou, nc=tensor_nc)
    box_conf[0] = box_conf[0].detach().numpy()
    box_conf[0] = np.delete(box_conf[0], 5, axis=1)
    box_conf[0][box_conf[0] < 0] = 0

    # Convert to [x, y, w, h, conf] format
    fixed = [[i[0], i[1], i[2]-i[0], i[3]-i[1], i[4]] for i in box_conf[0]]

    end = time.time() - start
    timing += end

    f.write("Boxes:\n\n")
    f.write("\n".join(" ".join(map(str, x)) for x in np.array(fixed)))
    f.write("\n\n")


def box_processing_writing(dictionary, config_path, iou):
    """Process and write all predictions."""
    global f
    for named, tensor in zip(dictionary["names"], dictionary["tensors"]):
        f.write(f"Image Prediction: {named}\n\n")
        res_exp(tensor, config_path, iou)


def preprocess_image(img_path, size=(640, 640)):
    """Preprocess image for ONNX inference."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


def tensor_fix(pth, model_name, img_size=(640, 640)):
    """Run ONNX inference on all test images."""
    onnx_path = os.path.join(pth, f"./{model_name}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    print(f"Loading ONNX model from: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"Model inputs: {input_name}")
    print(f"Model outputs: {output_names}")

    tensor_index = {"names": [], "tensors": []}

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    test_folder = os.path.join(pth, "test_data")
    all_files = sorted(glob(os.path.join(test_folder, "*.*")))
    image_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
    
    print(f"Found {len(image_paths)} test images in {test_folder}")

    for img_path in image_paths:
        img_input = preprocess_image(img_path, size=img_size)
        outputs = session.run(output_names, {input_name: img_input})
        tensor_index["names"].append(os.path.basename(img_path))
        tensor_index["tensors"].append(tuple(torch.from_numpy(out) for out in outputs))

    return tensor_index


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python post_processing.py <path_to_output> <model_name> <iou_threshold> <img_height> <img_width>")
        print("Example: python post_processing.py ./quantize_result best_state_dict 0.45 640 640")
        sys.exit(1)

    timing = 0
    pth = sys.argv[1]
    name = sys.argv[2]
    iou = float(sys.argv[3])
    img_height = int(sys.argv[4])
    img_width = int(sys.argv[5])
    img_size = (img_width, img_height)

    # Load config path
    config_path = os.path.join(pth, f"{name}_config.pkl")
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print(f"Make sure you have run quantization with calibration mode first.")
        sys.exit(1)

    print("\n" + "="*60)
    print("YOLOv11 ONNX Post-Processing")
    print("="*60)
    print(f"Output path: {pth}")
    print(f"Model name: {name}")
    print(f"Config file: {config_path}")
    print(f"IOU threshold: {iou}")
    print(f"Image size: {img_size}")
    print("="*60 + "\n")

    f = open(f"output_boxes_{name}.txt", 'w')

    x_dict = tensor_fix(pth, name, img_size=img_size)
    box_processing_writing(x_dict, config_path, iou)
    
    f.close()

    print(f'\n✓ Post-processing complete!')
    print(f'Total time: {timing:.3f}s')
    print(f'Average per image: {timing/len(x_dict["names"]):.3f}s')
    print(f'Results saved to: output_boxes_{name}.txt\n')
