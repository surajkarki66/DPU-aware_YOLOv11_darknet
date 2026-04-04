from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_QM = Path(__file__).resolve().parent.parent / "Quantized_Model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

from eval_core import (  # noqa: E402
    DetectMapEvaluator,
    DetectYOLO,
    OBB26YOLO,
    OBBMapEvaluator,
    apply_quant_meta,
    iter_images,
    print_and_save_metrics,
    resolve_source_from_data,
    stretch_meta,
)


def _fixpoints_list(arr: np.ndarray) -> list[int]:
    flat = np.asarray(arr).reshape(-1)
    return [int(x) for x in flat.tolist()]


def _shapes_list(arr: np.ndarray) -> list[tuple[int, ...]]:
    a = np.asarray(arr, dtype=object)
    if a.ndim == 0:
        t = a.item()
        return [tuple(int(x) for x in t)]
    out: list[tuple[int, ...]] = []
    for item in a.tolist():
        if isinstance(item, (list, tuple)):
            out.append(tuple(int(x) for x in item))
        else:
            raise ValueError(f"Unexpected output_shapes entry: {item!r}")
    return out


def dequantize_int8_outputs(tensors: list[np.ndarray], fixpoints: list[int]) -> list[np.ndarray]:
    if len(tensors) != len(fixpoints):
        raise ValueError(f"Got {len(tensors)} tensors but {len(fixpoints)} fixpoints")
    scale = [float(2.0 ** (-fp)) for fp in fixpoints]
    return [t.astype(np.float32) * s for t, s in zip(tensors, scale)]


def _classify_obb_output_shape(shape: tuple[int, ...], min_box_c: int, ne: int) -> tuple[str, int]:
    """Return ('box'|'ang', spatial_area) for NCHW or NHWC 4D tensor shapes from the DPU."""
    if len(shape) != 4:
        raise ValueError(f"Expected 4D output shape, got {shape}")
    _, a, b, c = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    if a >= min_box_c:
        return "box", b * c
    if c >= min_box_c:
        return "box", a * b
    if a == ne or (ne == 1 and a == 1):
        return "ang", b * c
    if c == ne or (ne == 1 and c == 1):
        return "ang", a * b
    raise ValueError(
        f"Cannot classify OBB output shape {shape} for min_box_c={min_box_c}, ne={ne} "
        "(expect 3 box+cls tensors and 3 angle tensors)"
    )


def obb_dpu_to_decode_order(
    shapes: list[tuple[int, ...]],
    reg_max: int,
    nc: int,
    ne: int,
) -> list[int]:
    """
    Map NPZ/DPU output indices to the order expected by ``OBB26YOLO.decode``:
    three box+cls heads then three angle heads, each trio sorted large grid -> small (stride 8,16,32).
    """
    if len(shapes) != 6:
        raise ValueError(f"OBB needs 6 output shapes, got {len(shapes)}")
    min_box_c = 4 * reg_max + nc
    box_entries: list[tuple[int, int]] = []
    ang_entries: list[tuple[int, int]] = []
    for idx, sh in enumerate(shapes):
        kind, area = _classify_obb_output_shape(sh, min_box_c, ne)
        if kind == "box":
            box_entries.append((-area, idx))
        else:
            ang_entries.append((-area, idx))
    if len(box_entries) != 3 or len(ang_entries) != 3:
        raise ValueError(
            f"Expected 3 box+cls and 3 angle outputs from shapes {shapes}; "
            f"got {len(box_entries)} box-like and {len(ang_entries)} angle-like"
        )
    box_entries.sort(key=lambda t: t[0])
    ang_entries.sort(key=lambda t: t[0])
    return [t[1] for t in box_entries] + [t[1] for t in ang_entries]


def reorder_outputs(outputs: list[np.ndarray], perm: list[int]) -> list[np.ndarray]:
    return [outputs[j] for j in perm]


def _to_nchw_box(x: np.ndarray, min_box_c: int) -> np.ndarray:
    """DPU may emit box+cls as NCHW (1,C,H,W) or NHWC (1,H,W,C)."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D box+cls tensor, got shape {x.shape}")
    _, a, b, c = x.shape
    if a >= min_box_c:
        return x
    if c >= min_box_c:
        return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    raise ValueError(
        f"box+cls tensor shape {x.shape} is neither NCHW nor NHWC with C>={min_box_c}"
    )


def _to_nchw_angle(x: np.ndarray, ne: int) -> np.ndarray:
    """
    Angle heads are often NHWC (1,H,W,ne) from the DPU. ``ne=1`` breaks generic BCHW detection
    (H>=1), so we branch on exact channel counts.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D angle tensor, got shape {x.shape}")
    _, a, b, c = x.shape
    if a == ne:
        return x
    if c == ne:
        return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    raise ValueError(f"angle tensor shape {x.shape} is neither (1,{ne},H,W) nor (1,H,W,{ne})")


def obb_outputs_to_nchw(outputs: list[np.ndarray], reg_max: int, nc: int, ne: int) -> list[np.ndarray]:
    """After DPU order fix: force [3 box+cls | 3 angle] tensors to NCHW for ``OBB26YOLO.decode``."""
    if len(outputs) != 6:
        raise ValueError(f"Expected 6 OBB outputs, got {len(outputs)}")
    min_box_c = 4 * reg_max + nc
    out: list[np.ndarray] = []
    for j in range(3):
        out.append(_to_nchw_box(outputs[j], min_box_c))
    for j in range(3, 6):
        out.append(_to_nchw_angle(outputs[j], ne))
    return out


def _detect_level_spatial_area(shape: tuple[int, ...], min_box_c: int) -> int:
    _, a, b, c = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    if a >= min_box_c:
        return b * c
    if c >= min_box_c:
        return a * b
    raise ValueError(
        f"Detect output shape {shape}: expected NCHW or NHWC with C>={min_box_c}"
    )


def detect_dpu_to_decode_order(shapes: list[tuple[int, ...]], reg_max: int, nc: int) -> list[int]:
    """
    Permute three detect heads so the largest feature map (stride 8) is first, then 16, then 32.
    Compiler order does not always match training export.
    """
    if len(shapes) != 3:
        raise ValueError(f"Detect needs 3 output shapes, got {len(shapes)}")
    min_box_c = 4 * reg_max + nc
    entries = [(-_detect_level_spatial_area(sh, min_box_c), i) for i, sh in enumerate(shapes)]
    entries.sort(key=lambda t: t[0])
    return [t[1] for t in entries]


def load_npz_predictions(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    names = data["image_names"]
    image_names = [str(x) for x in names.tolist()]
    output_shapes = _shapes_list(data["output_shapes"])
    fixpoints = _fixpoints_list(data["output_fixpoints"])
    num_outputs = int(np.asarray(data["num_outputs"]).reshape(()))

    meta: dict = {
        "image_names": image_names,
        "output_shapes": output_shapes,
        "output_fixpoints": fixpoints,
        "num_outputs": num_outputs,
    }
    if "model_type" in data.files:
        raw_mt = np.asarray(data["model_type"]).item()
        if isinstance(raw_mt, bytes):
            meta["model_type"] = raw_mt.decode("utf-8", errors="replace")
        else:
            meta["model_type"] = str(raw_mt)
    if "img_height" in data.files:
        meta["img_height"] = int(np.asarray(data["img_height"]).reshape(()))
    if "img_width" in data.files:
        meta["img_width"] = int(np.asarray(data["img_width"]).reshape(()))
    if "reg_max" in data.files:
        meta["reg_max"] = int(np.asarray(data["reg_max"]).reshape(()))
    if "strides" in data.files:
        meta["strides"] = [int(x) for x in np.asarray(data["strides"]).tolist()]

    preds: list[list[np.ndarray]] = []
    nimg = len(image_names)
    for i in range(nimg):
        level = []
        for j in range(num_outputs):
            key = f"pred_{i}_output_{j}"
            if key not in data.files:
                raise KeyError(f"Missing {key} in {npz_path}")
            level.append(np.asarray(data[key]))
        preds.append(level)

    data.close()
    meta["predictions"] = preds
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLOv11 from NPZ (DPU int8 outputs + fix_point).")
    p.add_argument("--npz", type=str, required=True, help="Path to predictions .npz from fpga_inference")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        choices=("obb", "detect"),
        help="Override task; default: infer from NPZ model_type or output count (6=obb, 3=detect)",
    )
    p.add_argument("--data", type=str, required=True, help="Dataset YAML (names + val/test image paths)")
    p.add_argument(
        "--quant-meta",
        type=str,
        default=None,
        help="Optional pkl from vai_quantize (tensor_ch=reg_max, tensor_nc=nc, strides)",
    )
    p.add_argument("--imgsz", type=int, default=416, help="Square input size used when capturing NPZ (DPU resize)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU (rotated for OBB)")
    p.add_argument("--nc", type=int, default=1, help="Number of classes")
    p.add_argument("--reg-max", type=int, default=16, help="DFL reg_max; overridden by --quant-meta or NPZ reg_max")
    p.add_argument("--ne", type=int, default=1, help="OBB: angle channels per level")
    p.add_argument("--strides", type=str, default="8,16,32", help="Comma-separated strides (overridden by quant-meta / NPZ)")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    p.add_argument("--save-dir", type=str, default=None, help="Output directory for visualizations + metrics.json")
    p.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections as txt (same format as eval_onnx)",
    )
    return p.parse_args()


def _infer_task(npz_meta: dict, explicit: str | None) -> str:
    if explicit:
        return explicit
    mt = npz_meta.get("model_type")
    if isinstance(mt, str) and mt.lower() in ("obb", "detect"):
        return mt.lower()
    n = npz_meta["num_outputs"]
    if n == 6:
        return "obb"
    if n == 3:
        return "detect"
    raise ValueError(f"Cannot infer --task from num_outputs={n}; set --task explicitly")


def _apply_npz_arch(npz_meta: dict, args: argparse.Namespace) -> None:
    if "reg_max" in npz_meta:
        args.reg_max = int(npz_meta["reg_max"])
    if "strides" in npz_meta and len(npz_meta["strides"]) == 3:
        args.strides = ",".join(str(int(s)) for s in npz_meta["strides"])
    if "img_height" in npz_meta and "img_width" in npz_meta:
        h, w = npz_meta["img_height"], npz_meta["img_width"]
        if h == w:
            args.imgsz = int(h)


def run_detect_npz(
    args: argparse.Namespace,
    npz_meta: dict,
    name_to_path: dict[str, Path],
    save_dir: Path,
) -> None:
    strides = [int(s.strip()) for s in args.strides.split(",") if s.strip()]
    if len(strides) != 3:
        raise ValueError(f"Expected 3 strides, got {strides}")

    head = DetectYOLO(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        nc=args.nc,
        reg_max=args.reg_max,
        strides=strides,
        max_det=args.max_det,
    )
    map_eval = DetectMapEvaluator(head.names)
    fixpoints = npz_meta["output_fixpoints"]
    shapes = npz_meta["output_shapes"]
    det_perm = detect_dpu_to_decode_order(shapes, head.reg_max, head.nc)
    if det_perm != [0, 1, 2]:
        print(
            f"[INFO] Detect: DPU output index remap {det_perm} by grid size "
            f"(large -> small vs strides {strides})"
        )
    min_box_c = 4 * head.reg_max + head.nc

    print(f"NPZ images: {len(npz_meta['image_names'])}")
    print(f"Task: detect | imgsz: {head.imgsz} (square DPU resize -> stretch_meta)")

    for i, name in enumerate(npz_meta["image_names"]):
        im_path = name_to_path.get(name)
        if im_path is None:
            print(f"[WARN] No image path for NPZ entry {name!r}, skipping")
            continue
        im0 = cv2.imread(str(im_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {im_path}")
            continue

        raw = npz_meta["predictions"][i]
        outputs = dequantize_int8_outputs(raw, fixpoints)
        outputs = reorder_outputs(outputs, det_perm)
        outputs = [_to_nchw_box(o, min_box_c) for o in outputs]
        xyxy, scores = head.decode(outputs)

        h0, w0 = im0.shape[:2]
        meta = stretch_meta(h0, w0, head.imgsz)
        det = head.postprocess(xyxy, scores, meta)
        vis = head.draw(im0, det)
        out_file = save_dir / im_path.name
        cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{im_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{im_path.name}: {len(det)} detections -> {out_file}")
        map_eval.update(im_path, det, im0.shape[:2])

    print_and_save_metrics(map_eval.finalize(), save_dir, "Detect (NPZ)")


def run_obb_npz(
    args: argparse.Namespace,
    npz_meta: dict,
    name_to_path: dict[str, Path],
    save_dir: Path,
) -> None:
    strides = [int(s.strip()) for s in args.strides.split(",") if s.strip()]
    if len(strides) != 3:
        raise ValueError(f"Expected 3 strides, got {strides}")

    head = OBB26YOLO(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        nc=args.nc,
        reg_max=args.reg_max,
        ne=args.ne,
        strides=strides,
        max_det=args.max_det,
    )
    map_eval = OBBMapEvaluator(head.names)
    fixpoints = npz_meta["output_fixpoints"]
    shapes = npz_meta["output_shapes"]
    obb_perm = obb_dpu_to_decode_order(shapes, head.reg_max, head.nc, head.ne)
    if obb_perm != list(range(6)):
        print(
            f"[INFO] OBB: DPU tensor order differs from decode; remapping indices {obb_perm} "
            f"-> three box+cls then three angle (large grid first, strides {strides})"
        )

    print(f"NPZ images: {len(npz_meta['image_names'])}")
    print(f"Task: obb | imgsz: {head.imgsz} (square DPU resize -> stretch_meta)")

    for i, name in enumerate(npz_meta["image_names"]):
        im_path = name_to_path.get(name)
        if im_path is None:
            print(f"[WARN] No image path for NPZ entry {name!r}, skipping")
            continue
        im0 = cv2.imread(str(im_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {im_path}")
            continue

        raw = npz_meta["predictions"][i]
        outputs = dequantize_int8_outputs(raw, fixpoints)
        outputs = reorder_outputs(outputs, obb_perm)
        outputs = obb_outputs_to_nchw(outputs, head.reg_max, head.nc, head.ne)
        xywh, cls_scores, angles = head.decode(outputs)

        h0, w0 = im0.shape[:2]
        meta = stretch_meta(h0, w0, head.imgsz)
        det = head.postprocess(xywh, cls_scores, angles, meta)
        vis = head.draw(im0, det)
        out_file = save_dir / im_path.name
        cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{im_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{im_path.name}: {len(det)} detections -> {out_file}")
        map_eval.update(im_path, det, im0.shape[:2])

    print_and_save_metrics(map_eval.finalize(), save_dir, "OBB (NPZ)")


def main() -> None:
    args = parse_args()
    apply_quant_meta(args)

    npz_path = Path(args.npz).resolve()
    npz_meta = load_npz_predictions(npz_path)
    _apply_npz_arch(npz_meta, args)

    task = _infer_task(npz_meta, args.task)
    n_out = npz_meta["num_outputs"]
    if task == "detect" and n_out != 3:
        raise ValueError(f"detect expects 3 NPZ outputs, got {n_out}")
    if task == "obb" and n_out != 6:
        raise ValueError(f"obb expects 6 NPZ outputs, got {n_out}")

    source = resolve_source_from_data(args.data)
    images = iter_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in source: {source}")
    name_to_path = {p.name: p for p in images}

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path("runs/detect_npz_eval") if task == "detect" else Path("runs/obb_npz_eval")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded NPZ: {npz_path}")
    print(f"Resolved task: {task} | val/test root: {source}")

    if task == "detect":
        run_detect_npz(args, npz_meta, name_to_path, save_dir)
    else:
        run_obb_npz(args, npz_meta, name_to_path, save_dir)


if __name__ == "__main__":
    main()
