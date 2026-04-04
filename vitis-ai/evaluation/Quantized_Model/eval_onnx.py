"""ONNX Runtime evaluation for YOLOv11 Detect (3 outputs) or OBB (6 outputs); writes metrics and visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from eval_core import (
    DetectMapEvaluator,
    DetectYOLO,
    OBB26YOLO,
    OBBMapEvaluator,
    apply_quant_meta,
    iter_images,
    print_and_save_metrics,
    resolve_source_from_data,
)

__all__ = [
    "DetectONNX",
    "OBB26ONNX",
    "DetectMapEvaluator",
    "OBBMapEvaluator",
    "main",
]


class DetectONNX(DetectYOLO):
    def __init__(
        self,
        model: str,
        data: str | None,
        imgsz: int,
        conf: float,
        iou: float,
        nc: int,
        reg_max: int,
        strides: list[int],
        max_det: int,
    ) -> None:
        super().__init__(data, imgsz, conf, iou, nc, reg_max, strides, max_det)

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model, providers=providers or available)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        in_shape = input_meta.shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int) and in_shape[2] == in_shape[3]:
            fixed = int(in_shape[2])
            if self.imgsz != fixed:
                print(f"[INFO] Overriding imgsz from {self.imgsz} to ONNX input size {fixed}")
                self.imgsz = fixed


class OBB26ONNX(OBB26YOLO):
    def __init__(
        self,
        model: str,
        data: str | None,
        imgsz: int,
        conf: float,
        iou: float,
        nc: int,
        reg_max: int,
        ne: int,
        strides: list[int],
        max_det: int,
    ) -> None:
        super().__init__(data, imgsz, conf, iou, nc, reg_max, ne, strides, max_det)

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model, providers=providers or available)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        in_shape = input_meta.shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int) and in_shape[2] == in_shape[3]:
            fixed = int(in_shape[2])
            if self.imgsz != fixed:
                print(f"[INFO] Overriding imgsz from {self.imgsz} to ONNX input size {fixed}")
                self.imgsz = fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 quantized ONNX (Detect or OBB).")
    parser.add_argument(
        "--task",
        type=str,
        default="obb",
        choices=("obb", "detect"),
        help="Task type: 'obb' for oriented boxes or 'detect' for axis-aligned detection.",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML for class names and validation images")
    parser.add_argument(
        "--quant-meta",
        type=str,
        default=None,
        help="Path to config pkl from vai_quantize run_model_exports: "
        "(tensor_no, tensor_stride, tensor_ch, tensor_nc, task); tensor_ch is DFL reg_max (e.g. 16)",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="Rotated NMS IoU threshold")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    parser.add_argument("--reg-max", type=int, default=16, help="DFL reg_max (YOLOv11 detect head ch=16); overridden by --quant-meta tensor_ch")
    parser.add_argument("--ne", type=int, default=1, help="OBB only: angle output channels (ignored for --task detect)")
    parser.add_argument("--strides", type=str, default="8,16,32", help="Comma-separated strides per output level")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections as txt. detect: x1 y1 x2 y2 conf cls, obb: x y w h conf cls angle",
    )
    return parser.parse_args()


def run_obb(args: argparse.Namespace, images: list[Path], save_dir: Path, source: Path, strides: list[int]) -> None:
    evaluator = OBB26ONNX(
        model=args.model,
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

    print(f"Loaded model: {args.model}")
    print(f"Task: OBB")
    print(f"Input source: {source}")
    print(f"Images found: {len(images)}")
    print(f"Output dir: {save_dir}")

    map_eval = OBBMapEvaluator(evaluator.names)

    for im_path in images:
        im0 = cv2.imread(str(im_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {im_path}")
            continue

        x, meta = evaluator.preprocess_image(im0)
        outputs = evaluator.session.run(None, {evaluator.input_name: x})
        outputs = [o.astype(np.float32) for o in outputs]
        xywh, cls_scores, angles = evaluator.decode(outputs)
        det = evaluator.postprocess(xywh, cls_scores, angles, meta)

        vis = evaluator.draw(im0, det)
        out_file = save_dir / im_path.name
        cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{im_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{im_path.name}: {len(det)} detections -> {out_file}")

        map_eval.update(im_path, det, im0.shape[:2])

    results = map_eval.finalize()
    print_and_save_metrics(results, save_dir, "OBB")


def run_detect(args: argparse.Namespace, images: list[Path], save_dir: Path, source: Path, strides: list[int]) -> None:
    evaluator = DetectONNX(
        model=args.model,
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        nc=args.nc,
        reg_max=args.reg_max,
        strides=strides,
        max_det=args.max_det,
    )

    print(f"Loaded model: {args.model}")
    print(f"Task: Detect")
    print(f"Input source: {source}")
    print(f"Images found: {len(images)}")
    print(f"Output dir: {save_dir}")

    map_eval = DetectMapEvaluator(evaluator.names)

    for im_path in images:
        im0 = cv2.imread(str(im_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {im_path}")
            continue

        x, meta = evaluator.preprocess_image(im0)
        outputs = evaluator.session.run(None, {evaluator.input_name: x})
        outputs = [o.astype(np.float32) for o in outputs]
        xyxy, scores = evaluator.decode(outputs)
        det = evaluator.postprocess(xyxy, scores, meta)

        vis = evaluator.draw(im0, det)
        out_file = save_dir / im_path.name
        cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{im_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{im_path.name}: {len(det)} detections -> {out_file}")

        map_eval.update(im_path, det, im0.shape[:2])

    results = map_eval.finalize()
    print_and_save_metrics(results, save_dir, "Detect")


def main() -> None:
    args = parse_args()
    apply_quant_meta(args)

    strides = [int(s.strip()) for s in args.strides.split(",") if s.strip()]
    if len(strides) != 3:
        raise ValueError(f"Expected 3 strides, got {strides}")

    source = resolve_source_from_data(args.data)
    images = iter_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in source: {source}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path("runs/detect_onnx_eval") if args.task == "detect" else Path("runs/obb26_onnx_eval")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "detect":
        run_detect(args, images, save_dir, source, strides)
    else:
        run_obb(args, images, save_dir, source, strides)


if __name__ == "__main__":
    main()
