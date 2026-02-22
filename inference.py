import os
import warnings
import torch
import yaml
import cv2
import numpy as np
import time

from argparse import ArgumentParser
from datetime import datetime

from utils import util

warnings.filterwarnings("ignore")


def inference(model, args, params):
    source_type = args.source
    if source_type == "image":
        source_path = args.source_path
        frame = cv2.imread(source_path)

        if frame is None:
            print(f"Error: Could not read image from {source_path}")
            return
        
        # Start timing for single image inference
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Preprocessing, Inference, and Post-processing for a single image
        # 1. Pre-process (Don't time this if checking Model Speed)
        image = frame.copy()
        shape = image.shape[:2]

        r = args.input_size / max(shape[0], shape[1])
        if r != 1:
            resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
        height, width = image.shape[:2]

        # Scale ratio (new / old)
        r = min(1.0, args.input_size / height, args.input_size / width)

        # Compute padding
        pad = int(round(width * r)), int(round(height * r))
        w = (args.input_size - pad[0]) / 2
        h = (args.input_size - pad[1]) / 2

        if (width, height) != pad:
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

        # Convert HWC to CHW, BGR to RGB
        x = image.transpose((2, 0, 1))[::-1]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).unsqueeze(dim=0).cuda().half()/255.0

        # 2. Pure Inference Timer
        start_event.record()
        outputs = model(x)
        end_event.record()
        torch.cuda.synchronize()
        model_latency = start_event.elapsed_time(end_event)

        # 3. Post-process (NMS) Timer
        task = getattr(args, 'task', 'detect')
        t0 = time.time()
        if task == 'obb':
            outputs = util.non_max_suppression_obb(outputs, 0.15, 0.2)[0]
        else:
            outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
        nms_time = (time.time() - t0) * 1000
        
        # End timing and calculate latency
        end_event.record()
        torch.cuda.synchronize()

        # Total System Latency
        total_latency = model_latency + nms_time
        
        if outputs is not None:
            scale = min(height / shape[0], width / shape[1])
            if task == 'obb':
                outputs[:, 0] -= w
                outputs[:, 1] -= h
                outputs[:, :4] /= scale
                outputs[:, 0].clamp_(0, shape[1])
                outputs[:, 1].clamp_(0, shape[0])
                outputs[:, 2].clamp_(0, shape[1])
                outputs[:, 3].clamp_(0, shape[0])
                for box in outputs:
                    box = box.cpu().numpy()
                    xywhr = np.concatenate([box[:4], [box[6]]])
                    score, cls_idx = box[4], int(box[5])
                    class_name = params['names'][cls_idx]
                    label = f"{class_name} {score:.2f}"
                    util.draw_rotated_box(frame, xywhr, cls_idx, label)
            else:
                outputs[:, [0, 2]] -= w
                outputs[:, [1, 3]] -= h
                outputs[:, :4] /= scale
                outputs[:, 0].clamp_(0, shape[1])
                outputs[:, 1].clamp_(0, shape[0])
                outputs[:, 2].clamp_(0, shape[1])
                outputs[:, 3].clamp_(0, shape[0])
                for box in outputs:
                    box = box.cpu().numpy()
                    _, _, _, _, score, index = box
                    class_name = params['names'][int(index)]
                    label = f"{class_name} {score:.2f}"
                    util.draw_box(frame, box, index, label)

        # Display latency on the image
        latency_text = f"Model: {model_latency:.1f}ms | NMS: {nms_time:.1f}ms | Total: {total_latency:.1f}ms"
        cv2.putText(frame, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Inference Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        task = getattr(args, 'task', 'detect')
        if source_type == "video":
            camera = cv2.VideoCapture(args.source_path)
        elif source_type == "camera":
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Get video properties
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = camera.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output2.mp4', fourcc, fps, (width, height))

        if not camera.isOpened():
            print("Error opening video stream or file")
            return

        start_time = datetime.now()
        frame_count = 0
        fps_display = 0.0
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        while camera.isOpened():
            success, frame = camera.read()
            if success:
                frame_count += 1
                current_time = datetime.now()
                elapsed_time = (current_time - start_time).total_seconds()
                if elapsed_time > 1.0:
                    fps_display = frame_count / elapsed_time
                    frame_count = 0
                    start_time = current_time

                start_event.record()

                # --- TIMER 1: Start System Timer ---
                t_start_system = time.time()

                # 1. Pre-processing (CPU)
                t_prep_start = time.time()
                image = frame.copy()

                shape = image.shape[:2]
                r = args.input_size / max(shape[0], shape[1])
                if r != 1:
                    resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                    image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
                height, width = image.shape[:2]
                r = min(1.0, args.input_size / height, args.input_size / width)
                pad = int(round(width * r)), int(round(height * r))
                w = (args.input_size - pad[0]) / 2
                h = (args.input_size - pad[1]) / 2
                if (width, height) != pad:
                    image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
                left, right = int(round(w - 0.1)), int(round(w + 0.1))
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
                x = image.transpose((2, 0, 1))[::-1]
                x = np.ascontiguousarray(x)

                x = torch.from_numpy(x).unsqueeze(dim=0).cuda().half() / 255
                # x = x.unsqueeze(dim=0)
                # x = x.cuda()
                # x = x.half()
                # x = x / 255
                t_prep_end = time.time()

                # 2. Inference (GPU)
                # We use CUDA events for precise GPU timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()

                outputs = model(x)
                end_event.record()
                torch.cuda.synchronize() # Wait for GPU to finish
                inference_time_ms = start_event.elapsed_time(end_event) # Pure Model Time
                
                # 3. NMS (CPU)
                t_nms_start = time.time()
                if task == 'obb':
                    outputs = util.non_max_suppression_obb(outputs, 0.15, 0.2)[0]
                else:
                    outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
                t_nms_end = time.time()
                # Calculate Latencies
                preprocess_ms = (t_prep_end - t_prep_start) * 1000
                nms_ms = (t_nms_end - t_nms_start) * 1000
                e2e_latency_ms = preprocess_ms + inference_time_ms + nms_ms

                # 4. Visualization (CPU - Slow!)
                scale = min(height / shape[0], width / shape[1])
                if outputs is not None:
                    if task == 'obb':
                        outputs[:, 0] -= w
                        outputs[:, 1] -= h
                        outputs[:, :4] /= scale
                        outputs[:, 0].clamp_(0, shape[1])
                        outputs[:, 1].clamp_(0, shape[0])
                        outputs[:, 2].clamp_(0, shape[1])
                        outputs[:, 3].clamp_(0, shape[0])
                        for box in outputs:
                            box = box.cpu().numpy()
                            xywhr = np.concatenate([box[:4], [box[6]]])
                            score, cls_idx = box[4], int(box[5])
                            class_name = params['names'][cls_idx]
                            label = f"{class_name} {score:.2f}"
                            util.draw_rotated_box(frame, xywhr, cls_idx, label)
                    else:
                        outputs[:, [0, 2]] -= w
                        outputs[:, [1, 3]] -= h
                        outputs[:, :4] /= scale
                        outputs[:, 0].clamp_(0, shape[1])
                        outputs[:, 1].clamp_(0, shape[0])
                        outputs[:, 2].clamp_(0, shape[1])
                        outputs[:, 3].clamp_(0, shape[0])
                        for box in outputs:
                            box = box.cpu().numpy()
                            x1, y1, x2, y2, score, index = box
                            class_name = params['names'][int(index)]
                            label = f"{class_name} {score:.2f}"
                            util.draw_box(frame, box, index, label)
                
                # fps_text = f"FPS: {fps_display:.2f}"
                # latency_text = f"Latency: {latency_ms:.2f} ms"
                # cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # cv2.imshow('Frame', frame)
                # out.write(frame)

                # 5. FPS Calculation (Moving Average)
                # We use t_start_system to capture the FULL loop time
                system_latency_ms = (time.time() - t_start_system) * 1000
                current_fps = 1000.0 / (system_latency_ms + 1e-8)
                
                # Smoothing the FPS display so it doesn't flicker
                fps_display = 0.9 * fps_display + 0.1 * current_fps

                # --- DISPLAY STATS ---
                # Line 1: Real System Speed
                theoretical_fps = 1000.0 / e2e_latency_ms
                cv2.putText(frame, f"System FPS: {fps_display:.1f} | Model Potential: {theoretical_fps:.0f} FPS", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Line 2: The breakdown (Why is it slow?)
                info_text = f"Pre:{preprocess_ms:.1f}ms | Inf:{inference_time_ms:.1f}ms | NMS:{nms_ms:.1f}ms"
                cv2.putText(frame, info_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # # Line 3: Theoretical Max FPS (If you removed display/webcam bottleneck)
                # cv2.putText(frame, f"Model Potential: {theoretical_fps:.0f} FPS", (10, 90), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow('Inference', frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        camera.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--version', default='n', type=str)
    parser.add_argument('--task', default='detect', type=str, choices=['detect', 'obb'],
                        help='Task: detect or obb')
    parser.add_argument('--weights', default='', type=str,
                        help='Path to full checkpoint best.pt (default: runs/train_<version>/best.pt)')
    parser.add_argument('--source', type=str, choices=["image", "video", "camera"], required=True,
                        help="Inference source: 'image', 'video', or 'camera'")
    parser.add_argument('--source-path', type=str, default='./data/example.jpg',
                        help="Path to source file (for image/video mode)")
    parser.add_argument('--hyp', default='data/hyps/args.yaml', type=str,
                        help='Path to YAML config file')

    args = parser.parse_args()
    print(args)

    # --- STRATEGY: Define Dynamic Save Directory ---
    run_name = f"train_{args.version}"
    args.save_dir = os.path.join("runs", run_name)
    print(f"Output Directory: {args.save_dir}")
    # -----------------------------------------------

    with open(args.hyp, errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    model_path = (args.weights or os.path.join(args.save_dir, "best.pt")).strip()
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location="cuda", weights_only=False)
    if not isinstance(ckpt, dict):
        print("Error: Only full checkpoint (best.pt with 'model' or 'ema') is supported. State-dict-only files are not supported.")
        return
    model = ckpt.get('ema') or ckpt.get('model')
    if model is None or not isinstance(model, torch.nn.Module):
        print("Error: Checkpoint must contain 'model' or 'ema' (nn.Module). Use a full best.pt from training.")
        return
    model = model.float().fuse().cuda().half().eval()

    inference(model, args, params)

    torch.cuda.empty_cache()

    time_end = datetime.now()
    print("Finished at Date and Time:", time_end.strftime("%Y-%m-%d %H:%M:%S"))
    time_duration = time_end - time_start
    # Format the duration as Days HH:MM:SS
    days = time_duration.days
    seconds = time_duration.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_duration = f"{days} Days {hours:02}:{minutes:02}:{seconds:02}"
    print(f"Code execution time: {formatted_duration}")


if __name__ == "__main__":
    main()