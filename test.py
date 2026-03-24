import os
import warnings
import torch
import tqdm
import yaml
import json

from torch.utils import data
from argparse import ArgumentParser
from datetime import datetime

from utils import util
from utils.dataset import Dataset, OBBDataset
from models.yolo import (
    yolo_v11_n, yolo_v11_n_obb,
    yolo_v11_t, yolo_v11_t_obb,
    yolo_v11_s, yolo_v11_s_obb,
    yolo_v11_m, yolo_v11_m_obb,
    yolo_v11_l, yolo_v11_l_obb,
    yolo_v11_x, yolo_v11_x_obb,
)

warnings.filterwarnings("ignore")


def _build_model_from_args(args, params):
    """Build model from args.version and args.task for state_dict loading."""
    version = getattr(args, 'version', 'n')
    task = getattr(args, 'task', 'detect')
    nc = len(params.get('names', [])) or 1
    factories = {
        ('n', 'detect'): yolo_v11_n,
        ('n', 'obb'): yolo_v11_n_obb,
        ('t', 'detect'): yolo_v11_t,
        ('t', 'obb'): yolo_v11_t_obb,
        ('s', 'detect'): yolo_v11_s,
        ('s', 'obb'): yolo_v11_s_obb,
        ('m', 'detect'): yolo_v11_m,
        ('m', 'obb'): yolo_v11_m_obb,
        ('l', 'detect'): yolo_v11_l,
        ('l', 'obb'): yolo_v11_l_obb,
        ('x', 'detect'): yolo_v11_x,
        ('x', 'obb'): yolo_v11_x_obb,
    }
    fn = factories.get((version, task))
    if fn is None:
        raise ValueError(f"Unsupported version={version!r} or task={task!r}. Use version in n,t,s,m,l,x and task in detect,obb.")
    return fn(nc, exclude_post_process=False, activation=getattr(args, 'activation', 'relu'))


@torch.no_grad()
def test(args, params, model=None, mode="val"):
    data_dir = args.data_dir
    version = args.version
    filenames = []
    if mode == "val":
        with open(f'{data_dir}/val2017.txt') as f:
            for filename in f.readlines():
                filename = os.path.basename(filename.rstrip())
                filenames.append(f'{data_dir}/images/val2017/' + filename)
    
    if mode =="test":
        with open(f'{data_dir}/test2017.txt') as f:
            for filename in f.readlines():
                filename = os.path.basename(filename.rstrip())
                filenames.append(f'{data_dir}/images/test2017/' + filename)

    task = getattr(args, 'task', 'detect')
    if task == 'obb':
        dataset = OBBDataset(filenames, args.input_size, params, augment=False)
        collate_fn = OBBDataset.collate_fn
    else:
        dataset = Dataset(filenames, args.input_size, params, augment=False)
        collate_fn = Dataset.collate_fn
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=collate_fn)

    if not model:
        path = (getattr(args, 'weights', '') or '').strip()
        if not path:
            path = os.path.join("runs", f"train_{version}", "best.pt")
        print(f"Testing model: {path}")
        ckpt = torch.load(f=path, map_location='cuda', weights_only=False)
        model = ckpt.get('ema') or ckpt.get('model') if isinstance(ckpt, dict) else None
        if model is None:
            model = ckpt
        if not isinstance(model, torch.nn.Module):
            state_dict = model if isinstance(model, dict) else ckpt
            model = _build_model_from_args(args, params)
            model.load_state_dict(state_dict, strict=False)
        model = model.float().fuse()

    model = model.cuda()
    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        scale_obb = torch.tensor((w, h, w, h, 1), device=samples.device, dtype=samples.dtype)
        # Inference
        outputs = model(samples)
        # NMS
        if task == 'obb':
            outputs = util.non_max_suppression_obb(outputs, confidence_threshold=0.001, iou_threshold=0.45)
        else:
            outputs = util.non_max_suppression(outputs, confidence_threshold=0.0001, iou_threshold=0.65)

        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            if task == 'obb':
                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
                if output.shape[0] == 0:
                    metrics.append((metric, torch.zeros(0, device=output.device), torch.zeros(0, device=output.device), cls.squeeze(-1)))
                    continue
                if cls.shape[0]:
                    target_xywhr = box * scale_obb
                    metric = util.compute_metric_obb(output, cls.squeeze(-1), target_xywhr, iou_v)
                metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))
            else:
                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
                if output.shape[0] == 0:
                    metrics.append((metric, torch.zeros(0, device=output.device), torch.zeros(0, device=output.device), cls.squeeze(-1)))
                    continue
                if cls.shape[0]:
                    target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                    metric = util.compute_metric(output[:, :6], target, iou_v)
                metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))
    
    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    # Update compute_ap call if it uses save paths internally:
    if len(metrics) and metrics[0].any():
        # Pass save_dir to compute_ap
        _, _, m_pre, m_rec, map50, mean_ap = util.compute_ap(
            version, 
            *metrics, 
            plot=True, 
            names=params["names"],
            save_dir=args.save_dir
        )
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    
    # Save metrics to JSON only
    if mode == "test":
        metrics_dict = {
            "mean_ap": float(mean_ap),
            "map50": float(map50),
            "recall": float(m_rec),
            "precision": float(m_pre)
        }
        with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)
    
    # Return results
    model.float()  # for training

    return mean_ap, map50, m_rec, m_pre


def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--version', default='n', type=str)
    parser.add_argument('--data-dir', default='./coco_data', type=str,
                        help='Path to data directory') 
    parser.add_argument('--hyp', default='data/hyps/args.yaml', type=str,
                        help='Path to YAML config file')
    parser.add_argument('--weights', default='', type=str,
                        help='Path to checkpoint .pt for standalone test (default: runs/train_{version}/best.pt)')
    parser.add_argument('--task', default='detect', type=str, choices=('detect', 'obb'),
                        help='Task: detect or obb (must match the model)')
    parser.add_argument('--mode', default='test', type=str, choices=('test', 'val'))
    args = parser.parse_args()
    print(args)

    # --- STRATEGY: Define Dynamic Save Directory ---
    run_name = f"test_{args.version}"
    args.save_dir = os.path.join("runs", run_name)
    print(f"Output Directory: {args.save_dir}")
    # -----------------------------------------------

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    with open(args.hyp, errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    test(args, params, mode=args.mode)

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
