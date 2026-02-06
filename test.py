import os
import warnings
import torch
import tqdm
import yaml

from torch.utils import data
from argparse import ArgumentParser
from datetime import datetime

from utils import util
from utils.dataset import Dataset
from utils.plotting import plot_mAP

warnings.filterwarnings("ignore")


@torch.no_grad()
def test(args, params, model=None):
    data_dir = args.data_dir  
    version = args.version
    epochs = args.epochs
    filenames = []
    with open(f'{data_dir}/val2017.txt') as f:
        for filename in f.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/val2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        path = os.path.join(args.save_dir, "best.pt")
        print(f"Testing model: {path}")
        model = torch.load(f=path, map_location='cuda', weights_only=False)
        model = model['model'].float().fuse()

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
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs, confidence_threshold=0.0001, iou_threshold=0.65)
        
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()
            
            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))
    
    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    # Update compute_ap call if it uses save paths internally:
    if len(metrics) and metrics[0].any():
        # Pass save_dir to compute_ap
        _, _, m_pre, m_rec, map50, mean_ap = util.compute_ap(
            version,
            epochs, 
            *metrics, 
            plot=plot, 
            names=params["names"],
            save_dir=args.save_dir
        )
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training

    plot_mAP(args)
    return mean_ap, map50, m_rec, m_pre


def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--version', default='n', type=str)
    parser.add_argument('--data-dir', default='./coco_data', type=str,
                        help='Path to data directory') 
    parser.add_argument('--hyp', default='data/hyps/args.yaml', type=str,
                        help='Path to YAML config file')

    args = parser.parse_args()
    print(args)

    # --- STRATEGY: Define Dynamic Save Directory ---
    run_name = f"{args.version}{args.epochs}"
    args.save_dir = os.path.join("runs", run_name)
    print(f"Output Directory: {args.save_dir}")
    # -----------------------------------------------

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    with open(args.hyp, errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    test(args, params)

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
