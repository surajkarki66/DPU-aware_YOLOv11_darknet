import copy
import csv
import os
import warnings
import torch
import tqdm
import yaml

from torch.utils import data
from argparse import ArgumentParser
from datetime import datetime

from models.yolo import *
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

from utils.plotting import plot_mAP


def train(args, params):
    # Model
    version = args.version
    activation = args.activation
    
    if version == 'n':
        model = yolo_v11_n(len(params['names']), 
                              exclude_post_process=False,
                              activation=activation)
    elif version == 's':
        model = yolo_v11_s(len(params['names']),
                              exclude_post_process=False,
                              activation=activation)
    elif version == 'm':
        model = yolo_v11_m(len(params['names']),
                              exclude_post_process=False,
                              activation=activation)
    elif version == 'l':
        model = yolo_v11_l(len(params['names']),
                              exclude_post_process=False,
                              activation=activation)
    elif version == 'x':
        model = yolo_v11_x(len(params['names']),
                              exclude_post_process=False,
                              activation=activation)
    else:
        raise ValueError(f"Unsupported YOLOv11 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")

    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(f'{args.data_dir}/train2017.txt') as f:
        for filename in f.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{args.data_dir}/images/train2017/' + filename)
        print("filename lists: ", len(filenames))

    # check if file exists
    existing_count = 0
    nonexisting_count = 0

    for filepath in filenames:
        if os.path.exists(filepath):
            existing_count += 1
        else:
            nonexisting_count += 1

    print(f"Number of existing files: {existing_count}")
    print(f"Number of non-existing files: {nonexisting_count}")

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    
    # loading data
    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    # Use args.save_dir ---
    csv_file = os.path.join(args.save_dir, 'results.csv')

    # Initialize lists to record mAP and epoch numbers
    mAP_list = []
    epoch_list = []
    # Write file
    with open(csv_file, 'w', newline='') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl', 'train_loss',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()


        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                from test import test
                last = test(args, params, ema.ema)
                current_mAP = last[0]  # mAP computed from test()
                mAP_list.append(current_mAP)
                epoch_list.append(epoch + 1)
                total_train_loss = avg_box_loss.avg + avg_cls_loss.avg + avg_dfl_loss.avg

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'train_loss': str(f'{total_train_loss:.3f}'),
                                 'mAP': str(f'{last[0]:.6f}'),
                                 'mAP@50': str(f'{last[1]:.6f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                if current_mAP > best:
                    best = current_mAP

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema),
                        }
                # Save to specific dir with clean names ---
                last_path = os.path.join(args.save_dir, 'last.pt')
                best_path = os.path.join(args.save_dir, 'best.pt')
                
                torch.save(save, f=last_path)
                
                if best == current_mAP:
                    torch.save(save, f=best_path)
                
                del save

    if args.local_rank == 0:
        util.strip_optimizer(f'./runs/{version}{args.epochs}/last.pt')  # strip optimizers
        util.strip_optimizer(f'./runs/{version}{args.epochs}/best.pt')  # strip optimizers
    plot_mAP(args)


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    print(f"params amount: {len(params['names'])}")
    version = args.version
    if version == 'n':
        model = yolo_v11_n(len(params['names'])).fuse()
    elif version == 's':
        model = yolo_v11_s(len(params['names'])).fuse()
    elif version == 'm':
        model = yolo_v11_m(len(params['names'])).fuse()
    elif version == 'l':
        model = yolo_v11_l(len(params['names'])).fuse()
    elif version == 'x':
        model = yolo_v11_x(len(params['names'])).fuse()
    else:
        raise ValueError(f"Unsupported YOLOv11 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def zip_weights_directory(args):
    # Zip the specific folder
    target_dir = args.save_dir
    output_zip = f"{args.save_dir}.zip"

    if not os.path.exists(target_dir):
        print(f"Error: {target_dir} does not exist.")
        return

    import shutil
    shutil.make_archive(args.save_dir, 'zip', target_dir)
    print(f"Successfully zipped {target_dir} to {output_zip}")


def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--version', default='n', type=str)
    parser.add_argument('--activation', default='relu', type=str, 
                        help='Activation function (e.g., relu, silu, etc.)')
    parser.add_argument('--data-dir', default='./coco_data', type=str,
                        help='Path to data directory')
    parser.add_argument('--hyp', default='data/hyps/args.yaml', type=str,
                        help='Path to YAML config file')
    parser.add_argument('--zip', action='store_true')

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

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    with open(args.hyp, errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    train(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()

    if args.zip:
        zip_weights_directory(args)

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