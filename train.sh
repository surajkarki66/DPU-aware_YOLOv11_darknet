#python3 train.py --input-size 416 --batch-size 32 --epochs 900 --version n --activation relu --hyp data/hyps/args_hpd.yaml
#python3 train.py --input-size 416 --batch-size 32 --epochs 900 --version t --activation relu --hyp data/hyps/args_hpd.yaml
#python3 train.py --input-size 416 --batch-size 32 --epochs 900 --version s --activation relu --hyp data/hyps/args_hpd.yaml
#python3 train.py --input-size 416 --batch-size 32 --epochs 900 --version m --activation relu --hyp data/hyps/args_hpd.yaml


# OBB
#python3 train.py --task obb --input-size 416 --batch-size 32 --epochs 900 --version n --activation relu --hyp data/hyps/args_hpd.yaml --data-dir ./coco_data_obb
#python3 train.py --task obb --input-size 416 --batch-size 32 --epochs 900 --version t --activation relu --hyp data/hyps/args_hpd.yaml --data-dir ./coco_data_obb
#python3 train.py --task obb --input-size 416 --batch-size 32 --epochs 900 --version s --activation relu --hyp data/hyps/args_hpd.yaml --data-dir ./coco_data_obb
#python3 train.py --task obb --input-size 416 --batch-size 32 --epochs 900 --version m --activation relu --hyp data/hyps/args_hpd.yaml --data-dir ./coco_data_obb

# OBB
#python3 train.py --task obb --input-size 416 --batch-size 32 --epochs 900 --version n --activation relu --hyp data/hyps/args_dota_large.yaml --data-dir ./dota.v1_large_coco
#python3 train.py --task obb --input-size 416 --batch-size 32 --epochs 900 --version t --activation relu --hyp data/hyps/args_dota_large.yaml --data-dir ./dota.v1_large_coco
#python3 train.py --task obb --input-size 1024 --batch-size 32 --epochs 900 --version n --activation relu --hyp data/hyps/args_dota_large.yaml --data-dir ./dotav1.5

# VOC
#python3 train.py --input-size 416 --batch-size 32 --epochs 900 --version n --activation relu --hyp data/hyps/args_voc.yaml --data-dir ./VOC_COCO

# Kitti
#python3 train.py --input-size 416 --batch-size 32 --epochs 900 --version n --activation relu --hyp data/hyps/args_kitti.yaml --data-dir ./kitti_COCO

# COCO
python3 train.py --input-size 416 --batch-size 32 --epochs 300 --version n --activation relu --hyp data/hyps/args_coco.yaml --data-dir ./coco
