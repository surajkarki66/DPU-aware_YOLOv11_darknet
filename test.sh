#python3 test.py --input-size 416 --version n --hyp data/hyps/args_hpd.yaml --weights runs/train_n/best.pt
#python3 test.py --input-size 416 --version s --hyp data/hyps/args_hpd.yaml --weights runs/train_s/best.pt
#python3 test.py --input-size 416 --version t --hyp data/hyps/args_hpd.yaml --weights runs/train_t/best.pt
#python3 test.py --input-size 416 --version m --hyp data/hyps/args_hpd.yaml --weights runs/train_m/best.pt

#python3 test.py --input-size 416 --version n --hyp data/hyps/args_hpd.yaml --task obb --weights runs/train_n/best.pt --data-dir ./coco_data_obb
#python3 test.py --input-size 416 --version t --hyp data/hyps/args_hpd.yaml --task obb --weights runs/train_t/best.pt --data-dir ./coco_data_obb
#python3 test.py --input-size 416 --version s --hyp data/hyps/args_hpd.yaml --task obb --weights runs/train_s/best.pt --data-dir ./coco_data_obb
#python3 test.py --input-size 416 --version m --hyp data/hyps/args_hpd.yaml --task obb --weights runs/train_m/best.pt --data-dir ./coco_data_obb

#python3 test.py --input-size 416 --version n --hyp data/hyps/args_dota_large.yaml --task obb --weights runs/train_n/best.pt --data-dir ./dota.v1_large_coco --mode val
#python3 test.py --input-size 416 --version t --hyp data/hyps/args_dota_large.yaml --task obb --weights runs/train_t/best.pt --data-dir ./dota.v1_large_coco --mode val
#python3 test.py --input-size 1024 --version n --hyp data/hyps/args_dota_large.yaml --task obb --weights runs/train_n/best.pt --data-dir ./dotav1.5 --mode val

#python3 test.py --input-size 416 --version n --hyp data/hyps/args_voc.yaml --weights runs/train_n/best.pt --data-dir ./VOC_COCO --mode val
python3 test.py --input-size 416 --version n --hyp data/hyps/args_kitti.yaml --weights runs/train_n/best.pt --data-dir ./kitti_COCO --mode val
