# python train.py --data ./data/VOC_ours.yaml --cfg ./models/yolov5n.yaml --weights ./weights/yolov5n.pt --batch-size 16 --epochs 100 --name sign_noaug_baseline_100ep_16bs

python val.py --data ./data/VOC_ours.yaml --weights ./runs/train/sign_noaug_baseline_100ep_16bs/weights/best.pt --task test --name sign_noaug_baseline_100ep_16bs

# python detect.py --weights ./runs/train/sign_noaug_baseline_100ep_16bs/weights/best.pt --source ./data/images/sign_unlimit_color.jpg --name sign_noaug_baseline_100ep_16bs