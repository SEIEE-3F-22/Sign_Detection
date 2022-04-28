python train.py --data ./data/coco128.yaml --cfg ./models/yolov5n.yaml --weights ./weights/yolov5n.pt --batch-size 64 --epochs 20

python detect.py --weights ./runs/train/exp3/weights/best.pt --source ./data/images/wangzige.jpg --name wangzige