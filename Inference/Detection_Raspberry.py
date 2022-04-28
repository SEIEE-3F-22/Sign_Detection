import torch
import torchvision
from openvino.inference_engine import IECore, IENetwork
import cv2
import numpy as np
import time
import os

import subprocess

# 推断
# def Inference(net, exec_net, image_file):
#     Read image
#     img0 = cv2.imread(image_file)
####################### Model Inference ##########################

def Inference(net, exec_net, img0, size): #3.21修改，视频流直接传入图片+修改推理图片大小

    print("img0 Size:", img0.shape)
    # Padded resize
    img = cv2.resize(img0, (size, size)) #640
    print("img Size:", img.shape)
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.half()
    img = img[None]

    '''
    # 模型输入图片，进行推理
    n, c, h, w = net.inputs[input_blob].shape
    frame = cv2.imread(image_file)
    initial_h, initial_w, channels = frame.shape
    # 按照AI模型要求放缩图片

    image = cv2.resize(frame, (w, h))
    image = torch.from_numpy(image)
    # 下面这两步特别关键！！！！不这么处理推断结果就会出大错！！
    image = image.half()

    image = image[None]
    image = image.transpose(1, 3)
    image = image.transpose(2, 3)
    '''

    print("image shape is: {}".format(img.shape))
    print("Starting inference in synchronous mode")
    start = time.time()
    res = exec_net.infer(inputs={input_blob: img})
    end = time.time()
    print("Infer Time:{}ms".format((end - start) * 1000))
    # return torch.from_numpy(res[out_blob])  # res.shape = [1, 25200, 8]
    return torch.from_numpy(res['output'])
    # return res


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        print("---x shape---:", x.shape)
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


####################### Visualization ##########################
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


class Annotator: #3.21添加，用于可视化结果，只使用于cv2格式

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        # cv2
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

# 3.21修改，用类管理可视化参数
class visual_cfg:
    def __init__(self):
        self.line_thickness = 3

# 3.21修改，用类管理参数
class rtmp_cfg:
    def __init__(self, cap):
        # RTMP服务器地址
        # 改为B站直播
        self.rtmp = r'rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_545363788_34913186&key=cb32c652863fbe24139323becbdffedb&schedule=rtmp&pflag=1'
        # 其他参数配置
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        sizeStr = str(self.size[0]) + 'x' + str(self.size[1])
        self.command = ['ffmpeg',
                   '-y', '-an',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', sizeStr,
                   '-r', '25',
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   self.rtmp]
        self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)

# 3.21修改，用类管理参数
class model_cfg:
    # def __init(self):
    DEVICE = 'CPU' #'MYRIAD'
    model_xml = './models/Myonnx_n_test.xml'
    model_bin = './models/Myonnx_n_test.bin'
    # model_xml = '/home/pi/MyCode/best.fp16.s255.xml'
    # model_bin = '/home/pi/MyCode/best.fp16.s255.bin'
    # model_xml = '/home/pi/newYolov5/yolov5s.xml'
    # model_bin = '/home/pi/newYolov5/yolov5s.bin'
    confidence = 0.6
    conf_thres = 0.25
    iou_thres =  0.45
    agnostic_nms = False
    max_det = 300
    img_size = 640 #3.21添加，图片大小
    # 4.10 bug，classes传入nms里的不能是str，必须是0,1,2，到时候再换成classes_str
    classes_str = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']  # class names
    classes = [i for i in range(len(classes_str))]
    # classes = None
    # self.classes = [] #ours
    num_classes = len(classes)


if __name__ == "__main__":

    # 选择推理视频流
    # 法一：从树莓派摄像头中读取
    cap = cv2.VideoCapture(0)
    # 法二：指定本地视频路径
    # video_path = '../../../0-Datasets/Video/war.mp4'
    # cap = cv2.VideoCapture(video_path)
    # 法三：指定本地图片
    # img_path = '../../../0-Datasets/Detection/coco128_yolo/less/000000000081.jpg'
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 640))

    # 初始化模型参数, 可在类中修改
    model_cfg = model_cfg()
    # 初始化可视化参数，可在类中修改
    visual_cfg = visual_cfg()
    # 初始化rtmp参数, 需要修改rtmp路径
    rtmp_cfg = rtmp_cfg(cap)
    # 初始化设备
    ie = IECore()

    # 读取IR模型
    net = ie.read_network(model=model_cfg.model_xml, weights=model_cfg.model_bin)
    # 转换输入输出张量
    print("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    # 载入模型到CPU
    print("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=1, device_name=model_cfg.DEVICE)
    # 推断
    print("Start Inference!")
    # pic_list = os.listdir(image_file)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # frame = cv2.imread(img_path)
            prediction = Inference(net, exec_net, frame, model_cfg.img_size)
            # print("****prediction's shape is:******", prediction.shape)
            # print(prediction)
            output = non_max_suppression(prediction, model_cfg.conf_thres, model_cfg.iou_thres, model_cfg.classes, model_cfg.agnostic_nms, max_det=model_cfg.max_det)
            # output: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            # cls_id = ans[0][:, -1].unique()
            # 3.21修改，增加推理结果可视化
            for i, det in enumerate(output):  # per image
                im0 = cv2.resize(frame,(model_cfg.img_size, model_cfg.img_size)) #原图
                annotator = Annotator(im0, line_width=visual_cfg.line_thickness)

                if len(det): #有检测框才需要标注
                    # Rescale boxes from img_size to im0 size
                    img_size = (model_cfg.img_size, model_cfg.img_size) #模型推理图大小
                    det[:, :4] = scale_coords(img_size, det[:, :4], im0.shape).round()

                    names = model_cfg.classes_str   # classes name
                    # Print results
                    # ……
                    # Visualize results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # Stream results
                    frame_process = annotator.result()
                else: # 无框直接出结果
                    frame_process = im0
                    # print(0)
                cv2.imshow('test', frame_process)
                cv2.waitKey(10)
                out.write(frame_process)
                rtmp_cfg.pipe.stdin.write(frame_process.tostring())
    while True:
        if cv2.getWindowProperty('test', 0) == -1:  # 当窗口关闭时为-1，显示时为0
            break
        cv2.waitKey(1)
    cv2.destroyWindow('test')
    cap.release()
    rtmp_cfg.pipe.terminate()
