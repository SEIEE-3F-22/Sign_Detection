# Sign_Detection

    |——Datasets                  # 数据集
    |  |——coco128                  # 小型测试数据集
    |  |——VOC_v1                   # 自制标志检测数据集_V1
    |——Inference                 # 模型推理脚本
    |  |——models                   # openvino模型
    |  |——video                    # 目标检测测试视频
    |  |——Detection_Raspberry.py   #树莓派目标检测测试脚本
    |——Yolov5                    # Yolov5项目
    |  |——data                     # 数据集配置文件夹
    |  |  |——images                  #单张测试图片文件夹
    |  |  |——VOC_ours.yaml           #标志检测数据集配置文件
    |  |  |——coco128_ours.yaml       #coco128测试数据集配置文件
    |  |——runs                     # 训练/测试/推理结果
    |——pt2onnx_n.ipynb           # pt模型转onnx模型
    |——scripts_ours.sh           # 训练/测试/推理脚本
