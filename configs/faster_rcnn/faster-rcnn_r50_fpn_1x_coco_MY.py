_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', # 模型配置文件
    # '../_base_/datasets/coco_detection.py', 修改为VOC0712.py
    '../_base_/datasets/voc0712_MY.py', # 数据集配置文件 VOC0712.py
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py' # 配置学习率，迭代次数，模型加载路径等等
]
