# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 19:12
# @Author  : sjh
# @Site    : 
# @File    : 可视化coco分割.py
# @Comment :
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# 加载COCO数据集
coco_annotation_path = r'D:\xian_yu\daipao\label\annotations\train.json'
coco = COCO(coco_annotation_path)
# 从COCO数据集中获取图片信息
image_ids = coco.getImgIds()
image_data = coco.loadImgs(image_ids[0])[0]  # 获取第一个图片信息

# 读取图片
image_path = os.path.join(os.path.dirname(coco_annotation_path), image_data['file_name'])
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图片中的所有分割标注
annotation_ids = coco.getAnnIds(imgIds=image_data['id'])
annotations = coco.loadAnns(annotation_ids)

# 可视化图片和分割标注
plt.imshow(image)
plt.axis('off')

for ann in annotations:
    mask = coco.annToMask(ann)  # 获取分割掩码
    # 绘制分割轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='blue', linewidth=2)

plt.show()
