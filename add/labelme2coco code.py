# -*- coding: utf-8 -*-
# @Time    : 2024/8/30 19:38
# @Author  : sjh
# @Site    : 
# @File    : labelme2coco.py
# @Comment :
import os
import labelme2coco

import os

# 输入和输出路径
input_folder = r'D:\xian_yu\daipao\MY-mmdetection\data\coco\json'
output_file = r'D:\xian_yu\daipao\20240830\data1\merged_coco.json'
labelme2coco.get_coco_from_labelme_folder(labelme_folder = input_folder, )
# 创建 Labelme2CoCo 实例
# labelme2coco = labelme2coco.convert(input_folder, train_split_rate=0.8)

# 执行转换并保存为 COCO 格式的 JSON 文件
