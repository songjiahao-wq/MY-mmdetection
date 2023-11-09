# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 11:49
# @Author  : sjh
# @Site    : 
# @File    : detect.py
# @Comment :
# -*- coding: UTF-8 -*-

import os
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

# 使用前修改下面的路径
image_path = r'E:\BaiduNetdiskDownload\work_dirs\rtmdet_s_8xb32-300e_coco\img'
savepath = r'E:\BaiduNetdiskDownload\work_dirs'

config_file = r'E:\BaiduNetdiskDownload\work_dirs\rtmdet_s_8xb32-300e_coco\rtmdet_s_8xb32-300e_coco.py'
checkpoint_file = r'E:\BaiduNetdiskDownload\work_dirs\rtmdet_s_8xb32-300e_coco\best_coco_bbox_mAP_epoch_200.pth'

# gpu 不行 会 Out of memory
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

for filename in os.listdir(image_path):
    imgpath = os.path.join(image_path, filename)

    print('process ', imgpath)
    img = mmcv.imread(imgpath)
    result = inference_detector(model, img)
    out_file = os.path.join(savepath, filename)

    ### old implementation invalid
    # show_result_pyplot(model, img, result, out_file, score_thr=0.3)
    ###

    # show the results
    visualizer.add_datasample('result',
                              img, data_sample=result,
                              draw_gt=False,
                              wait_time=0,
                              out_file=out_file,
                              pred_score_thr=0.3
                              )
    # vis
    # visualizer.show()
    # break
