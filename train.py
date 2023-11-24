import os
import subprocess

if __name__ =='__main__':
    # subprocess.run(['python', 'setup.py', 'install'])
    subprocess.run(['python', './tools/train.py', 'configs/detr/detr_r50_8xb2-150e_coco.py']) # 训练
    # subprocess.run(['python', './tools/analysis_tools/get_flops.py', 'configs/detr/detr_r18_8xb2-500e_coco.py',
    #                 '--cfg-options','scale=(1, 3, 640, 640) ']) # 计算参数量和计算量
    # python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
"""
# 计算fps
python  tools/analysis_tools/benchmark.py ^
       work_dirs/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco.py ^
       --checkpoint work_dirs/dino-4scale_r50_8xb2-12e_coco/best_coco_bbox_mAP_epoch_10.pth ^
       --launcher none
"""