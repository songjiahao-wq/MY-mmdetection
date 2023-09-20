import os
import subprocess

if __name__ =='__main__':
    # subprocess.run(['python', 'setup.py', 'install'])
    # subprocess.run(['python', './tools/train.py', 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco_MY.py'])
    subprocess.run(['python', './tools/train.py', 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'])