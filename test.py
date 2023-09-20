import os
import subprocess

if __name__ =='__main__':
    # subprocess.run(['python', 'setup.py', 'install'])
    subprocess.run(['python', './tools/test.py', 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py','work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth',
                    '--show'])