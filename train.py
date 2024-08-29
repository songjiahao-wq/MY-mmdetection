import os
import subprocess

if __name__ =='__main__':

    # command = ['python', './tools/train.py',
    # 'configs/fsaf/fsaf_r50_fpn_1x_coco.py',
    # '--cfg-options', 'train_cfg.max_epochs=100','mmobel.bbox_head.num_classes=8','scale=(1, 3, 640, 640)']#, f'model.backbone.init_cfg=None']
    # subprocess.run(command) # 训练
    # subprocess.run(['python', 'setup.py', 'install'])
    # c ='dict(layer='Conv2d', std=0.001, type='Normal')'
    command = ['python',
               './tools/train.py',
               'configs/solov2/solov2_r50_fpn_1x_coco.py',
               ]
    command = ['python',
               './tools/train.py',
               'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
               ]
    subprocess.run(command) # 训练


"""
# 计算fps
python  tools/analysis_tools/benchmark.py ^
       work_dirs/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco.py ^
       --checkpoint work_dirs/dino-4scale_r50_8xb2-12e_coco/best_coco_bbox_mAP_epoch_10.pth ^
       --launcher none
"""
"""
subprocess.run(['python', './tools/analysis_tools/get_flops.py', 'configs/detr/detr_r18_8xb2-500e_coco.py',
                '--cfg-options','scale=(1, 3, 640, 640) ']) # 计算参数量和计算量
                
# 不加载预训练模型 '--cfg-options' f'model.backbone.init_cfg=Normal'
    # python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
"""


"""
Two-stage-detectors
configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
configs/detr/detr_r50_8xb2-150e_coco.py
configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py
configs/libra_rcnn/libra-retinanet_r50_fpn_1x_coco.py
configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py
configs/cascade_rpn/cascade-rpn_fast-rcnn_r50-caffe_fpn_1x_coco.py
configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
    One-stage-detectors
configs/ssd/ssd512_coco.py
configs/retinanet/retinanet_r50_fpn_1x_coco.py
configs/rtmdet/rtmdet_s_8xb32-300e_coco.py
configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py
configs/fsaf/fsaf_r50_fpn_1x_coco.py
configs/nas_fpn/retinanet_r50_nasfpn_crop640-50e_coco.py
configs/nas_fcos/nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco.py
configs/yolox/yolox_s_8xb8-300e_coco.py
configs/yolof/yolof_r50-c5_8xb8-1x_coco.py
configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py
"""