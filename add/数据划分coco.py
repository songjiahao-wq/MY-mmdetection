import os
import shutil
from sklearn.model_selection import train_test_split

# 指定 pic 和 json 文件夹的路径
pic_dir = r'F:\BaiduNetdiskDownload\train_maskrcnn\mask-rcnn_RoadDefectDataset\pic'
json_dir = r'F:\BaiduNetdiskDownload\train_maskrcnn\mask-rcnn_RoadDefectDataset\json'

# 获取 pic 和 json 文件夹中所有文件的列表
pic_files = [os.path.join(pic_dir, f) for f in os.listdir(pic_dir) if os.path.isfile(os.path.join(pic_dir, f))]
json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f))]

# 将文件划分为训练集和验证集 (70% train, 30% validation)
pic_train, pic_val, json_train, json_val = train_test_split(pic_files, json_files, test_size=0.3, random_state=42)

# 创建训练和验证目录来存储文件
train_dir = r'F:\BaiduNetdiskDownload\train_maskrcnn\mask-rcnn_RoadDefectDataset\train'
val_dir = r'F:\BaiduNetdiskDownload\train_maskrcnn\mask-rcnn_RoadDefectDataset\val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'pic'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'json'), exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(val_dir, 'pic'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'json'), exist_ok=True)

# 将文件移动到新的训练和验证目录
for f in pic_train:
    shutil.move(f, os.path.join(train_dir, 'pic'))
for f in pic_val:
    shutil.move(f, os.path.join(val_dir, 'pic'))
for f in json_train:
    shutil.move(f, os.path.join(train_dir, 'json'))
for f in json_val:
    shutil.move(f, os.path.join(val_dir, 'json'))
