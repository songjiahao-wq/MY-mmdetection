import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv,FEM,RFCAConv
from models.Models.Attention.my_attention import *
from models.Models.Attention.MultiScaleAttention import *
# from models.ExpCode.FEM import *
from models.Models.fighting_model.backbone.inceptionNext import inceptionnext_tiny
from models.Models.fighting_model.attention.SKAttention import SKAttention

m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = RFCAConv(128,128, 3, 2)
results = profile(input=torch.randn(1, 128, 80, 80), ops=[m1, m2], n=1)




# Example usage
# input_tensor = torch.randn(8, 256, 32, 32)  # batch_size=8, in_channels=256, height=32, width=32
# model = FeatureEnhancementModule(in_channels=256)
# output_tensor = model(input_tensor)

