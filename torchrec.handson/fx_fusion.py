import torch
from torch.fx.experimental.optimization import fuse
from torchvision.models import resnet18

model = resnet18()
model.eval() # 必须在eval模型下fuse
print(model)
print("----")
fused_model = fuse(model)
print("fused_model")
print(fused_model)
