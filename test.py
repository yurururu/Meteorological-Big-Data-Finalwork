import torch
print(torch.__version__)               # 应为 2.3.0
print(torch.cuda.is_available())       # 应为 True
print(torch.version.cuda)              # 应为 '12.1'