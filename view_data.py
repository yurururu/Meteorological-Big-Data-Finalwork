import torch
import matplotlib.pyplot as plt
import numpy as np

# 只读取数据路径
input_data = torch.load("./data/T1/val/input_val.pt", map_location='cpu')
target_data = torch.load("./data/T1/val/target_val.pt", map_location='cpu')

# 查看基础结构
print("Input type:", type(input_data))
print("Input shape:", input_data.shape if isinstance(input_data, torch.Tensor) else 'Not tensor')
print("Target shape:", target_data.shape if isinstance(target_data, torch.Tensor) else 'Not tensor')

# 如果是张量，取一张图像出来看看
if isinstance(input_data, torch.Tensor):
    sample = input_data[0]
    print("Single sample shape:", sample.shape)



# # 取出前10张图像
# images = target_data[:20, 0, 0, :, :]  # shape: [10, 64, 64]
#
# # 设置颜色映射为Blues，统一色阶范围（推荐你也可以看图像最大值来动态调整）
# # vmin = images.min().item()  # 例如 0.0
# # vmax = images.max().item()  # 例如 1.0 或 70等
values = np.array([45, 40, 35, 30, 25, 20, 15])
# # 定义颜色列表，每个数值对应一个颜色
# # 例如，这里我们使用从蓝色到白色的渐变
# # colors = ['white','lightblue','blue','dodgerblue']
#
# # 创建分段的颜色映射
# # cmap = LinearSegmentedColormap.from_list('my_cmap', colors,y_max)
# cmap = "Blues"
#
# # 可视化
# plt.figure(figsize=(12, 5))
# for i in range(20):
#     plt.subplot(4, 5, i+1)
#     im = plt.imshow(images[i].numpy(), cmap=cmap)
#     plt.title(f"Image {i+1}", fontsize=10)
#     plt.axis('off')
#
# # 可选添加统一colorbar（如果你觉得对比需要）
# cbar_ax = plt.axes([0.92, 0.15, 0.015, 0.7])  # colorbar位置
# plt.colorbar(im, cax=cbar_ax)
#
# plt.tight_layout(rect=[0, 0, 0.9, 1])  # 给colorbar留空间
# plt.show()

# 取前3个样本
samples = input_data[:3]  # shape: [3, 3, 1, 32, 32]

# 创建 3行×3列 子图（每行一个样本，每列一个时间帧）
fig, axs = plt.subplots(3, 3, figsize=(10, 8))

for i in range(3):  # 遍历3个样本
    for t in range(3):  # 每个样本3帧
        axs[i, t].imshow(samples[i][t][0].numpy(), cmap='Blues')
        axs[i, t].set_title(f'Sample {i+1}, T{t+1}', fontsize=10)
        axs[i, t].axis('off')

plt.tight_layout()
plt.show()