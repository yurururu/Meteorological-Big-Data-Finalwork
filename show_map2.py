import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.optim as optim
import torch.nn as nn
import models
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
def show_map(args):
    i = 10
    print("path is:\t", os.getcwd())
    home_path = os.getcwd()
    input_train = torch.load(home_path + '/data/' + args.dataset + '/train/input_train.pt')
    # max1 = input_train.max()
    target_train = torch.load(home_path + '/data/' + args.dataset + '/train/target_train.pt')
    # 测试集
    input_val = torch.load('./data/'+args.dataset+'/'+ args.test_val_train+'/input_'+ args.test_val_train+'.pt')
    target_val = torch.load('./data/'+args.dataset+'/'+ args.test_val_train+'/target_'+ args.test_val_train+'.pt')
    # 预测
    if args.model == 'gan':
        en_pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'_ensemble.pt')
        pred = torch.mean(en_pred, dim=1)
        en_pred = en_pred.detach().cpu().numpy()
        y_pred = pred[i][0][0]
    else:
        pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'.pt')
        y_pred = pred[i][0][0]
        y_pred1 = pred[i][1][0]
        y_pred2 = pred[i][2][0]

    x1 = input_val[i][0][0]
    x2 = input_val[i][1][0]
    x3 = input_val[i][2][0]

    y1 = target_val[i][0][0]
    y2 = target_val[i][1][0]
    y3 = target_val[i][2][0]

    # dataset_name = ["TCW8", "TCW4", "TCW2"]
    #
    # tcw8 = torch.load('./data/' + dataset_name[0] + '/val/input_val.pt')[0][0][0]
    # tcw4 = torch.load('./data/' + dataset_name[1] + '/val/input_val.pt')[0][0][0]
    # tcw2 = torch.load('./data/' + dataset_name[2] + '/val/input_val.pt')[0][0][0]
    # target = torch.load('./data/' + dataset_name[0] + '/val/target_val.pt')[0][0][0]

    # x1 = input_val[i][0][0]
    # y1 = target_val[i][0][0]
    # print(input_val[0])
    # x_max = x1.max()
    # y_max = y1.max()
    # print("*****",x_max,"****",x_min)
    # 您提供的数值列表
    values = np.array([45, 40, 35, 30, 25, 20, 15])
    # 定义颜色列表，每个数值对应一个颜色
    # 例如，这里我们使用从蓝色到白色的渐变
    # colors = ['white','lightblue','blue','dodgerblue']

    # 创建分段的颜色映射
    # cmap = LinearSegmentedColormap.from_list('my_cmap', colors,y_max)
    cmap = "Blues"

    fig, axs = plt.subplots(3, 3, figsize=(10, 8))  # 创建一个包含两个子图的图表
    # fig, axs = plt.subplots(1, 3, figsize=(15, 10))  # 创建一个包含两个子图的图表

    # # 在第一个子图中显示x1
    # # 在第一个子图中显示x1
    # im1 = axs[0].imshow(x1, cmap=cmap)
    # axs[0].set_title('x1', fontsize=16, fontweight='bold')
    # # im1 = axs[0].imshow(tcw8, cmap=cmap)
    # # axs[0].set_title('tcw8')
    # # 为第一个子图添加颜色条
    # # cbar1 = fig.colorbar(im1, ax=axs[0])
    # # axs[0].set_aspect('auto')
    #
    # # 在第二个子图中显示y1
    # im2 = axs[1].imshow(y1, cmap=cmap)
    # axs[1].set_title('y1', fontsize=16, fontweight='bold')
    # # im2 = axs[1].imshow(tcw4, cmap=cmap)
    # # axs[1].set_title('tcw4')
    # # 为第二个子图添加颜色条
    # # cbar2 = fig.colorbar(im2, ax=axs[1])
    # # axs[1].set_aspect('auto')  # 设置第二个子图的显示比例为'auto'
    # im3 = axs[2].imshow(y_pred, cmap=cmap)
    # axs[2].set_title('y_pred', fontsize=16, fontweight='bold')
    # # im3 = axs[2].imshow(tcw2, cmap=cmap)
    # # axs[2].set_title('tcw2')
    # # im4 = axs[3].imshow(target, cmap=cmap)
    # # axs[3].set_title('target')

    # 时间序列可视化
    im1 = axs[0,0].imshow(x1, cmap=cmap)
    axs[0,0].set_title('LR_frame1')
    im2 = axs[0,1].imshow(x2, cmap=cmap)
    axs[0,1].set_title('LR_frame2')
    im3 = axs[0,2].imshow(x3, cmap=cmap)
    axs[0,2].set_title('LR_frame3')
    im4 = axs[1,0].imshow(y1, cmap=cmap)
    axs[1,0].set_title('HR_frame1')
    im5 = axs[1,1].imshow(y2, cmap=cmap)
    axs[1,1].set_title('HR_frame2')
    im6 = axs[1,2].imshow(y3, cmap=cmap)
    axs[1,2].set_title('HR_frame3')
    im7 = axs[2,0].imshow(y_pred, cmap=cmap)
    axs[2,0].set_title('None_1')
    im8 = axs[2,1].imshow(y_pred1, cmap=cmap)
    axs[2,1].set_title('None_2')
    im9 = axs[2,2].imshow(y_pred2, cmap=cmap)
    axs[2,2].set_title('None_3')
    cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cb = fig.colorbar(im4, shrink=1, cax=cax, orientation='vertical')



    for ax1 in axs:
        for ax in ax1:
            ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    # for ax in axs:
    #     ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)


    plt.tight_layout()  # 调整子图间距
    plt.show()

if __name__ == '__main__':
    from main import add_arguments
    args = add_arguments()
    args.dataset = 'T1'
    # args.dataset = 'TCW2'
    # args.dataset = 'TCW4'
    # args.dataset = 'TCW8'
    args.model = 'convgru'
    # args.model = 'cnn'
    # args.model = 'gan'
    # args.model_id = 'tcw4_convgru_noconstraints'
    # args.model_id = 'tcw4_convgru_softmaxconstraints'
    # args.model_id = 'tcw4_convgru_softconstraints'
    # args.model_id = 'twc2_cnn_noconstraints'
    # args.model_id = 'twc2_cnn_softmaxconstraints'
    # args.model_id = 'twc4_cnn_noconstraints'
    # args.model_id = 'twc4_cnn_softmaxconstraints'
    # args.model_id = 'twc4_gan_noconstraints'
    # args.model_id = 'twc4_gan_softmaxconstraints'
    # args.model_id = 'twc4_gan_softconstraints'
    # args.model_id = 'twc8_gan_noconstraints'
    # args.model_id = 'twc8_gan_softmaxconstraints'
    # args.model_id = 'twc4_gan_addconstraints'
    # args.model_id = 't1_convgru_softmaxconstraints'
    args.model_id = 't1_convgru_noconstraints'
    args.test_val_train = 'val'
    show_map(args)