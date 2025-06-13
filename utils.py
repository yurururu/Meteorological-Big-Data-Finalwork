import os
import torch
import torch.optim as optim
import torch.nn as nn
import models
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
# device = 'cuda'
device = torch.device("cuda")

def load_data(args):
    print("path is:\t", os.getcwd())
    home_path = os.getcwd()
    input_train = torch.load(home_path+'/data/'+args.dataset+'/train/input_train.pt')
    target_train = torch.load(home_path+'/data/'+args.dataset+'/train/target_train.pt')


    # 查看tensor里面的图片
    # test1 = input_train[0]
    # # test1 = target_train[0]
    # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    # img = test1[0]
    # pic = toPIL(img)
    # pic.save('random.jpg')

    if args.test_val_train == 'test':
        input_val = torch.load(home_path+'/data/'+args.dataset+'/test/input_test.pt')
        target_val = torch.load(home_path+'/data/'+args.dataset+'/test/target_test.pt')
    elif args.test_val_train == 'val':
        input_val = torch.load(home_path+'/data/'+args.dataset+'/val/input_val.pt')
        target_val = torch.load(home_path+'/data/'+args.dataset+'/val/target_val.pt')
    elif args.test_val_train == 'train':
        input_val = input_train
        target_val = target_train


    # 查看tensor里面的图片
    # test1 = input_val[0]
    #
    # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    # img1 = test1[0]
    # pic = toPIL(img1)
    # pic.save('random1.jpg')
    #
    # test2 = target_val[0]
    # img2 = test2[0]
    # pic = toPIL(img2)
    # pic.save('random2.jpg')
    #define dimesions
    global train_shape_in , train_shape_out, val_shape_in, val_shape_in
    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    val_shape_in = input_val.shape
    val_shape_out = target_val.shape
    #mean, std, max
    mean = target_train.mean()
    std = target_train.std()
    global max_val, min_val
    max_val = torch.zeros((args.dim_channels,1))
    min_val = torch.zeros((args.dim_channels,1))
    if args.dim_channels == 3:
        for i in range(args.dim_channels):
            max_val[i] = target_train[:, i, 0, ...].max()
            min_val[i] = target_train[:, i, 0, ...].min()

        # transform data(数据转换，归一化处理)
        for i in range(args.dim_channels):
            input_train[:, i, 0, ...] = (input_train[:, i, 0, ...] - min_val[i]) / (max_val[i] - min_val[i])
            target_train[:, i, 0, ...] = (target_train[:, i, 0, ...] - min_val[i]) / (max_val[i] - min_val[i])
            input_val[:, i, 0, ...] = (input_val[:, i, 0, ...] - min_val[i]) / (max_val[i] - min_val[i])
            target_val[:, i, 0, ...] = (target_val[:, i, 0, ...] - min_val[i]) / (max_val[i] - min_val[i])

    else:
        for i in range(args.dim_channels):
            max_val[i] = target_train[:,0,i,...].max()
            min_val[i] = target_train[:,0,i,...].min()

        #transform data(数据转换，归一化处理)
        for i in range(args.dim_channels):
            input_train[:,0,i,...] = (input_train[:,0,i,...]-min_val[i]) /(max_val[i]-min_val[i])
            target_train[:,0,i,...] = (target_train[:,0,i,...] -min_val[i])/(max_val[i]-min_val[i])
            input_val[:,0,i,...] = (input_val[:,0,i,...]-min_val[i])/(max_val[i]-min_val[i])
            target_val[:,0,i,...] = (target_val[:,0,i,...]-min_val[i])/(max_val[i]-min_val[i])
    
    train_data = TensorDataset(input_train,  target_train)
    val_data = TensorDataset(input_val, target_val)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) 
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    return [train, val, mean, std, max_val, train_shape_in, train_shape_out, val_shape_in, val_shape_out]

def load_model(args, discriminator=False):
    if discriminator:
        model = models.Discriminator()
    else:
        if args.model == 'convgru':
            model = models.ConvGRUGeneratorDet( number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, time_steps=args.dim_channels, constraints=args.constraints)
        elif args.model == 'flowconvgru':
            model = models.TimeEndToEndModel( number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, time_steps=3, constraints=args.constraints)
        elif args.model == 'gan':
            model = models.ResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=(args.model=='gan'), constraints=args.constraints, dim=args.dim_channels)
        elif args.model == 'cnn':
            model = models.ResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=(args.model=='gan'), constraints=args.constraints, dim=args.dim_channels)
    model = model.to(device)
    return model

def get_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_criterion(args, discriminator=False):
    # 如果是判别器则为True，生成器则为False
    if discriminator:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    return criterion

def mass_loss(output, in_val, args):
    ds_out = torch.nn.functional.avg_pool2d(output[:,0,0,:,:], args.upsampling_factor)
    return torch.nn.functional.mse_loss(ds_out, in_val)

def get_loss(output, true_value, in_val, args):
    if args.loss == 'mass_constraints':#质量约束损失函数
        return args.alpha*mass_loss(output, in_val[:,0,0,...], args) + (1-args.alpha)*torch.nn.functional.mse_loss(output, true_value)
    else:
        return torch.nn.functional.mse_loss(output, true_value)
    
def process_for_training(inputs, targets):
    #要使用数据进行操作时，将数据移入GPU
    inputs = inputs.to(device)            
    targets = targets.to(device)
    return inputs,  targets

def process_for_eval(outputs, targets, mean, std, max_val, args): 
    if args.model == 'gan':
        outputs[:,:,0,0,...] = outputs[:,0,0,...]*(max_val[0].to(device)-min_val[0].to(device))+min_val[0].to(device) 
        targets[:,0,0,...] = targets[:,0,0,...]*(max_val[0].to(device)-min_val[0].to(device))+min_val[0].to(device)
    else:
        if args.dim_channels == 3:
            for i in range(args.dim_channels):
                outputs[:, i, 0, ...] = outputs[:, i, 0, ...] * (max_val[i].to(device) - min_val[i].to(device)) + \
                                        min_val[i].to(device)
                targets[:, i, 0, ...] = targets[:, i, 0, ...] * (max_val[i].to(device) - min_val[i].to(device)) + \
                                        min_val[i].to(device)
        else:
            for i in range(args.dim_channels):
                outputs[:,0,i,...] = outputs[:,0,i,...]*(max_val[i].to(device)-min_val[i].to(device))+min_val[i].to(device)
                targets[:,0,i,...] = targets[:,0,i,...]*(max_val[i].to(device)-min_val[i].to(device))+min_val[i].to(device)
    return outputs, targets

def is_gan(args):
    if args.model == 'gan':
        return True
    else: 
        return False
    




    
