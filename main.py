import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from training import run_training
from training import evaluate_model
from utils import load_data
import numpy as np
import argparse
import os
import torch
from show_map import show_map

def add_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="era5_sr_data", help="choose a data set to use")
    parser.add_argument("--dataset", default="era5_twc", help="choose a data set to use")
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--model_id", default="test")
    # parser.add_argument("--model_id", default="models")
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--constraints", default="none") 
    parser.add_argument("--number_channels", default=32, type=int)
    parser.add_argument("--number_residual_blocks", default=4, type=int)
    parser.add_argument("--lr", default=0.001, help="learning rate", type=float)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--weight_decay", default=1e-9, type=float)
    parser.add_argument("--batch_size", default=64, type=int) 
    # parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--alpha", default=0.99, type=float)
    parser.add_argument("--test_val_train", default="val")  #使用哪个数据来进行验证
    parser.add_argument("--training_evalonly", default="training")
    parser.add_argument("--dim_channels", default=1, type=int)
    parser.add_argument("--adv_factor", default=0.0001, type=float)
    return parser.parse_args()

def main(args):
    #load data
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./data/prediction'):
        os.makedirs('./data/prediction')
    if args.training_evalonly == 'training':
        data = load_data(args)
        #run training
        run_training(args, data)    
    else:       
        data = load_data(args)
        #run training
        evaluate_model(data, args)


if __name__ == '__main__':
    print("path is:\t", os.getcwd())
    args = add_arguments()
    # args.dataset =  "era5_sr_data"
    # args.dataset = "T1"
    args.dataset = "TCW4"
    # args.dataset = "TCW8"
    # args.dataset = "TCW2"
    # args.upsampling_factor = 2
    args.upsampling_factor = 4
    # args.model = "cnn"
    args.model = "gan"
    # args.model = "convgru"
    """   
    # args.model_id = "twc_cnn_softmaxconstraints" #cnn softmax硬约束
    # args.model_id = "twc_cnn_softconstraints" #cnn软约束
    # args.model_id = "twc_cnn_noconstraints" #cnn无约束

    # args.model_id = "twc_gan_noconstraints"  # gan无约束
    # args.model_id = "twc_gan_softmaxconstraints"  # gan softmax约束
    # args.model_id = "twc_gan_softconstraints"  # gan 软约束

    # args.model_id = "twc_convgru_noconstraints"  # convgru 无约束
    # args.model_id = "twc_convgru_softmaxconstraints"  # convgru softmax约束"""

    args.model_id = "twc4_cnn_softmaxconstraints" #cnn softmax硬约束
    # args.model_id = "twc4_cnn_multconstraints"  # cnn mult硬约束
    # args.model_id = "twc4_cnn_addconstraints"  # cnn add硬约束
    # args.model_id = "twc4_cnn_softconstraints" #cnn软约束
    # args.model_id = "twc4_cnn_noconstraints" #cnn无约束

    # args.model_id = "twc4_gan_noconstraints"  # gan无约束
    # args.model_id = "twc4_gan_softmaxconstraints"  # gan softmax约束
    # args.model_id = "twc4_gan_softconstraints"  # gan 软约束
    args.model_id = "twc4_gan_addconstraints"  # gan add约束
    # args.model_id = "twc4_gan_multconstraints"  # gan mult约束

    # args.model_id = "t1_convgru_noconstraints"  # convgru 无约束
    # args.model_id = "t1_convgru_softconstraints"  # convgru soft约束
    # args.model_id = "t1_convgru_softmaxconstraints"  # convgru softmax约束

    # args.model_id = "twc2_cnn_softmaxconstraints" #cnn softmax硬约束
    # args.model_id = "twc2_cnn_multconstraints"  # cnn mult硬约束
    # args.model_id = "twc2_cnn_addconstraints"  # cnn add硬约束
    # args.model_id = "twc2_cnn_softconstraints" #cnn软约束
    # args.model_id = "twc2_cnn_noconstraints" #cnn无约束

    # args.model_id = "tcw8_convgru_noconstraints"  # convgru 无约束

    # args.model_id = "twc2_gan_noconstraints"  # gan无约束
    # args.model_id = "twc2_gan_softmaxconstraints"  # gan softmax约束
    # args.model_id = "twc2_gan_multconstraints"  # cnn mult硬约束
    # args.model_id = "twc2_gan_addconstraints"  # cnn add硬约束
    # args.model_id = "twc2_gan_softconstraints"  # gan 软约束

    # args.model_id = "twc8_gan_noconstraints"  # gan无约束
    # args.model_id = "twc8_gan_softmaxconstraints"  # gan softmax约束
    # args.model_id = "twc8_gan_softconstraints"  # gan 软约束
    # args.model_id = "twc8_gan_multconstraints"  # cnn mult硬约束
    # args.model_id = "twc8_gan_addconstraints"  # cnn add硬约束

    # args.model_id = "tcw4_convgru_noconstraints"  # convgru 无约束
    # args.model_id = "tcw4_convgru_softconstraints"  # convgru soft约束
    # args.model_id = "tcw4_convgru_softmaxconstraints"  # convgru softmax约束

    # args.model_id = "twc8_cnn_softmaxconstraints" #cnn softmax硬约束
    # args.model_id = "twc8_cnn_multconstraints"  # cnn mult硬约束
    # args.model_id = "twc8_cnn_addconstraints"  # cnn add硬约束
    # args.model_id = "twc8_cnn_softconstraints" #cnn软约束
    # args.model_id = "twc8_cnn_noconstraints" #cnn无约束

    # args.model_id = "test" #test
    # args.constraints = "softmax"
    # args.constraints = "soft"
    # args.constraints = "mult"
    # args.constraints = "none"
    args.constraints = "add"
    args.epochs = 20
    # args.epochs = 50
    # args.loss = "mass_constraints"

    # 是否评估
    args.training_evalonly = 'evalonly'
    # args.test_val_train = "test"
    # args.training_evalonly = 'training'

    # args.dim_channels = 3
    args.number_channels = 64
    # args.batch_size = 1

    # TCW可视化
    # show_map(args)

    main(args)