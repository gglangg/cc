import argparse
import copy
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
# from model import UNet
# from model import VNet
# from model import NestedUNet
# from model import UNet_3Plus
from model import resnet34
# from model import DeepLabHead
from model.deeplabv3 import DeepLabV3
# from model import layers
# from model import init_weights
from evaluation import *
from utils.dataset import BasicDataset
from utils.custom_sampler import CustomSampler
from torch.utils.data import DataLoader
import numpy as np


import warnings
warnings.filterwarnings("ignore")


train_img = 'data/FA/train/imgs/'
train_mask = 'data/FA/train/masks/'
val_img = 'data/FA/val/imgs/'
val_mask = 'data/FA/val/masks/'

dir_checkpoint = 'checkpoints/'
best_dsc = 0.0
best_epoch = 0
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of sochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type = float, nargs='?', default=0.0005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default= 0.4,
                        help='Downscaling factor of the images')
    parser.add_argument('-g', '--gradient-accumulations', dest='gradient_accumulations', type=int, default= 4,
                        help='gradient accumulations')

    return parser.parse_args()

def train_net(net,
              device,
              best_model_param,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              grad_accumulations = 2):

    # Get dataloader
    train = BasicDataset(train_img, train_mask, img_scale)
    val = BasicDataset(val_img, val_mask, img_scale)
    n_train = len(train)
    
    train_sampler = CustomSampler(train)
    
    train_loader = DataLoader(train, batch_size = batch_size, shuffle=False, sampler= train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size = batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    # print("-------------------------------------",train_loader.shape)
    global true_masks, masks_pred , best_dsc, best_epoch

    
    # Optimizer setting
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    bce = nn.BCEWithLogitsLoss()  
    min_loss = float('inf')
    
    # Train
    for epoch in range(1,epochs + 1):
        net.train()

        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_bce_loss = 0
        epoch_dsc = 0
        step = 1

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img',ascii=True,ncols=120) as pbar:
            
            for batch_i, batch in enumerate(train_loader):
                 # imgs       : input image(eye image)
                 # true_masks : ground truth
                imgs = batch['image']
                # print(imgs.shape)
                true_masks = batch['mask']
                # assert imgs.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'
                # assert imgs.shape[1] == net.input_channels, \
                #     f'Network has been defined with {net.input_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)
                # mask_type = torch.float32 if net.n_classes == 1 else torch.long
                # true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                dsc = dice_coeff(torch.sigmoid(masks_pred), true_masks)
                dice_loss = 1 - dsc
                bce_loss = bce(masks_pred, true_masks)
                
                # calculate loss
                loss =  bce_loss + dice_loss
                epoch_loss += loss.item()
                epoch_bce_loss += bce_loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_dsc  += dsc.item()
                
                # Back propogation
                loss.backward()
                
                if (batch_i + 1) % grad_accumulations == 0 or\
                    (len(train_loader) - batch_i < grad_accumulations and\
                      len(train_loader) - batch_i == 1):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                pbar.update(imgs.shape[0])
                step += 1
        
        val_score, ac, pc, se, sp, ap, true_masks, masks_pred = eval_net(net, val_loader, device)
        # print(val_score, ac, pc, se, sp, ap, true_masks, masks_pred)
        if val_score > best_dsc or (val_score == best_dsc and epoch_loss < min_loss):
            best_dsc = val_score
            best_epoch = epoch
            min_loss = epoch_loss
            best_model_params = copy.deepcopy(net.state_dict())

    torch.save(best_model_params, f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc_UnetwithResnet34_imgsize496_epoch20.pth')
    print("Best model name : " + f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc_UnetwithResnet34_imgsize496_epoch20.pth')
    

if __name__ == '__main__':
    args = get_args()
    
    print(torch.cuda.is_available())
    device = torch.device('cuda')
    print(device)
    # net = UNet(n_channels=1, n_classes=1)
    net = DeepLabV3()
    # print(net)
    # print(net.conv1)
    # net = VNet()
    # net = VNet().to(device)
    # net = NestedUNet()
    # net = resnet34(1, 1, False)
    # net = UNet_3Plus(in_channels=1, n_classes=1)
    net.to(device=device)
    # input = torch.randn(1, 1, 16, 128, 128) # BCDHW 
    # input = input.to(device)
    # # out = net(input) 
    # print("output.shape:", out.shape)

    best_model_params = copy.deepcopy(net.state_dict())

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  best_model_param = best_model_params,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  grad_accumulations = args.gradient_accumulations
                  )
    except KeyboardInterrupt:
        print('Saved interrupt')
        torch.save(best_model_params, f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc_UnetwithResnet34_imgsize600_epoch20.pth')
        print("Best model name : " + f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc_UnetwithResnet34_imgsize600_epoch20.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
