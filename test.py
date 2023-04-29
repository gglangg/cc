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
# from model import NestedUNet
# from model import UNet_3Plus
# from model import DeepLabV3
from model.deeplabv3 import DeepLabV3

from model import resnet34
from evaluation import *
from utils.dataset import BasicDataset
from utils.custom_sampler import CustomSampler
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

PRED_MODEL = 'epoch_11_dsc_0.5675_best_val_dcsc_UnetwithResnet34_imgsize496_epoch20.pth'
IMG_INPUT = 'data/FA/test/imgs/'
GT_INPUT = 'data/FA/test/masks/'

THRESHOLD = 0.5
SCALE = 0.4

DICE_SAVE = 'output/model_output/'
# MASK_SAVE = os.path.join(DICE_SAVE,'predict/')
# PROB_SAVE = os.path.join(DICE_SAVE,'probs/')
# val_img = 'data/FA/val/imgs/'
# val_mask = 'data/FA/val/masks/'

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default= PRED_MODEL,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                            help='filenames of input images', default = IMG_INPUT)
    parser.add_argument('--gt', '-g', metavar='GT', nargs='+',
                            help='filenames of mask images', default = GT_INPUT)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default= THRESHOLD)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default= SCALE)

    return parser.parse_args()







# model = torch.load(PATH_TO_WEIGHTS)
# net = UNet(n_channels=1, n_classes=1)
net = DeepLabV3()
# net = resnet34(1, 1, False)
# net = NestedUNet(n_channels=1, n_classes=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
args = get_args()
net.load_state_dict(torch.load(args.model, map_location=device))
# net.eval()

batch_size = 1
img_scale=0.5

test = BasicDataset(IMG_INPUT, GT_INPUT, SCALE)
n_test = len(test)
# test_sampler = CustomSampler(train)

tot_ac = 0
tot_pc = 0
tot_se = 0
tot_sp = 0
tot_ap = 0
tot = 0


# train_loader = DataLoader(train, batch_size = batch_size, shuffle=False, sampler= train_sampler, num_workers=8, pin_memory=True)
test_loader = DataLoader(test, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

val_score, ac, pc, se, sp, ap, true_masks, masks_pred = eval_net(net, test_loader, device)

print("Test score:",val_score)