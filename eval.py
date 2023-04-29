import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import *
from sklearn.metrics import average_precision_score, f1_score,recall_score,precision_score
from sklearn.utils.multiclass import type_of_target
import sys

def get_average_precision_score(preds, true_masks):
    
    ap = 0
    for i, pred in enumerate(preds):
        
        prob = pred.squeeze(0).cpu().numpy()
        prob = prob.reshape(prob.shape[0] * prob.shape[1],1)
        
        gt = true_masks[i].squeeze(0).cpu().numpy()
        gt = gt.reshape(gt.shape[0] * gt.shape[1],1)
        gt[gt > 0] = 1
          
        ap += average_precision_score(gt,prob)
    
    return ap / ( i + 1)
    

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    
    tot_ac = 0
    tot_pc = 0
    tot_se = 0
    tot_sp = 0
    tot_ap = 0
    tot = 0
    global true_masks, masks_pred 
    
    with tqdm(total=n_val, desc='Validation round', unit='img',ascii=True,ncols=120) as pbar:
    # with tqdm(total=n_val, desc='Validation round', unit='img') as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                # mask_pred, att_map = net(imgs)
                mask_pred = net(imgs)
                      
            pred = torch.sigmoid(mask_pred)   
            pred = (pred > 0.5).float()

            tot += dice_coeff(pred, true_masks).item()
            tot_ac += get_accuracy(pred, true_masks)
            tot_pc += get_precision(pred, true_masks)
            tot_se += get_sensitivity(pred, true_masks)
            tot_sp += get_specificity(pred, true_masks)
            tot_ap += get_average_precision_score(pred, true_masks)
            pbar.update(1)
            
    net.train()

    return tot / n_val, tot_ac/n_val, tot_pc/n_val, tot_se/n_val, tot_sp/n_val, tot_ap/n_val,\
           true_masks, mask_pred
