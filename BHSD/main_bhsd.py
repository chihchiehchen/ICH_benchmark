from utils import count_params, meanIOU
from sklearn.metrics import confusion_matrix
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD, AdamW
import json 
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.losses import DiceLoss

from network.monai_net import Resunet, UNETR_2d, U_Net_vanilla,UNETR_coord #,SwinUNETR_2d
from sklearn.metrics import confusion_matrix
from network.monai_net import  UNETR_2d,U_Net_vanilla,SwinUNETR_2d
from network.vanilla_unet import U_Net_coord
from network.archs import UNext,SUTM 
from ich_dataset_bhsd import ICH_B_Dataset
from ms_loss import levelsetLoss,gradientLoss2d
import torchvision.utils as v_utils
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    
    parser.add_argument('--batch-size', type=int, default=10)
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--lr_min', type=float, default=0.00001)
    parser.add_argument('--r-ratio', type=float, default=0.00001)
    parser.add_argument('--tv', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--img-key', type=int, default=255)
    parser.add_argument('--mixup', type=int, default=-100)
    parser.add_argument('--agc', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--model', type=str, choices=['unet','unet_coord','sutm','unetr','swinunetr'],
                        default='unext')  #todo 
    parser.add_argument('--opt', type=str, default='AdamW')
    # semi-supervised settings
    parser.add_argument('--in-ch', type=int, default=3)
    parser.add_argument('--nb-classes', type=int, default=6)
    parser.add_argument('--id-json', type=str, default='final_stat.json')
    parser.add_argument('--class-weights', type=float, default = [1.0, 15.0, 3.0, 10.0, 8.0, 5.0],nargs='+') #
    parser.add_argument('--save-model-path', type=str, default='./') 

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    valset = ICH_B_Dataset(mode = 'val')
    valloader = DataLoader(valset, batch_size=4,shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    global MODE
    MODE = 'train'

    trainset = ICH_B_Dataset(mode = MODE)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=True)
    
    model, optimizer = init_basic_elems(args)

    print('\nParams: %.1fM' % count_params(model))
    
    if args.lr_scheduler:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs,eta_min = args.lr_min)

    best_model, checkpoints, stats = train(model, trainloader, valloader, optimizer,args,lr_scheduler = lr_scheduler )
    
    json_new = os.path.join(args.save_model_path,args.id_json)
    with open(json_new,'w') as f:
        json.dump(stats, f)  

def init_basic_elems(args):
    model_zoo = {'unet': U_Net_vanilla,'unetr': UNETR_2d,'unet_coord':U_Net_coord,'sunext':SUTM, 'swinunetr': SwinUNETR_2d}
    model = model_zoo[args.model](in_ch =args.in_ch,out_ch = args.nb_classes)

    if args.opt == 'SGD':
        optimizer = SGD([{'params': [param for name, param in model.named_parameters()
                                 ],
                          'lr': args.lr }],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = AdamW([{'params': [param for name, param in model.named_parameters()
                                 ],
                          'lr': args.lr }],
                        lr=args.lr, weight_decay=1e-5)

    if not args.parallel:
        model = model.cuda() #no dataparrallel
    else:
        model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader, valloader, optimizer, args ,lr_scheduler = None):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []
    if args.nb_classes ==2:
        weights = [1,5] 
        
    else:
        weights = args.class_weights
    class_weights = torch.FloatTensor(weights).cuda()
    final_stat_dict = {}

    for epoch in range(args.epochs):
        final_stat_dict[epoch] = {}
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"] ,previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)
        
        for i, (element) in enumerate(tbar):
            
            l_img, l_mask,l_mask_name, l_stat = element[0],element[1],element[2],element[3]
            l_img, l_mask = l_img.cuda(), l_mask.cuda()
            
            
            l_pred = model(l_img)
            
            

            criterion = CrossEntropyLoss(weight = class_weights)
            criterionLS = levelsetLoss()
            criterionTV = gradientLoss2d()
            
            ce_loss = 3*criterion(l_pred, l_mask.long())
            loss_L = criterionLS(l_pred, l_mask.long(),args.nb_classes)
            loss_A = criterionTV(l_pred) *args.tv
            criterionD = DiceLoss(include_background=True,to_onehot_y=True, softmax= True,reduction='mean')
            d_loss = criterionD(l_pred, torch.unsqueeze(l_mask,1))
            loss = ce_loss + d_loss+ args.r_ratio*(loss_L+loss_A)
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            iters += 1
            
            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))
            tbar.set_description('current_loss: %.3f' % (loss))
      
            if i % 500 == 0 :
                print('ce_loss/d_loss:', ce_loss.item(), d_loss.item())
                print('ls_loss/TV_loss:', loss_L.item(), loss_A.item())
        metric = meanIOU(num_classes=args.nb_classes)

        model.eval()
        tbar = tqdm(valloader)
        pred_dict ={} 
        gt_dict = {}
        for i in range(6):
            pred_dict[i] =[] 
            gt_dict[i] = []

        with torch.no_grad():
            for i, (element) in enumerate(tbar):
                img, mask, l_mask_name, l_stat = element[0],element[1],element[2],element[3]
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                for iter in range(6):
                    gt_dict[iter] += list(l_stat[:,iter].cpu().numpy())
                for b_i in range(pred.size()[0]):
                    pred_slice = np.unique(pred[b_i].cpu().numpy())
                    for iter in range(1,6):
                        if iter in pred_slice:
                            pred_dict[iter]+= [1]
                        else:
                            pred_dict[iter]+= [0]
                    if len(pred_slice) > 1:
                        pred_dict[0]+= [1]
                    else:
                        pred_dict[0]+= [0]

        

        mIOU *= 100.0
        
        final_stat_dict[epoch]['miou'] = mIOU
        final_stat_dict[epoch]['conf'] = {}
        for i in range(6):
        
            conf = confusion_matrix(gt_dict[i],pred_dict[i])
            print(conf.tolist(),i ,conf[0][0]/(conf[0][0]+ conf[0][1]),conf[1][1]/(conf[1][0]+ conf[1][1]))
            final_stat_dict[epoch]['conf'][i] = [float(conf[0][0]/(conf[0][0]+ conf[0][1])), float(conf[1][1]/(conf[1][0]+ conf[1][1])),float(conf[0][0]),float(conf[0][1]),float(conf[1][0]),float(conf[1][1])]
            
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model,  previous_best)))
            previous_best = mIOU
            if args.parallel:
                torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model, mIOU)))
            else:
                torch.save(model.state_dict(),
                       os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model, mIOU)))

            best_model = deepcopy(model)
        
        
        if lr_scheduler:
            lr_scheduler.step()  
            print(epoch,'lr_scheduler.get_last_lr' ,lr_scheduler.get_last_lr()[0])  
    
    
    return best_model, checkpoints, final_stat_dict


if __name__ == '__main__':
    args = parse_args()



    print()
    print(args)

    main(args)


