from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os, numpy as np
from PIL import Image
from utils.utils import get_subdf
from natsort import natsorted
import PIL,cv2,json 

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop, 
    Blur,   
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    Rotate,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    ShiftScaleRotate     
)


class ICH_B_Dataset(Dataset):
    def __init__(self,  root_dir="/home/chihchieh/BHSD/output",img_size=512,mode ='train',nb_classes=6): #/home/chihchieh/projects/ich_ssl_train/selected_list_0822.csv
        # mask_dir : doctor annotations, around 30 cases, pretrain_dir: kaggle_ready , around 1000 cases(900 with ich), unmask_dir: imgs without masks
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        self.img_size = img_size
        self.case_stat = json.load(open('./stat_report.json', 'rb'))
        self.slice_stat = json.load(open('./slice_stat.json', 'rb'))
        self.id_to_label = {1:'edh',2:'ich',5:'ivh',4:'sah',3:'sdh'}
        self.nb_classes = nb_classes
        self.mode = mode
       
        self.key_ratio = round(255/(self.nb_classes -1))
        self.cand = self.cal_cand()
        #print(os.listdir(self.img_dir))
        #print('wait')
        #print(self.cand)
        if mode == 'train':
            self.dir_list = [x for x in os.listdir(self.img_dir) if x+'.nii.gz' not in self.cand]
        else:
            self.dir_list = [ x.replace('.nii.gz','') for x in list(self.cand)]
        self.preprocessing()
        print('dir length: ', len(self.dir_list))
    def cal_cand(self): # the way we pick the val set
        cand = set()
        ord_list = [1,5,3,4,2]
        pick_num = [11,11,11,11,11]
        for i in range(len(ord_list)):
            count = 0
            ord = ord_list[i]
            num = pick_num[i]

            final = {}
            for key in self.case_stat:
        
                final[key] = self.case_stat[key][ord]
            pred_list = sorted(final.items(), key=lambda item: item[1])
            cand_list =[ pred_list[i][0] for i in range(len(pred_list)) if (pred_list[i][0] not in cand and pred_list[i][1] >0)  ]
            l = len(cand_list)
            #print(i,l,len(pred_list))
            assert l >= num + 1
            for j in range(0,l, l//(num+1)):
                index = min(l-1, j)
                id = cand_list[index]
                cand.add(id)
                count += 1
                if count == num:
                    break
        print('cand length: ', len(cand))   
        return cand
    
    def preprocessing(self):
        self.final_list = []
        for dirname in self.dir_list:
            dirpath = os.path.join(self.img_dir, dirname)
            for i_name in os.listdir(dirpath):
                final_dict = {}
                img_name = os.path.join(dirpath,i_name)
                init_image = Image.open(img_name)
                label_name = img_name.replace(self.img_dir,self.mask_dir)
                targetmask = (Image.open(label_name).convert('L'))

                if init_image.size[0] != self.img_size or init_image.size[1] != self.img_size:
                
                    init_image=init_image.resize((self.img_size, self.img_size))
                    targetmask = targetmask.resize((self.img_size, self.img_size),resample =  PIL.Image.NEAREST)  

                init_image = np.array(init_image)
            
                targetmask = np.array(targetmask)

                targetmask = np.round((targetmask*(1/self.key_ratio)))

                final_dict['filename'] =  img_name
                final_dict['image'] =  init_image 
                final_dict['mask'] = targetmask 
                final_dict['stat'] = self.slice_stat[i_name]
                self.final_list.append(final_dict)

    def __len__(self):
        
        print('current len', len(self.final_list))
        
        return len(self.final_list)

    def __getitem__(self, index):
        final_dict = self.final_list[index]
        init_image = final_dict['image']
        targetmask = final_dict['mask']
        image_name = final_dict['filename']
        stat = final_dict['stat']
        stat[0] = max(stat[1:])


        aug=Compose([RandomBrightnessContrast(p= 0.5)  ,ShiftScaleRotate(scale_limit=(0,0.20), rotate_limit=45, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),Blur(p=0.5)],p=1)   
        augmented = aug(image=init_image, mask=targetmask)
        tr=transforms.Compose([transforms.ToTensor() ])
        if self.mode == 'train':
            aug_image= tr(augmented["image"]) 
                
            aug_target_mask = augmented['mask']
        
            return aug_image, aug_target_mask, image_name, np.array(stat)


        else:
            
            tr = transforms.Compose([   transforms.ToTensor()    ])
            return tr(init_image), targetmask,  image_name, np.array(stat)


        

    

