# NonlocalMNIST

Since our collected ICH datasets are unavailable, we construct simulated datasets to partially exlain why UNet and UNETR achieve suboptimal results in ICH multi-class segmentation tasks. The results suggest that either network architectures or training strategies shall be modified for these two networks. 

## Create masks for MNIST simulated segmentation dataset
   Extract mnist_png.tar.gz in the repository, find train and test directories and run 
   ```bash
   python create_mask.py --dir_list address_of_training_directories address_of_testing_directories
   ```

## Requirements
   ```bash
   pip install numpy pillow cv2 torch torchvision albumentations  monai==1.2.0 timm==0.9.12
   ```
   To use DCN-UNet, follow [the link](https://github.com/OpenGVLab/InternImage/issues/84) to compile DCNv3.

## Results

## LRSR (segmentation, resize to 512 x 512)
   We test if network architectures are capatable of detecting long range relations. Run the following command:
    
   ```bash
   python main_train.py --model unet(unet_coord/unetr/sutm/swin_unetr_2d) --save-model-path address_of_save_model_path
   ```
    
   |     Method                   |  Params |  FLOPs  |  mIoU  |
   | :-------------------------:  | :-----: | :-----: | :----: |
   |     UNet                     |  34.53  |  262.17 |  74.51 |
   |     UNet (with  CoordConv)   |  34.53  |  262.70 |  79.00 |
   |     UNETR                    |  85.47  |  105.65 |  93.59 |
   |     SUTM-L (Ours)          |  11.97  |  25.65  |  97.66 |

## LTLRMoDR (segmentation, with multi-scale scaling, '0' digit as background label, randomly mix two digits in one image)
   This is a harder task. We set the label of '0' digit the same as the background label to simulate bones, tumors, califications in ICH dataset,
   and create inbalance dataset to simulate the data distribution of ICH dataset. Finally we randomly mix two digits in one image. Run the following command:

   ```bash
   python main_train.py --model unet(unet_coord/unetr/sutm/swin_unetr) --background True --balance extreme --class-weights 1 1.12 1.125 1.143 1.167 2 2.25 3.33 5 10 --transform center_scaling  --lr_scheduler True  --cutmix 0 --save-model-path address_of_save_model_path 
   ``` 

   |     Method                   |  Params |  FLOPs  |  mIoU  |
   | :-------------------------:  | :-----: | :-----: | :----: |
   |     UNETR                    |  85.47  |  105.65 |  73.13 |
   |     SwinUNETR                |   6.28  |   19.61 |  92.69 |
   |     DCN-UNet                 |  17.52  |   38.07 |  93.60 | 
   |     SUTM-L (Ours)          |  11.97  |  25.65  |  93.72 |

## Illustrations (red for wrong predictions)

## UNet in Task 1  
<img src="https://github.com/chihchiehchen/ICH-TEST-BENCHMARK/blob/main/NonlocalMNIST/pic/unet_check_394.png" height="240px" width="240px" /><img src="https://github.com/chihchiehchen/ICH-TEST-BENCHMARK/blob/main/NonlocalMNIST/pic/unet_check_4563.png" height="240px" width="240px" />

## UNETR in Task 2  
<img src="https://github.com/chihchiehchen/ICH-TEST-BENCHMARK/blob/main/NonlocalMNIST/pic/unetr_check_183.png" height="240px" width="240px" /><img src="https://github.com/chihchiehchen/ICH-TEST-BENCHMARK/blob/main/NonlocalMNIST/pic/unetr_check_1840.png" height="240px" width="240px" />
