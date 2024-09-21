import cv2,os, PIL,shutil
import numpy as np, nibabel as nib
from PIL import Image, ImageDraw, ImageFont


def nii_to_png(filename,output_dir,ww = 80,wc = 40):
        
        ds = nib.load(filename)
        
        
        np_data = np.array(ds.dataobj)
        lower_bound = wc - 0.5*ww
        upper_bound = wc + 0.5*ww

        np_data = np.where(np_data < lower_bound, lower_bound, np_data )
        np_data = np.where(np_data > upper_bound, upper_bound, np_data)

        np_data = 255/(upper_bound - lower_bound)*(np_data - lower_bound)
        np_data = np_data.astype('uint8')

        l = np_data.shape[2] 
        print(l,output_dir)
        for i in range(l):
            #print(output_dir, filename, filename..split('.')[0]+"_"+str(i)+'.png')

            path = os.path.join(output_dir,filename.split('/')[-1].split('.')[0]+"_"+str(i)+'.png')
            print(path)
            im = Image.fromarray(np.flip(np.swapaxes(np_data[:,:,i],0,1)))
            im.save(path)       
       

def nii_to_png_3ch(filename,output_dir,ww = 130,wc = 25):
        
        ds = nib.load(filename)
        
        
        np_data = np.array(ds.dataobj)
        lower_bound = wc - 0.5*ww
        upper_bound = wc + 0.5*ww

        np_data = np.where(np_data < lower_bound, lower_bound, np_data )
        np_data = np.where(np_data > upper_bound, upper_bound, np_data)

        np_data = 255/(upper_bound - lower_bound)*(np_data - lower_bound)
        np_data = np_data.astype('uint8')

        l = np_data.shape[2] 
        #print(l,output_dir)
        for i in range(l):
            #print(output_dir, filename, filename..split('.')[0]+"_"+str(i)+'.png')
            
            path = os.path.join(output_dir,filename.split('/')[-1].split('.')[0]+"_"+str(i)+'.png')
            prev = np.flip(np.swapaxes(np_data[:,:,max(i-1,0)],0,1))
            curr = np.flip(np.swapaxes(np_data[:,:,i],0,1))
            pro = np.flip(np.swapaxes(np_data[:,:,min(i+1,l-1)],0,1))
            #print(path)
            image = np.stack([prev,curr,pro],axis= 2)
            
            im = Image.fromarray(image)
            im.save(path) 


def label_to_png(filename,output_dir, mul = 51):
        
        ds = nib.load(filename)
        
        
        np_data = 51*np.array(ds.dataobj)
       
        np_data = np_data.astype('uint8')

        l = np_data.shape[2] 
        print(l,output_dir)
        for i in range(l):
            #print(output_dir, filename, filename..split('.')[0]+"_"+str(i)+'.png')
            path = os.path.join(output_dir,filename.split('/')[-1].split('.')[0]+"_"+str(i)+'.png')
            print(path)
            im = Image.fromarray(np.flip(np.swapaxes(np_data[:,:,i],0,1)))
            im.save(path)       



source = "your bhsd dir"
output = os.path.join(source, 'output')
img_dir = os.path.join(source,'images')
label_dir = os.path.join(source,'ground_truths')


os.makedirs(output,exist_ok= True)

image_list = os.listdir(img_dir)

for f in image_list:
    image_name = os.path.join(img_dir,f)
    label_name = os.path.join(label_dir,f)
    
    image_output = os.path.join(output, 'images',f.split('.')[0])
    label_output = os.path.join(output, 'labels',f.split('.')[0])
    os.makedirs(image_output,exist_ok= True)
    nii_to_png_3ch(image_name,image_output)

    