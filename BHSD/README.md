# BHSD

Here's files for BHSD training and validation:

bhsd_to_png.py : preprocess nii.gz files to png files.
slice_stat.json: annotations of each slices.
stat_report.json: the areas of subtype haemorrhage of each case.

Run the following command:
    
```bash
python main_bhsd.py --model unet(unet_coord/unetr/sutm/swin_unetr_2d) --save-model-path address_of_save_model_path
```
