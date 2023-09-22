# SAM_WSSS
SAM for Weakly-Supervised Semantic Segmentation (WSSS).

## Environment
Our code is based on [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). Please refer to it on installation.

## Usage
### PASCAL VOC
SAM (text input)
```
python sam_text_input_voc.py --img_root YOUR_DATA_PATH --mask_dir YOUR_OUTPUT_PATH
```

SAM (tero-shot)
```
python sam_zero_shot_voc.py --img_root YOUR_DATA_PATH --mask_dir YOUR_OUTPUT_PATH
```

### MS COCO
SAM (text input)
```
python sam_text_input_coco.py --img_root YOUR_DATA_PATH --mask_dir YOUR_OUTPUT_PATH
```

SAM (tero-shot)
```
python sam_zero_shot_coco.py --img_root YOUR_DATA_PATH --mask_dir YOUR_OUTPUT_PATH
```

## Acknowledgment
Thanks to [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).
