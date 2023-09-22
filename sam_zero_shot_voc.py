import os
import cv2
import torch
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import multiprocessing
import torchvision.transforms as transforms

import sys
sys.path.append('Tag2Text')
from class_names import *
from Tag2Text import inference_ram
from Tag2Text.models import tag2text
from segment_anything import build_sam, SamPredictor
import GroundingDINO.groundingdino.datasets.transforms as T
from grounded_sam_demo import get_grounding_output, load_model, load_image


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])
    return dataset_list

def perform(process_id, dataset_list, args):
    device_id = "cuda:{}".format(process_id)
    # load ram
    ram_model = tag2text.ram(pretrained=args.ram_ckpt,image_size=384,vit='swin_l').to(device_id)
    ram_model.eval()
    # load grounding
    model = load_model(args.config_file, args.grounded_ckpt, device=device_id)
    # load sam
    predictor = SamPredictor(build_sam(args.sam_ckpt).to(device_id))

    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    dino_class_names = ['aeroplane', 'bicycle', 'wild bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','dining table', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'monitor/tv']
    
    def check(tag):
        for i,classnames in enumerate(ram_class_names_voc):
            for classname in classnames:
                if classname in tag:
                    return i
        return -1
    
    # dict_name2id = {name: i for i, name in enumerate(class_names)}
    # dict_name2id_ram = {name: i for i, name in enumerate(ram_class_names)}
    databin = dataset_list[process_id]
    for img_name in tqdm(databin, position=process_id, desc=f'[PID{process_id}]'):
        # if os.path.exists(os.path.join(args.mask_dir,img_name.replace('jpg','png'))):
        #     continue
        img_path = os.path.join(args.img_root, img_name)

        image_pil, image = load_image(img_path)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), normalize])
        raw_image = image_pil.resize((384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device_id)

        res = inference_ram.inference(raw_image , ram_model)
        tags = res[0].replace('|', ',').split(',')
        tags = [tag.strip() for tag in tags]

        # print(img_name, tags)
        # continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        label_list = []
        mask_save = np.zeros(image.shape[:2], dtype=np.uint8)
        for tag in tags:
            label_id = check(tag)
            if label_id > -1 and (label_id not in label_list):
                label_list.append(label_id)
                image_pil, img = load_image(img_path)
                boxes_filt, pred_phrases = get_grounding_output(model, img, dino_class_names[label_id], args.box_threshold, args.text_threshold, device=device_id)

                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
                if transformed_boxes.shape[0] == 0:
                    continue
                masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes = transformed_boxes.to(device_id), multimask_output = False)

                masks = masks[:,0,:,:].cpu().numpy().astype(np.uint8)
                for i in range(masks.shape[0]):
                    mask_save = np.maximum(mask_save, masks[i]*(label_id+1))

        if len(label_list) == 0:
            print("{} not have valid object".format(img_name))
        imageio.imsave(os.path.join(args.mask_dir,img_name.replace('jpg','png')), mask_save)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_root', type=str, default='YOUR_DATA_PATH/VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument("--config_file", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--ram_ckpt", type=str, default="./checkpoints/ram_swin_large_14m.pth")
    parser.add_argument("--grounded_ckpt", type=str, default="./checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)

    # required args
    parser.add_argument('--split_file', type=str, default='./voc/train.txt')
    parser.add_argument('--mask_dir', type=str, default='./voc/mask_sam_zero_shot/')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    train_list = np.loadtxt(args.split_file, dtype=str)
    train_list = [x + '.jpg' for x in train_list]

    if not os.path.exists(args.mask_dir):
        os.makedirs(args.mask_dir)

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers, args=(dataset_list, args))

