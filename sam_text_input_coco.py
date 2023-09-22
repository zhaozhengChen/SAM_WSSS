import os
import cv2
import torch
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from lxml import etree
from torch import multiprocessing
import torchvision.datasets as dset
from segment_anything import build_sam, SamPredictor
from grounded_sam_demo import get_grounding_output, load_model, load_image

def parse_xml_to_dict(xml):
    if len(xml) == 0: 
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child) 
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result: 
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

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

def perform(process_id, dataset_list, train_labels_dict, args):
    device_id = "cuda:{}".format(process_id)
    # load sam
    predictor = SamPredictor(build_sam(args.sam_ckpt).to(device_id))
    # load grounding
    model = load_model(args.config_file, args.grounded_ckpt, device=device_id)
    

    class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']
    
    databin = dataset_list[process_id]
    for img_name in tqdm(databin, position=process_id, desc=f'[PID{process_id}]'):
        img_path = os.path.join(args.img_root, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        labels = train_labels_dict[img_name]
        mask_save = np.zeros(image.shape[:2], dtype=np.uint8)
        for idx in range(len(class_names)):
            if labels[idx] > 0:
                image_pil, img = load_image(img_path)
                boxes_filt, pred_phrases = get_grounding_output(model, img, class_names[idx], args.box_threshold, args.text_threshold, device=device_id)

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
                    mask_save = np.maximum(mask_save, masks[i]*(idx+1))

        if np.sum(labels) == 0:
            print("{} not have valid object".format(img_name))
        imageio.imsave(os.path.join(args.mask_dir,img_name.replace('jpg','png')), mask_save)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_root', type=str, default='YOUR_DATA_PATH/MSCOCO/train2014/')
    parser.add_argument('--mask_dir', type=str, default='./mscoco/mask_sam_text_input/')
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument("--config_file", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--grounded_ckpt", type=str, default="./checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)

    args = parser.parse_args()
    os.makedirs(args.mask_dir, exist_ok=True)

    train_labels_dict = np.load('./mscoco/train_labels_dict.npy', allow_pickle=True).item()
    train_list = list(train_labels_dict.keys())

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, train_labels_dict, args)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers, args=(dataset_list,train_labels_dict, args))

