import os
import cv2
import torch
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from lxml import etree
from torch import multiprocessing
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

def perform(process_id, dataset_list, args):
    device_id = "cuda:{}".format(process_id)
    # load sam
    predictor = SamPredictor(build_sam(args.sam_ckpt).to(device_id))
    # load grounding
    model = load_model(args.config_file, args.grounded_ckpt, device=device_id)
    

    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    dino_class_names = ['aeroplane', 'bicycle', 'wild bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','dining table', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'monitor/tv']

    dict_name2id = {name: i for i, name in enumerate(class_names)}
    databin = dataset_list[process_id]
    for img_name in tqdm(databin, position=process_id, desc=f'[PID{process_id}]'):
        if os.path.exists(os.path.join(args.mask_dir,img_name.replace('jpg','png'))):
            continue
        img_path = os.path.join(args.img_root, img_name)
        xmlfile = img_path.replace('/JPEGImages', '/Annotations')
        xmlfile = xmlfile.replace('.jpg', '.xml')
        with open(xmlfile) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str) 
        data = parse_xml_to_dict(xml)["annotation"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        label_list = []
        mask_save = np.zeros(image.shape[:2], dtype=np.uint8)
        for obj in data["object"]:
            if obj["name"] not in label_list:
                label_list.append(obj["name"])
                label_id = dict_name2id[obj["name"]]
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
    parser.add_argument('--split_file', type=str, default='./voc/train_aug.txt')
    parser.add_argument('--mask_dir', type=str, default='./voc/mask_sam_text_input/')
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument("--config_file", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--grounded_ckpt", type=str, default="./checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)

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

