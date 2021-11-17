import torch
import numpy as np
import json
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import os
import cv2

# from nets.SegRank import SegRank
from nets.SegAttention import SegAttention
from nets.SegRankBaseline import SegRank
from data import PlacePulseDataset, AdaptTransform
from utils.image_gen import shape_attention, masked_attention_images, get_palette, segmentation_to_image, attention_to_images
from utils.metrics import mass_center
import seg_transforms
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= '../datasets/placepulse/'
OUTPUT_IMAGES_PATH = '../storage/output_images'
MODELS = {
    'wealthy':'../storage/models_seg/segattn_resnet_wealthy_segattn_model_0.6627424906132666.pth',
    'depressing': '../storage/models_seg/segattn_resnet_depressing_segattn_model_0.6433576233183856.pth',
    'safety': '../storage/models_seg/segattn_resnet_safety_segattn_model_0.6404490788126919.pth',
    'boring': '../storage/models_seg/segattn_resnet_boring_segattn_model_0.6105881211180124.pth',
    'lively': '../storage/models_seg/segattn_resnet_lively_segattn_model_0.633638033589923.pth',
    'beautiful':'../storage/models_seg/segattn_resnet_beautiful_segattn_model_0.6799476369495167.pth',
}

# MODELS = {
#     'wealthy':'../storage/models_seg/segrank_resnet_wealthy_15_drop2d_model_0.6189377346683355.pth',
#     'depressing': '../storage/models_seg/segrank_resnet_depressing_15_drop_2d_model_0.6230848281016442.pth',
#     'safety': '../storage/models_seg/segrank_resnet_safety_15_drop_2d_model_0.6024660951893551.pth',
#     'boring': '../storage/models_seg/segrank_resnet_boring_15_drop_2d_model_0.5851125776397516.pth',
#     'lively': '../storage/models_seg/segrank_resnet_lively_15_drop_2d_model_0.602934744576627.pth',
#     'beautiful':'../storage/models_seg/segrank_resnet_beautiful_15_drop_2d_model_0.6543367346938775.pth',
# }

# MODELS = {
#     'wealthy':'../storage/models_seg/sgrb_resnet_wealthy_sgrb_model_0.6099812265331664.pth',
#     'depressing': '../storage/models_seg/sgrb_resnet_depressing_sgrb_model_0.6173860239162929.pth',
#     'safety': '../storage/models_seg/sgrb_resnet_safety_sgrb_model_0.5923426305015353.pth',
#     'boring': '../storage/models_seg/sgrb_resnet_boring_sgrb_model_0.5802115683229814.pth',
#     'lively': '../storage/models_seg/sgrb_resnet_lively_sgrb_model_0.5896824702589223.pth',
#     'beautiful':'../storage/models_seg/sgrb_resnet_beautiful_sgrb_model_0.6493018259935553.pth',
# }

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 8
N_LAYERS=1
N_HEADS=1
SOFTMAX=True

CS_CLASSES = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]

transformers = transforms.Compose([
        AdaptTransform(seg_transforms.ToArray()),
        AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
        AdaptTransform(seg_transforms.Resize((244,244))),
        AdaptTransform(seg_transforms.ToTorchDims())
        ])

def generate_batch_stats(ids,forward_dict, output_dict, attribute, original_image):
    for i in range(len(ids)):
        if ids[i] not in output_dict:
            output_dict[ids[i]] = {'id':ids[i]}
        image_dict = output_dict[ids[i]]

        if attribute not in image_dict:
            image_dict[attribute] = {}
            image_dict[attribute]['score'] = float(forward_dict['output'].squeeze().cpu().numpy()[i])
        
        if 'images' not in image_dict:
                image_dict['images'] = {}
        images = image_dict['images']

        if 'segmentation' not in images:
            seg_img = segmentation_to_image(forward_dict['segmentation'][i].cpu(), original_image[i], get_palette())
            file_path = f'{segmentations_path}/{ids[i]}.png'
            cv2.imwrite(file_path, cv2.cvtColor(seg_img,cv2.COLOR_RGB2BGR)) 
            images['segmentation'] = f'segmentations/{ids[i]}.png'

        if 'attention' not in forward_dict:
            if 'object_metrics' not in image_dict[attribute]:
                image_dict[attribute]['object_metrics'] = process_images_no_attention(forward_dict['segmentation'][i].cpu(), original_image[i])
            
        else:
            if 'object_metrics' not in image_dict[attribute]:
                attention_map = shape_attention(forward_dict['attention'][0][i].cpu())
                image_dict[attribute]['object_metrics'] = process_images(attention_map, forward_dict['segmentation'][i].cpu(), original_image[i])

            if 'attention' not in images:
                images['attention'] = {}

            attention_dict = images['attention']
            if attribute not in attention_dict:
                attention_images, _ = attention_to_images(original_image[i], attention_map)
                file_path = f'{attentions_path}/{attribute}/{ids[i]}.png'
                cv2.imwrite(file_path, cv2.cvtColor(attention_images[0],cv2.COLOR_RGB2BGR))
                attention_dict[attribute] = f'attentions/{attribute}/{ids[i]}.png'

def process_images_no_attention(segmentation, original):
    seg = torch.from_numpy(np.asarray(np.argmax(segmentation, axis=0), dtype=np.uint8)).long()
    seg = torch.nn.functional.one_hot(seg, num_classes=19).permute([2,0,1]).float()
    total_seg = seg.sum()
    
    metrics = {}

    for idx, single_seg in enumerate(seg):
        sum_seg = single_seg.sum()
        if sum_seg != 0.0:
            idx_metrics = {
                'segmentation': float((sum_seg * 100 / total_seg).numpy()),
                'mass_center': mass_center(single_seg)
            }
            metrics[CS_CLASSES[idx]] = idx_metrics
    return metrics


def process_images(attention_map, segmentation, original):
    seg = torch.from_numpy(np.asarray(np.argmax(segmentation, axis=0), dtype=np.uint8)).long()
    seg = torch.nn.functional.one_hot(seg, num_classes=19).permute([2,0,1]).float()
    masked, _, _, _ = masked_attention_images(original,segmentation, attention_map)
    total_seg = seg.sum()
    sums = np.fromiter(map(lambda x: x.sum(), masked),dtype=np.float)
    total = masked.sum()
    sorted_idx = sums.argsort()[::-1]
    
    metrics = {}

    for i, idx in enumerate(sorted_idx):
        sum_seg = seg[idx].sum()
        if sum_seg != 0.0:
            idx_metrics = {
                'attention': masked[idx].sum() * 100 / total,
                'segmentation': float((sum_seg * 100 / total_seg).numpy()),
                'mass_center': mass_center(seg[idx])
            }
            idx_metrics['ratio'] = idx_metrics['attention'] / idx_metrics['segmentation']
            metrics[CS_CLASSES[idx]] = idx_metrics
    return metrics

f = open('log.txt', 'w')
image_hash = {}

segmentations_path = f'{OUTPUT_IMAGES_PATH}/segmentations'
if not os.path.exists(segmentations_path):
    os.makedirs(segmentations_path)

attentions_path = f'{OUTPUT_IMAGES_PATH}/attentions'
if not os.path.exists(attentions_path):
    os.makedirs(attentions_path)

for attribute, model in MODELS.items():

    if not os.path.exists(f'{OUTPUT_IMAGES_PATH}/attentions/{attribute}'):
        os.makedirs(f'{OUTPUT_IMAGES_PATH}/attentions/{attribute}')

    dataset=PlacePulseDataset(
        f'{PLACE_PULSE_PATH}/{attribute}/val.csv',
        IMAGES_PATH,
        transform=transformers,
        equal=True,
        return_ids=True,
        return_images=True
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, drop_last=True)

    net = SegAttention(model=models.resnet50, image_size=(244,244), n_layers=N_LAYERS, n_heads=N_HEADS, softmax=SOFTMAX)
    # net = SegRank(image_size=(244,244), n_layers=N_LAYERS, n_heads=N_HEADS, softmax=SOFTMAX)
    # net = SegRank(image_size=(244,244),softmax=SOFTMAX)
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device)
    net.eval()
    f.write(f'loaded {model}\n')
    f.flush()
    
    for index,batch in enumerate(loader):
        input_left = batch['left_image'].to(device)
        input_right = batch['right_image'].to(device)
        label = batch['winner'].to(device)
        left_id = batch['left_id']
        right_id = batch['right_id']
        left_original = batch['left_image_original']
        right_original = batch['right_image_original']
        with torch.no_grad():
            forward_dict = net(input_left,input_right)
        generate_batch_stats(left_id, forward_dict['left'], image_hash, attribute, left_original)
        generate_batch_stats(right_id, forward_dict['right'], image_hash, attribute, right_original)
        f.write(f'{index}/{len(loader)}\n')
        f.flush()

    with open('attnsegrank_stats.jsonl', 'w') as outfile:
        for _id, _hash in image_hash.items():
            json.dump(_hash, outfile)
            outfile.write('\n')

f.close()
