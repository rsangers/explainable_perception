import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader
from skimage import io
import torch.distributed as dist
import os
import cv2

from nets.SegRank import SegRank
from nets.SegAttention import SegAttention
from data import PlacePulseDataset, AdaptTransform
from utils.image_gen import shape_attention, clear_zeros
import seg_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= '../datasets/placepulse'
OUTPUT_IMAGES_PATH = '../storage/output_images/segrank/average_attentions'
OUTPUT_CLASS_IMAGES_PATH = '../storage/output_images/segrank/class_attentions/'
# MODELS = {
#     'wealthy':'../storage/models_seg/segattn_resnet_wealthy_segattn_model_0.6627424906132666.pth',
#     'depressing': '../storage/models_seg/segattn_resnet_depressing_segattn_model_0.6433576233183856.pth',
#     'safety': '../storage/models_seg/segattn_resnet_safety_segattn_model_0.6404490788126919.pth',
#     'boring': '../storage/models_seg/segattn_resnet_boring_segattn_model_0.6105881211180124.pth',
#     'lively': '../storage/models_seg/segattn_resnet_lively_segattn_model_0.633638033589923.pth',
#     'beautiful':'../storage/models_seg/segattn_resnet_beautiful_segattn_model_0.6799476369495167.pth',
# }

MODELS = {
    'wealthy':'../storage/models_seg/segrank_resnet_wealthy_15_drop2d_model_0.6189377346683355.pth',
    'depressing': '../storage/models_seg/segrank_resnet_depressing_15_drop_2d_model_0.6230848281016442.pth',
    'safety': '../storage/models_seg/segrank_resnet_safety_15_drop_2d_model_0.6024660951893551.pth',
    'boring': '../storage/models_seg/segrank_resnet_boring_15_drop_2d_model_0.5851125776397516.pth',
    'lively': '../storage/models_seg/segrank_resnet_lively_15_drop_2d_model_0.602934744576627.pth',
    'beautiful':'../storage/models_seg/segrank_resnet_beautiful_15_drop_2d_model_0.6543367346938775.pth',
}

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 32
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

def get_image_batch(image_list,start,end,image_dir):
    batch = image_list[start:end]
    return np.array([ io.imread(f'{image_dir}/{x}') for x in batch])

def transform_batch(batch):
    transform = transforms.Compose([
        seg_transforms.ToArray(),
        seg_transforms.SubstractMean(IMG_MEAN),
        seg_transforms.Resize((244,244)),
        seg_transforms.ToTorchDims()
        ])
    result = torch.stack([ torch.from_numpy(transform(image)) for image in batch])
    return result

def normalize_attention(attention):
    input_min = np.min(attention)
    input_max = np.max(attention)
    return ((attention - input_min)/(input_max - input_min) * 255).astype('uint8')

def clear_outliers(attention):
    mean = attention.mean()
    std = attention.std()
    _min = mean - 2.6 * std
    _max = mean + 2.6 * std
    attention = np.where(attention >= _min, attention, _min)
    return np.where(attention <= _max, attention, _max)

images = os.listdir(IMAGES_PATH)

for attribute, model in MODELS.items():
    if not os.path.exists(f'{OUTPUT_IMAGES_PATH}/{attribute}'):
        os.makedirs(f'{OUTPUT_IMAGES_PATH}/{attribute}')

    if not os.path.exists(f'{OUTPUT_CLASS_IMAGES_PATH}/{attribute}'):
        os.makedirs(f'{OUTPUT_CLASS_IMAGES_PATH}/{attribute}')


    net = SegRank(image_size=(244,244), n_layers=N_LAYERS, n_heads=N_HEADS, softmax=SOFTMAX)
    # net = SegAttention(model=models.resnet50, image_size=(244,244), n_layers=N_LAYERS, n_heads=N_HEADS, softmax=SOFTMAX)
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device)
    net.eval()
    print(f'loaded {model}\n')
    attentions = None
    total_seg = None
    count = 0
    for index in range(0,len(images),BATCH_SIZE):
        start = index
        end = min((index + BATCH_SIZE,len(images)))
        batch = transform_batch(get_image_batch(images,start,end,IMAGES_PATH)).to(device)

        with torch.no_grad():
            forward_dict = net(batch,batch)
        
        for i in range(batch.size(0)):    
            segmentation = forward_dict['left']['segmentation'][i]
            attention_map = shape_attention(forward_dict['left']['attention'][0][i])
            seg = torch.from_numpy(np.asarray(np.argmax(segmentation.cpu(), axis=0), dtype=np.uint8)).long().to(device)
            seg = torch.nn.functional.one_hot(seg, num_classes=19).permute([2,0,1]).float()
            atm = attention_map.squeeze(1)
            if attentions is not None:
                attentions += torch.mul(seg,atm)
            else:
                attentions = torch.mul(seg,atm)
            if total_seg is not None:
                total_seg += seg
            else:
                total_seg = seg
            count += 1
        print(f'{index//BATCH_SIZE}/{len(images)//BATCH_SIZE}')

    result = attentions / count 
    normalized = normalize_attention(result.cpu().numpy())
    for i, attention in enumerate(normalized):
        file_path = f'{OUTPUT_IMAGES_PATH}/{attribute}/{CS_CLASSES[i]}.png'
        cv2.imwrite(file_path, cv2.applyColorMap(attention, cv2.COLORMAP_JET))

    clamped_seg = torch.clamp(total_seg, min=1)
    per_pixel_avg = (attentions/clamped_seg).cpu().numpy()
    for i in range(per_pixel_avg.shape[0]):
        per_pixel_avg[i] = clear_zeros(per_pixel_avg[i]) #set empty attentions to min value and make scale logarithmic
        per_pixel_avg[i] = clear_outliers(per_pixel_avg[i])
        per_pixel_avg[i] = np.log10(per_pixel_avg[i])
        per_pixel_avg[i] = normalize_attention(per_pixel_avg[i])
    per_pixel_avg = per_pixel_avg.astype('uint8')
    for i, attention in enumerate(per_pixel_avg):
        file_path = f'{OUTPUT_CLASS_IMAGES_PATH}/{attribute}/{CS_CLASSES[i]}.png'
        cv2.imwrite(file_path, cv2.applyColorMap(attention, cv2.COLORMAP_JET))
