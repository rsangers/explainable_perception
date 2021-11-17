import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader
from skimage import io
import torch.distributed as dist
import os
import cv2
import json
import sys

from data import PlacePulseDataset, AdaptTransform
import seg_transforms

sys.path.append('segmentation')
from segmentation.networks.pspnet import Seg_Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= '../datasets/placepulse'
RESTORE_FROM = '../storage/pspnets/CS_scenes_60000.pth'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 32
NUM_CLASSES = 19
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
        seg_transforms.Resize((340,480)),
        seg_transforms.ToTorchDims()
        ])
    result = torch.stack([ torch.from_numpy(transform(image)) for image in batch])
    return result

images = os.listdir(IMAGES_PATH)
seg_net = Seg_Model(num_classes=NUM_CLASSES)
seg_net.load_state_dict(torch.load(RESTORE_FROM, map_location=device))
seg_net.to(device)
seg_net.eval()
total_seg = None
print('loaded model')
attentions = None
count = 0
for index in range(0,len(images),BATCH_SIZE):
    start = index
    end = min((index + BATCH_SIZE,len(images)))
    batch = transform_batch(get_image_batch(images,start,end,IMAGES_PATH)).to(device)

    with torch.no_grad():
        segmentation_batch = seg_net(batch)[0]
    
    for i in range(batch.size(0)):    
        segmentation = segmentation_batch[i]
        seg = torch.from_numpy(np.asarray(np.argmax(segmentation.cpu(), axis=0), dtype=np.uint8)).long().to(device)
        seg = torch.nn.functional.one_hot(seg, num_classes=19).permute([2,0,1]).float()
        if total_seg is not None:
            total_seg += seg
        else:
            total_seg = seg
        count += 1
    print(f'{index//BATCH_SIZE}/{len(images)//BATCH_SIZE}')

total_seg = total_seg.view(NUM_CLASSES, total_seg.size(1) * total_seg.size(2))
percents = total_seg.sum(dim=1)/count/total_seg.size(1)*100
seg_table = { _cls:float(percents[i].cpu()) for i,_cls in enumerate(CS_CLASSES)}
json.dump(seg_table, open('segmentation_stats_large_img.json','w'))
