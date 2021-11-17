
# coding: utf-8
import pandas as pd
import numpy as np
import os
from skimage import io
import json
from torchvision import transforms
import torchvision.models as models
import torch 
from nets.RankCnn import RankCnn


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#constants
DATA_PATH = 'rank.csv'
ATTRIBUTES = ['wealthy',
    'depressing',
    'safety',
    'lively',
    'boring',
    'beautiful'
]
IMAGE_DIR = 'pp_cropped'
BATCH_SIZE=2
MODEL_DIR = 'models'

MODELS = {
    'wealthy':'rcnn_vgg_wealthy_model_10.pth',
    'depressing':'rcnn_vgg_depressing_model_1.pth',
    'safety':'rcnn_vgg_safety_model_9.pth',
    'lively':'rcnn_vgg_lively_model_3.pth',
    'boring':'rcnn_vgg_boring_model_2.pth',
    'beautiful': 'rcnn_vgg_beautiful_model_2.pth'
}

def get_image_batch(data,start,end,image_dir):
    image_ids = data['id']
    batch = image_ids[start:end]
    return np.array([ io.imread(f'{image_dir}/{x}.jpg') for x in batch])

def forward(net,batch):
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((244,244)),
            transforms.ToTensor()
            ])
    result = torch.stack([ transform(image) for image in batch])
    with torch.no_grad():   
        return net.forward(result)

# def forward(net,batch):
#     return np.random.randint(-3, 15, batch.shape[0])

data = pd.read_csv(DATA_PATH)

for attribute in ATTRIBUTES:
    net = RankCnn(models.vgg19)
    net.to(device)
    net.load_state_dict(torch.load(f'{MODEL_DIR}/{MODELS[attribute]}'))
    net.eval()
    print(f'Starting {attribute}')
    for i in range(0,len(data),BATCH_SIZE):
        start = i
        end = min((i+BATCH_SIZE,len(data)))
        batch = get_image_batch(data,start,end,IMAGE_DIR)
        ranks = forward(net,batch).view(end-start).cpu().numpy()
        data.loc[start:end-1,attribute] = ranks
        print(f'{end}/{len(data)}\r', end="")

meta_data = {}
for attribute in ATTRIBUTES:
    maxi = np.max(data[attribute])
    mini = np.min(data[attribute])
    meta_data[attribute] = {
        'max': float(maxi),
        'min': float(mini)
    }
    data[attribute] = 10*(data[attribute] - mini)/(maxi - mini) - 5

data.to_csv('../rank_full.csv',index=False)
json.dump(meta_data,open('meta.json','w'))

