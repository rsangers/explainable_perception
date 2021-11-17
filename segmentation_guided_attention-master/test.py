# coding: utf-8
####### FIXME: DEPRECATED FILE
## dependencies
import argparse
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision import models as models

from data import PlacePulseDataset, AdaptTransform
import seg_transforms

#script args
def arg_parse():
    parser = argparse.ArgumentParser(description='Training place pulse')
    parser.add_argument('--cuda', help="1 to run with cuda else 0", default=1, type=bool)
    parser.add_argument('--csv', help="dataset csv path", default="votes_clean.csv", type=str)
    parser.add_argument('--dataset', help="dataset images directory path", default="placepulse/", type=str)
    parser.add_argument('--attribute', help="placepulse attribute to train on", default="wealthy", type=str, choices=['wealthy','lively', 'depressing', 'safety','boring','beautiful'])
    parser.add_argument('--batch_size', help="batch size", default=32, type=int)
    parser.add_argument('--num_workers', help="number of workers for data loader", default=4, type=int)
    parser.add_argument('--model_dir', help="directory to load and save models", default='models/', type=str)
    parser.add_argument('--model', help="model to use, sscnn or rsscnn", default='sscnn', type=str, choices=['rscnn','scnn'])
    parser.add_argument('--epoch', help="epoch to load training", default=1, type=int)
    parser.add_argument('--cuda_id', help="gpu id", default=0, type=int)
    parser.add_argument('--premodel', help="premodel to use, alex or vgg or dense", default='alex', type=str, choices=['alex','vgg','dense'])
    return parser

        
if __name__ == '__main__':    
    parser = arg_parse()
    args = parser.parse_args()
    print(args)

    if args.model not in  ["segrank", 'sgrb', 'segattn']:
        train_transforms = transforms.Compose([
                AdaptTransform(transforms.ToPILImage()),
                AdaptTransform(transforms.Resize((244,244))),
                AdaptTransform(transforms.ToTensor())
                ])

        val_transforms = transforms.Compose([
                AdaptTransform(transforms.ToPILImage()),
                AdaptTransform(transforms.Resize((244,244))),
                AdaptTransform(transforms.ToTensor())
                ])
        return_images = True
    else:
        IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
        train_transforms = transforms.Compose([
                AdaptTransform(seg_transforms.ToArray()),
                AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
                AdaptTransform(seg_transforms.Resize((244,244))),
                AdaptTransform(seg_transforms.ToTorchDims())
                ])

        val_transforms = transforms.Compose([
                AdaptTransform(seg_transforms.ToArray()),
                AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
                AdaptTransform(seg_transforms.Resize((244,244))),
                AdaptTransform(seg_transforms.ToTorchDims())
                ])
        return_images = True
    
    data=PlacePulseDataset(args.csv,args.dataset,transforms.Compose([Rescale((224,224)),ToTensor()]), args.attribute)
    len_data = len(data)
    train_len = int(len_data*0.65)
    val_len = int(len_data*0.05)
    test_len = len_data-train_len-val_len
    train,val,test = random_split(data,[train_len , val_len, test_len])
    print(len(test))
    dataloader = DataLoader(test, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    if args.cuda:
        device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if args.model=="sscnn":
        from sscnn import SsCnn as Net
        from sscnn import test
    elif args.model=="rcnn":
        from rcnn import RCnn as Net
        from rcnn import test
    else:
        from rsscnn import RSsCnn as Net
        from rsscnn import test
    
    resnet18 = models.resnet101
    alexnet = models.alexnet
    vgg16 = models.vgg19
    dense = models.densenet161

    models = {
        'alex':models.alexnet,
        'vgg':models.vgg19,
        'dense':models.densenet161,
        'resnet':models.resnet50
    }

    net = Net(models[args.premodel])
    net.load_state_dict(torch.load(os.path.join(args.model_dir,'{}_{}_{}_model_{}.pth'.format(
            args.model,
            args.premodel,
            args.attribute,
            args.epoch
        ))))
    
    test(device,net,dataloader, args)


