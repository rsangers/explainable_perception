# coding: utf-8

## dependencies
import argparse
import os
from comet_ml import Experiment
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import numpy as np

from data import PlacePulseDataset, AdaptTransform
import seg_transforms
import logging
from datetime import date
import os



#script args
def arg_parse():
    parser = argparse.ArgumentParser(description='Training place pulse')
    parser.add_argument('--cuda', help="run with cuda", action='store_true')
    parser.add_argument('--csv', help="path to placepulse csv dirs", default="votes/", type=str)
    parser.add_argument('--dataset', help="dataset images directory path", default="placepulse/", type=str)
    parser.add_argument('--attribute', help="placepulse attribute to train on", default="wealthy", type=str,  choices=['wealthy','lively', 'depressing', 'safety','boring','beautiful','all'])
    parser.add_argument('--batch_size', help="batch size", default=32, type=int)
    parser.add_argument('--n_layers', help="number of attention layers for segrank", default=1, type=int)
    parser.add_argument('--n_heads', help="number of attention heads for segrank", default=1, type=int)
    parser.add_argument('--n_outputs', help="number of outputs for segrank", default=1, type=int)
    parser.add_argument('--softmax', help="use softmax on Pspnet", action='store_true')
    parser.add_argument('--logexp', help="use logexp loss", action='store_true')
    parser.add_argument('--lr', help="learning_rate", default=0.001, type=float)
    parser.add_argument('--resume','--r', help="resume training",action='store_true')
    parser.add_argument('--wd', help="weight decay regularization", default=0.0, type=float)
    parser.add_argument('--num_workers', help="number of workers for data loader", default=4, type=int)
    parser.add_argument('--model_dir', help="directory to load and save models", default='models/', type=str)
    parser.add_argument('--model', help="model to use, sscnn or rsscnn", default='rcnn', type=str, choices=['rsscnn','sscnn','rcnn', 'segrank', 'attentionrcnn', 'sgrb', 'segattn', 'transformer'])
    parser.add_argument('--epoch', help="epoch to load training", default=1, type=int)
    parser.add_argument('--max_epochs', help="maximum training epochs", default=10, type=int)
    parser.add_argument('--cuda_id', help="gpu id", default=0, type=int)
    parser.add_argument('--premodel', help="premodel to use, alex or vgg or dense", default='alex', type=str, choices=['alex','vgg','dense','resnet', 'deit_base', 'deit_small', 'deit_base_distilled'])
    parser.add_argument('--finetune','--ft', help="finetune premodel", action='store_true')
    parser.add_argument('--pbar','--pb', help="add pbars", action='store_true')
    parser.add_argument('--equal','--eq', help="use ties on data", action='store_true')
    parser.add_argument('--comet','--cm', help="use comet", action='store_true')
    parser.add_argument('--tag','--t', help="extra tag for comet and model name", default='', type=str)
    parser.add_argument('--attention_normalize','--at', help="how to normalize attention images for segrank.", default="local", type=str, choices=['local','global'])
    parser.add_argument('--reg', help="use rank reg.", action='store_true')
    parser.add_argument('--alpha', help="rank reg weight", default=0.1, type=float)
    parser.add_argument('--lr_decay', help="use lr_decay", action='store_true')
    parser.add_argument('--sgd', help="use sgd", action='store_true')
    return parser


if __name__ == '__main__':
    parser = arg_parse()
    args = parser.parse_args()
    print(args)
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s',filename=f'logs/{args.attribute}-{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.WARNING) #set the minimum level of message logging

    if args.model == 'transformer':
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        train_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        return_images = True
    elif args.model not in  ["segrank", 'sgrb', 'segattn']:
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

    train=PlacePulseDataset(
        f'{args.csv}/{args.attribute}/train.csv',
        args.dataset,
        transform=train_transforms,
        logger=logger,
        equal=args.equal,
        return_images=return_images
        )
    val=PlacePulseDataset(
        f'{args.csv}/{args.attribute}/val.csv',
        args.dataset,
        transform=val_transforms,
        logger=logger,
        return_images=return_images
        )
    dataloader = DataLoader(train, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

    if args.cuda:
        device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if args.model=="transformer":
        from nets.MyTransformer import MyTransformer as Net
        from train_scripts.SegRank import train
    elif args.model=="sscnn":
        from nets.sscnn import SsCnn as Net
        from train_scripts.sscnn import train
    elif args.model=="rcnn":
        from nets.rcnn import RCnn as Net
        from train_scripts.SegRank  import train
    elif args.model == "segrank":
        from nets.SegRank import SegRank as Net
        from train_scripts.SegRank import train
        import torch.distributed as dist
        dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
    elif args.model =='attentionrcnn':
        from nets.AttentionRCNN import AttentionRCNN as Net
        from train_scripts.SegRank import train
    elif args.model == 'sgrb':
        from nets.SegRankBaseline import SegRank as Net
        from train_scripts.SegRank import train
        import torch.distributed as dist
        dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
    elif args.model == 'segattn':
        from nets.SegAttention import SegAttention as Net
        from train_scripts.SegRank import train
        import torch.distributed as dist
        dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
    else:
        from nets.rsscnn import RSsCnn as Net
        from train_scripts.rsscnn import train

    models = {
        'alex':models.alexnet,
        'vgg':models.vgg19,
        'dense':models.densenet121,
        'resnet':models.resnet50,
        'deit_base':torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True),
        'deit_small': torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True),
        'deit_base_distilled': torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
    }
    if args.model == 'segrank':
        net = Net(
            image_size=(244,244),
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            softmax=args.softmax,
            n_outputs=args.n_outputs
        )
    elif args.model == 'transformer':
        net = Net(
            model=models[args.premodel]
        )

    elif args.model == 'attentionrcnn':
        net = Net(
            models[args.premodel],
            finetune=args.finetune,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    elif args.model == 'segattn':
        net = Net(
            models[args.premodel],
            image_size=(244,244),
            finetune=args.finetune,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            softmax=args.softmax,
        )
    elif args.model == 'sgrb':
        net = Net(image_size=(244,244))
    else:
        net = Net(models[args.premodel], finetune=args.finetune)
    if args.resume:
        net.load_state_dict(torch.load(os.path.join(args.model_dir,'{}_{}_{}_model_{}.pth'.format(
            args.model,
            args.premodel,
            args.attribute,
            args.epoch
        ))))

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="qTgbDpkUcJu1JZbDVUGudcgld",
                            project_name="urban-space-perception",
                            workspace="rsangers")
    tags = [args.premodel, args.attribute, args.model]
    if args.tag: tags.append(args.tag)
    experiment.add_tags(tags)
    experiment.log_parameters(
        {
            "batch_size": args.batch_size,
            "finetune": args.finetune,
            "ties": args.equal,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "attribute": args.attribute,
            "model": args.model,
            "premodel": args.premodel
        }
    )

    train(device,net,dataloader,val_loader, args, logger, experiment)


