# coding: utf-8

## dependencies
import argparse
from comet_ml import Experiment
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import os

from data import PlacePulseDataset, AdaptTransform
import logging
from datetime import date
from train_scripts.train_script import train


# Arguments to pass on to the trainer
def arg_parse():
    parser = argparse.ArgumentParser(description='Training place pulse')
    parser.add_argument('--cuda', help="run with cuda", action='store_true')
    parser.add_argument('--csv', help="path to placepulse csv dirs", default="votes/", type=str)
    parser.add_argument('--dataset', help="dataset images directory path", default="placepulse/", type=str)
    parser.add_argument('--attribute', help="placepulse attribute to train on", default="wealthy", type=str,
                        choices=['wealthy', 'lively', 'depressing', 'safety', 'boring', 'beautiful', 'all'])
    parser.add_argument('--batch_size', help="batch size", default=32, type=int)
    parser.add_argument('--resume', '--r', help="resume training", action='store_true')
    parser.add_argument('--model_dir', help="directory to load and save models", default='models/', type=str)
    parser.add_argument('--model', help="model to use, sscnn or rsscnn", default='rcnn', type=str,
                        choices=['rsscnn', 'sscnn', 'rcnn', 'segrank', 'attentionrcnn', 'sgrb', 'segattn',
                                 'transformer'])
    parser.add_argument('--epoch', help="epoch to load training", default=1, type=int)
    parser.add_argument('--max_epochs', help="maximum training epochs", default=10, type=int)
    parser.add_argument('--cuda_id', help="gpu id", default=0, type=int)
    parser.add_argument('--premodel', help="premodel to use, alex or vgg or dense", default='alex', type=str,
                        choices=['alex', 'vgg', 'dense', 'resnet', 'deit_base', 'deit_small', 'deit_base_distilled'])
    parser.add_argument('--finetune', '--ft', help="finetune premodel", action='store_true')
    parser.add_argument('--comet', '--cm', help="use comet", action='store_true')
    parser.add_argument('--lr_decay', help="use lr_decay", action='store_true')
    return parser


if __name__ == '__main__':
    args = arg_parse().parse_args()
    print(args)

    # Initialize logging
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s', filename=f'logs/{args.attribute}-{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.WARNING)  # set the minimum level of message logging

    # Define image transforms
    if args.model == 'transformer':
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        transforms = transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize(256, interpolation=3)),
            AdaptTransform(transforms.CenterCrop(224)),
            AdaptTransform(transforms.ToTensor()),
            AdaptTransform(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)),
        ])
    else:
        transforms = transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize((244, 244))),
            AdaptTransform(transforms.ToTensor())
        ])

    # Load the train and validation dataset
    train_set = PlacePulseDataset(
        f'{args.csv}/{args.attribute}/train.csv',
        args.dataset,
        transform=transforms,
        logger=logger,
        return_images=True
    )
    val_set = PlacePulseDataset(
        f'{args.csv}/{args.attribute}/val.csv',
        args.dataset,
        transform=transforms,
        logger=logger,
        return_images=True
    )
    dataloader = DataLoader(train_set, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, drop_last=True)

    # Define cpu/gpu device
    if args.cuda:
        device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Import model specific modules
    if args.model == "transformer":
        from nets.MyTransformer import MyTransformer as Net
    else:
        from nets.MyCnn import MyCnn as Net


    # Define models available
    models = {
        'alex': models.alexnet,
        'vgg': models.vgg19,
        'dense': models.densenet121,
        'resnet': models.resnet50,
        'deit_base': torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True),
        'deit_small': torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True),
        'deit_base_distilled': torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224',
                                              pretrained=True)
    }

    # Parse model
    if args.model == 'transformer':
        net = Net(
            model=models[args.premodel]
        )
    else:
        net = Net(models[args.premodel], finetune=args.finetune)

    # Resume training if requested
    if args.resume:
        net.load_state_dict(torch.load(os.path.join(args.model_dir, '{}_{}_{}_model_{}.pth'.format(
            args.model,
            args.premodel,
            args.attribute,
            args.epoch
        ))))

    # Model training feedback is handled by comet, change this based on your own account configuration
    experiment = Experiment(api_key="qTgbDpkUcJu1JZbDVUGudcgld",
                            project_name="urban-space-perception",
                            workspace="rsangers")

    # Add experiment-specific tags
    tags = [args.premodel, args.attribute, args.model]
    experiment.add_tags(tags)
    experiment.log_parameters(
        {
            "batch_size": args.batch_size,
            "finetune": args.finetune,
            "attribute": args.attribute,
            "model": args.model,
            "premodel": args.premodel
        }
    )

    train(device, net, dataloader, val_loader, args, logger, experiment)
