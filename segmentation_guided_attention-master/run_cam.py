import sys
import os
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import pandas as pd
import torch.nn as nn
import torchvision

import numpy as np
from data import PlacePulseDataset, AdaptTransform
import seg_transforms

from timeit import default_timer as timer
from utils.ranking import compute_ranking_loss, compute_ranking_accuracy

import matplotlib.pyplot as plt
from PIL import Image

import importlib
import pytorch_grad_cam
importlib.reload(pytorch_grad_cam)

from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
import math

def main(myattribute, premodeltype, modeltype, batch_size, num_workers, use_cuda, cuda_id, modelpath, csvpath, datapath, nImages, descending):
    train_transforms, val_transforms = define_transforms(modeltype)

    data = PlacePulseDataset(csvpath, datapath, val_transforms, myattribute, return_ids=True)
    print("Dataset size: ", len(data))

    if use_cuda:
        device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if modeltype == "rcnn" and premodeltype == "resnet":
        import nets.rcnn as rcnn

        net = rcnn.RCnn(models.resnet50, finetune=True)
        # net.rank_fc_1 = nn.Linear(8192, 4096)
    else:
        print("Model not available yet!")

    net.load_state_dict(torch.load(modelpath))
    net = net.eval().to(device)

    print("Running inference...")
    idToRank, idToImage = infer(data, net, nImages, device)

    print("Plotting images...")
    sortedImages, sortedRanks = showImages(idToImage, idToRank, descending)

    # net.cnn[8]=nn.AdaptiveAvgPool2d(output_size=(1, 1))

    print("Generating XAI cam...")
    nImages = 16

    plotWidth = math.ceil(math.sqrt(nImages))
    plotHeight = 1 + int((nImages - 1) / plotWidth)

    fig2, ax2 = plt.subplots(plotHeight, plotWidth, sharex=True, sharey=True, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(nImages):
        xIndex = int(i / plotWidth)
        yIndex = i % plotWidth

        camImage = np.float32(showcam(sortedImages[i], net)) / 255
        title = str(myattribute) + ": " + str(round(sortedRanks[i].item(), 2))

        ax2[xIndex, yIndex].imshow(camImage)
        ax2[xIndex, yIndex].title.set_text(title)

    for i in range(plotWidth * plotHeight - nImages):
        ax2[plotHeight - 1, plotWidth - 1 - i].axis('off')

    plt.draw()
    plt.show()



def define_transforms(modeltype):
    if modeltype not in ["segrank", 'sgrb', 'segattn']:
        train_transforms = transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize((244, 244))),
            AdaptTransform(transforms.ToTensor())
        ])
        val_transforms = transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize((244, 244))),
            AdaptTransform(transforms.ToTensor())
        ])
        return_images = True
    else:
        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        train_transforms = transforms.Compose([
            AdaptTransform(seg_transforms.ToArray()),
            AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
            AdaptTransform(seg_transforms.Resize((244, 244))),
            AdaptTransform(seg_transforms.ToTorchDims())
        ])
        val_transforms = transforms.Compose([
            AdaptTransform(seg_transforms.ToArray()),
            AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
            AdaptTransform(seg_transforms.Resize((244, 244))),
            AdaptTransform(seg_transforms.ToTorchDims())
        ])
        return_images = True
    return train_transforms, val_transforms



def infer(data, net, nImages, device):
    rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    idToRank = {}
    idToImage = {}

    with torch.no_grad():
        print("Total amount of imagepairs: ", data.__len__())
        start = timer()
        for i in range(nImages):
            sample = data.__getitem__(i)
            input_left, input_right, label, attribute = sample['left_image'], sample['right_image'], sample['winner'], \
                                                        sample['attribute']
            left_id, right_id = sample['left_id'], sample['right_id']

            label, attribute = torch.Tensor([label]).to(device).float(), torch.Tensor([attribute]).to(device)

            forward_dict = net(input_left.to('cuda').reshape(1, 3, 244, 244),
                               input_right.to('cuda').reshape(1, 3, 244, 244))
            output_rank_left, output_rank_right = forward_dict['left']['output'], forward_dict['right']['output']

            idToRank[left_id] = output_rank_left
            idToRank[right_id] = output_rank_right

            idToImage[left_id] = input_left
            idToImage[right_id] = input_right

            if i % 100 == 0:
                print("Currently at imagepair: ", i)
        end = timer()

    print("Total runtime: ", str(end - start))
    return idToRank, idToImage

def showImages(idToImage, idToRank, descending):
    rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    nImages = 16

    # imageDict = dict(zip(images, ranks))
    # imageDict = {k: v for k, v in sorted(imageDict.items(), key=lambda item: item[1], reverse=descending)}
    sortedIdToRank = {k: v for k, v in sorted(idToRank.items(), key=lambda item: item[1], reverse=descending)}
    sortedRanks = list(sortedIdToRank.values())

    sortedImages = [idToImage[key] for key in sortedIdToRank.keys()]

    with torch.no_grad():
        if nImages > 3:
            plotWidth = 4
            plotHeight = 1 + int((nImages - 1) / plotWidth)
        else:
            plotWidth = nImages
            plotHeight = 1
        fig1, ax1 = plt.subplots(plotHeight, plotWidth, sharex=True, sharey=True, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        for i in range(nImages):
            xIndex = int(i / plotWidth)
            yIndex = i % plotWidth
            ax1[xIndex, yIndex].imshow(sortedImages[i].permute(1, 2, 0))
            title = str(myattribute) + ": " + str(round(sortedRanks[i].item(), 2))
            ax1[xIndex, yIndex].title.set_text(title)

    plt.draw()
    return sortedImages, sortedRanks

def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img.copy()).unsqueeze(0)

def showcam(imageTensor, net):
    rgb_img = imageTensor.permute(1, 2, 0).numpy()
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).cuda()
    # model = net.cnn
    # target_layer = model[7][-1]
    model = net
    target_layer = model.cnn[7][-1]

    # target_layer = model[8]

    # print(net.forward(input_tensor))

    cam = EigenCAM(model=model, target_layer=target_layer, use_cuda=True)
    #cam = AblationCAM(model=model, target_layer=target_layer, use_cuda=True)
    cam.batch_size = 2

    # grayscale_cam = cam(input_tensor=input_tensor, target_category=0, aug_smooth=True, eigen_smooth=True)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=0)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # mask = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLOR_BGR2RGB)

    return torch.Tensor(visualization)

if __name__ == "__main__":
    print("GPU available: ", torch.cuda.is_available())
    print("Torch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    myattribute = "safety"  # don't forget to change modelpath as well!
    premodeltype = "resnet"
    modeltype = "rcnn"
    batch_size = 32  # chose this according to resources
    num_workers = 4
    use_cuda = True
    cuda_id = 0
    nImages = 100
    descending = True

    # modelpath = "models/rcnn_resnet_depressing_model_0.6417545180722891.pth"
    # modelpath = "models/rcnn_resnet_wealthy_model_0.6502016129032258.pth"
    modelpath = 'models/rcnn_resnet_safety_model_0.632892382413088.pth'
    csvpath = "votes_clean.csv"
    datapath = "placepulse/"

    main(myattribute, premodeltype, modeltype, batch_size, num_workers, use_cuda, cuda_id, modelpath, \
         csvpath, datapath, nImages, descending)
