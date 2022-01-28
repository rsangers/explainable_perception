import sys
from torchvision import transforms
import torchvision.models as models
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
import cv2
import torch
import math
import scipy
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import importlib

import pytorch_grad_cam

importlib.reload(pytorch_grad_cam)
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from data import PlacePulseDataset, AdaptTransform


def main(myattribute, premodeltype, modeltype, use_cuda, cuda_id, modelpath, csvpath, datapath,
         descending, save_images, visualisation):
    transform = define_transforms(modeltype)
    data = PlacePulseDataset(csvpath, datapath, transform, myattribute, return_ids=True)
    print("Dataset size: ", len(data))

    if use_cuda:
        device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if modeltype == "cnn" and premodeltype == "resnet":
        import nets.MyCnn as MyCnn

        net = MyCnn.MyCnn(models.resnet50, finetune=True)
    elif modeltype == "transformer" and premodeltype == "deit_small":
        import nets.MyTransformer as MyTransformer

        net = MyTransformer.MyTransformer(
            torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True))
    else:
        print("Model not available yet!")
        return

    net.load_state_dict(torch.load(modelpath))
    net = net.eval().to(device)

    print("Running inference...")
    idToRank, idToImage = infer(data, net, modeltype)

    print("Plotting images...")
    sortedImages, sortedRanks = showImages(idToImage, idToRank, descending)

    print("Generating XAI cam...")
    if not save_images:
        nImages = 16
        plotWidth = math.ceil(math.sqrt(nImages))
        plotHeight = 1 + int((nImages - 1) / plotWidth)

        fig2, ax2 = plt.subplots(plotHeight, plotWidth, sharex=True, sharey=True, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(50):  # Save the top 50 images
        if visualisation.startswith('Attention'):
            camImage = np.float32(showattention(sortedImages[i], net, visualisation))
        else:
            camImage = np.float32(showcam(sortedImages[i], net, modeltype, visualisation))

        if save_images:
            camImage = scipy.ndimage.zoom(camImage, (4, 4, 1), order=1)
            transformed_img = np.float32(sortedImages[i][[2, 1, 0], :, :].permute(1, 2, 0) * 256)
            if modeltype == 'transformer':
                if descending:
                    cv2.imwrite("images_transformer/" + myattribute + "/high_exp/img" + str(i) + ".jpg",
                                camImage[:, :, ::-1])
                    cv2.imwrite("images_transformer/" + myattribute + "/high/img" + str(i) + ".jpg", transformed_img)
                else:
                    cv2.imwrite("images_transformer/" + myattribute + "/low_exp/img" + str(i) + ".jpg",
                                camImage[:, :, ::-1])
                    cv2.imwrite("images_transformer/" + myattribute + "/low/img" + str(i) + ".jpg", transformed_img)
            else:
                if descending:
                    cv2.imwrite("images/" + myattribute + "/high_exp/img" + str(i) + ".jpg", camImage[:, :, ::-1])
                    cv2.imwrite("images/" + myattribute + "/high/img" + str(i) + ".jpg", transformed_img)
                else:
                    cv2.imwrite("images/" + myattribute + "/low_exp/img" + str(i) + ".jpg", camImage[:, :, ::-1])
                    cv2.imwrite("images/" + myattribute + "/low/img" + str(i) + ".jpg", transformed_img)
        elif i < 16:
            xIndex = int(i / plotWidth)
            yIndex = i % plotWidth

            camImage = camImage / 255
            title = str(myattribute) + ": " + str(round(sortedRanks[i].item(), 2))

            ax2[xIndex, yIndex].imshow(camImage)
            ax2[xIndex, yIndex].title.set_text(title)

    if not save_images:
        for i in range(plotWidth * plotHeight - nImages):
            ax2[plotHeight - 1, plotWidth - 1 - i].axis('off')

    plt.draw()
    plt.show()


def define_transforms(modeltype):
    if modeltype == "transformer":
        transform = transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize(256, interpolation=3)),
            AdaptTransform(transforms.CenterCrop(224)),
            AdaptTransform(transforms.ToTensor())
        ])
    else:
        transform = transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize((244, 244))),
            AdaptTransform(transforms.ToTensor())
        ])
    return transform


def infer(data, net, modeltype):
    idToRank = {}
    idToImage = {}

    with torch.no_grad():
        print("Total amount of imagepairs: ", data.__len__())
        start = timer()
        for i in range(data.__len__()):
            sample = data.__getitem__(i)
            input_left, input_right = sample['left_image'], sample['right_image']
            left_id, right_id = sample['left_id'], sample['right_id']

            if modeltype == 'transformer':
                forward_dict = net(input_left.to('cuda').reshape(1, 3, 224, 224),
                                   input_right.to('cuda').reshape(1, 3, 224, 224))
            else:
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
    nImages = 16

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


def showattention(imageTensor, model, visualisation):
    def show_mask_on_image(img, mask):
        img = np.float32(img)

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        return np.uint8(255 * cam)

    rgb_img = imageTensor[[2, 1, 0], :, :].permute(1, 2, 0).numpy()
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).cuda()

    if visualisation == "AttentionRollout":
        grad_rollout = VITAttentionRollout(model, discard_ratio=0.01, head_fusion='max')
        mask = grad_rollout(input_tensor)
    else:
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.01)
        mask = grad_rollout(input_tensor, category_index=0)

    np_img = np.array(rgb_img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    visualization = show_mask_on_image(np_img, mask)

    return torch.Tensor(visualization)


def showcam(imageTensor, net, modeltype, visualisation):
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, width, tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    rgb_img = imageTensor[[2, 1, 0], :, :].permute(1, 2, 0).numpy()
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).cuda()

    model = net

    if modeltype == 'transformer':
        target_layer = model.transformer.blocks[-1].norm1
    else:
        target_layer = model.cnn[7][-1]
        reshape_transform = None

    if visualisation == "AblationCAM":
        cam = AblationCAM(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=reshape_transform)
    elif visualisation == "EigenCAM":
        cam = EigenCAM(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=reshape_transform)
    elif visualisation == "EigenCAM":
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=reshape_transform)
    else:
        cam = ScoreCAM(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=reshape_transform)

    cam.batch_size = 64

    # Uses smoothing with the cost of extra runtime
    grayscale_cam = cam(input_tensor=input_tensor, target_category=0, aug_smooth=True, eigen_smooth=True)
    # grayscale_cam = cam(input_tensor=input_tensor, target_category=0)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return torch.Tensor(visualization)


if __name__ == "__main__":
    print("GPU available: ", torch.cuda.is_available())
    print("Torch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    myattribute = "depressing"  # The attribute that the model has trained on
    modeltype = "transformer"  # The model type, either transformer or cnn
    premodeltype = "deit_small"  # The premodel type, either deit_small or resnet

    use_cuda = True
    cuda_id = 0
    descending = True  # Descending will analyze the highest scoring images, ascending the lowest
    save_images = True  # Save images will save the top 50 images, if false it will only make a plot

    modelpath = "models/rcnn_resnet_wealthy_model_0.6502016129032258.pth"
    csvpath = "votes/votes_clean.csv"
    datapath = "placepulse/"

    visualisation_options = ["AblationCAM", "AttentionRollout", "AttentionGradRollout", "EigenCAM", "GradCam",
                             "ScoreCAM"]
    visualisation = visualisation_options[0]  # Choose a visualisation method from one of the options

    if visualisation.startswith("Attention"):
        sys.path.insert(0, "C:/Users/ruben/Downloads/vit-explain")  # Add path to your vit-explain repository
        from vit_rollout import VITAttentionRollout
        from vit_grad_rollout import VITAttentionGradRollout

    main(myattribute, premodeltype, modeltype, use_cuda, cuda_id, modelpath,
         csvpath, datapath, descending, save_images, visualisation)
