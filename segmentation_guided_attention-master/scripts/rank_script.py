from torchvision import transforms
import torchvision.models as models
from skimage import io
import torch
import numpy as np

from nets.RankCnn import RankCnn

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
if __name__ == '__main__':
    model_path='models/rcnn_vgg_wealthy_model_10.pth'
    image_path='pp_cropped/50e5f7d4d7c3df413b00056a.jpg'
    net = RankCnn(models.vgg19)
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((244,244)),
            transforms.ToTensor()
            ])
    image = io.imread(image_path)
    image = transform(image)
    with torch.no_grad():
        for _ in range(1000):
            print(net(image.view(1,*image.size())))