import sys, os

from PIL import Image
import torch
import torchvision

from ada_in_style import net
from ada_in_style.downloader import download_weights
from ada_in_style.function import adaptive_instance_normalization


PATH = os.path.dirname(__file__)


class AdaIN(object):

    def __init__(self, content_size=512, style_size=512, alpha=1, device='cpu'):

        self.alpha = alpha
        self.device = torch.device(device)

        self.decoder = net.decoder
        self.vgg = net.vgg

        self.decoder.eval()
        self.vgg.eval()

        self.load_weights(self.vgg,
            os.path.join(PATH, '../resources/vgg_normalised.pth'),
            '1srjH56JDrsKsZruoLE_wCfHzbNHsiuTX')
        self.load_weights(self.decoder,
            os.path.join(PATH, '../resources/decoder.pth'),
            '13G7Sw7KOoPUXv76pY0fCXyoLGKDSikhs')
        self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

        self.content_tf = self.test_transform(size=content_size, crop=False)
        self.style_tf = self.test_transform(size=style_size, crop=False)

    def load_weights(self, model, path, file_id):
        if not os.path.isfile(path):
            download_weights(file_id, path)
        weights = torch.load(path)
        model.load_state_dict(weights)

    def test_transform(self, size, crop):
        transform_list = []
        if size != 0:
            transform_list.append(torchvision.transforms.Resize(size))
        if crop:
            transform_list.append(torchvision.transforms.CenterCrop(size))
        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)
        return transform

    def __call__(self, content_image, style_image, output_path='im.png'):
        content = self.content_tf(Image.open(content_image))
        style = self.style_tf(Image.open(style_image))
        
        style = style.to(self.device).unsqueeze(0)
        content = content.to(self.device).unsqueeze(0)

        with torch.no_grad():
            content_f = self.vgg(content)
            style_f = self.vgg(style)
            feat = adaptive_instance_normalization(content_f, style_f)
            feat = feat * self.alpha + content_f * (1 - self.alpha)
            output = self.decoder(feat)

        im = output.cpu()
        torchvision.utils.save_image(im, output_path)