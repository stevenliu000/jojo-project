import os, time, pickle, argparse, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils
import numpy as np
from torchvision import transforms
from utils import ToRGB, RatioedResize, Zero, RGBToBGR
from conv2d_mtl import Conv2dMtl

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name',  help='')
parser.add_argument('--src_data', required=False, default='src_data_path',  help='sec data path')
parser.add_argument('--tgt_data', required=False, default='tgt_data_path',  help='tgt data path')
parser.add_argument('--vgg_model', required=False, default='pre_trained_VGG19_model_path/vgg19.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--pre_train_epoch', type=int, default=10)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--G_encoder_weight', required=True, default='pre_trained_model', help='pre_trained cartoongan model path')
parser.add_argument('--G_decoder_weight', required=True, default='pre_trained_model', help='pre_trained cartoongan model path')
parser.add_argument('--image_dir', required=True, default='image_dir', help='test image path')
parser.add_argument('--output_image_dir', required=True, default='output_image_dir', help='output test image path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

G_encoder = networks.TransformerEncoder()
G_decoder = networks.TransformerDecoder(conv=Conv2dMtl)
if torch.cuda.is_available():
		G_decoder.load_state_dict(torch.load(args.G_decoder_weight))
		G_encoder.load_state_dict(torch.load(args.G_encoder_weight), strict=False)
else:
    # cpu mode
    G_encoder.load_state_dict(torch.load(args.G_encoder_weight, map_location=lambda storage, loc: storage), strict=False)
    G_decoder.load_state_dict(torch.load(args.G_decoder_weight, map_location=lambda storage, loc: storage))
G_encoder.to(device)
G_decoder.to(device)
G_encoder.eval()
G_decoder.eval()

src_transform = transforms.Compose([
        ToRGB(),
        RatioedResize(args.input_size),
        transforms.ToTensor(),
        RGBToBGR(),
        Zero(),
])

# utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)
image_src = utils.data_load(os.path.join(args.image_dir), 'test', src_transform, 1, shuffle=True, drop_last=True)

with torch.no_grad():
    for n, (x, _) in enumerate(image_src):
        x = x.to(device)
        G_recon = G_encoder(x)[0]
				G_recon = G_decoder(x)[0]
        result = G_recon[0]
        path = os.path.join(args.output_image_dir, str(n + 1) + '.png')
				# BGR -> RGB
        result = result[[2, 1, 0], :, :]
				# deprocess, (0, 1)
        result = result.data.cpu().float() * 0.5 + 0.5
        vutils.save_image(result, path)

'''
valid_ext = ['.jpg', '.png']

for files in os.listdir(args.image_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(args.image_dir, files)).convert("RGB")
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h *1.0 / w
	if ratio > 1:
		h = args.input_size
		w = int(h*1.0/ratio)
	else:
		w = args.input_size
		h = int(w * ratio)
	input_image = input_image.resize((h, w), Image.BICUBIC)
	input_image = np.asarray(input_image)
	# RGB -> BGR
	input_image = input_image[:, :, [2, 1, 0]]
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
	input_image = -1 + 2 * input_image 
	if torch.cuda.is_available():
		input_image = Variable(input_image, volatile=True).cuda()
	else:
		input_image = Variable(input_image, volatile=True).float()
	# forward
	output_image = G(input_image)
	output_image = output_image[0]
	# BGR -> RGB
	output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	output_image = output_image.data.cpu().float() * 0.5 + 0.5
	# save
	vutils.save_image(output_image, os.path.join(args.output_image_dir, files[:-4] + '.jpg'))

print('Done!')
'''

