import os, time, pickle, argparse, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from edge_promoting import edge_promoting
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
from utils import ToRGB, RatioedResize, Zero, RGBToBGR

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name', help='')
parser.add_argument('--src_data', required=False, default='src_data_path', help='sec data path')
parser.add_argument('--tgt_data', required=False, default='tgt_data_path', help='tgt data path')
parser.add_argument('--vgg_model', required=False, default='pre_trained_VGG19_model_path/vgg19.pth',
                    help='pre-trained VGG19 model path')
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
parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--G_pre_trained_weight', required=False, default='', help='pre_trained_weight for G')
parser.add_argument('--save_period', type=int, required=False, default=2, help='of how many epochs it saves model')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

# results save path
if not os.path.isdir(os.path.join(args.name + '_results', 'Reconstruction')):
    os.makedirs(os.path.join(args.name + '_results', 'Reconstruction'))
if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
    os.makedirs(os.path.join(args.name + '_results', 'Transfer'))

# edge-promoting
if not os.path.isdir(os.path.join('data', args.tgt_data, 'pair')):
    print('edge-promoting start!!')
    edge_promoting(os.path.join('data', args.tgt_data, 'train'), os.path.join('data', args.tgt_data, 'pair'))
else:
    print('edge-promoting already done')

# data_loader
src_transform = transforms.Compose([
    ToRGB(),
    transforms.Resize((args.input_size, args.input_size)),
    transforms.ToTensor(),
    RGBToBGR(),
    Zero(),
])

tgt_transform = transforms.Compose([
    ToRGB(),
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    RGBToBGR(),
    Zero(),
])

src_transform_test = transforms.Compose([
    ToRGB(),
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    RGBToBGR(),
    Zero(),
])

train_loader_src = utils.data_load(os.path.join('data', args.src_data), 'train', src_transform, args.batch_size,
                                   shuffle=True, drop_last=True)
train_loader_tgt = utils.data_load(os.path.join('data', args.tgt_data), 'pair', tgt_transform, args.batch_size,
                                   shuffle=True, drop_last=True)
train_loader_tgt1 = utils.data_load(os.path.join('data', args.tgt_data), 'peper', tgt_transform, args.batch_size,
                                    shuffle=True, drop_last=True)
train_loader_tgt2 = utils.data_load(os.path.join('data', args.tgt_data), 'sprit', tgt_transform, args.batch_size,
                                    shuffle=True, drop_last=True)
train_loader_tgt3 = utils.data_load(os.path.join('data', args.tgt_data), 'name', tgt_transform, args.batch_size,
                                    shuffle=True, drop_last=True)

test_loader_src_large = utils.data_load(os.path.join('data', args.src_data), 'test', src_transform_test, 1,
                                        shuffle=True, drop_last=True)
train_loader_src_large = utils.data_load(os.path.join('data', args.src_data), 'train', src_transform_test, 1,
                                         shuffle=True, drop_last=True)

# network
# G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
G_e = networks.TransformerEncoder()
if args.latest_generator_model != '':
    if torch.cuda.is_available():
        G_e.load_state_dict(torch.load(args.latest_generator_model, strict=False))
    else:
        # cpu mode
        G_e.load_state_dict(
            torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage, strict=False))

if args.G_pre_trained_weight != '':
    print("loaded G_e weight!")
    if torch.cuda.is_available():
        G_e.load_state_dict(torch.load(args.G_pre_trained_weight, strict=False))
    else:
        # cpu mode
        G_e.load_state_dict(
            torch.load(args.G_pre_trained_weight, map_location=lambda storage, loc: storage, strict=False))
d = networks.TransformerDecoder()
D = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)

d1 = networks.TransformerDecoder()
D1 = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)

d2 = networks.TransformerDecoder()
D2 = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)

d3 = networks.TransformerDecoder()
D3 = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)

if args.latest_discriminator_model != '':
    if torch.cuda.is_available():
        d.load_state_dict(torch.load(args.latest_generator_model, strict=False))
        D.load_state_dict(torch.load(args.latest_discriminator_model))

        d1.load_state_dict(torch.load(args.latest_generator_model, strict=False))
        D1.load_state_dict(torch.load(args.latest_discriminator_model))

        d2.load_state_dict(torch.load(args.latest_generator_model, strict=False))
        D2.load_state_dict(torch.load(args.latest_discriminator_model))

        d3.load_state_dict(torch.load(args.latest_generator_model, strict=False))
        D3.load_state_dict(torch.load(args.latest_discriminator_model))
    else:
        d.load_state_dict(
            torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage, strict=False))
        D.load_state_dict(
            torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage, strict=False))

        d1.load_state_dict(
            torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage, strict=False))
        D1.load_state_dict(
            torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage, strict=False))

        d2.load_state_dict(
            torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage, strict=False))
        D2.load_state_dict(
            torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage, strict=False))

        d3.load_state_dict(
            torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage, strict=False))
        D3.load_state_dict(
            torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage, strict=False))

VGG = networks.VGG19(init_weights=args.vgg_model, feature_mode=True)
G_e.to(device)

D.to(device)
D1.to(device)
D2.to(device)
D3.to(device)

d.to(device)
d1.to(device)
d2.to(device)
d3.to(device)

VGG.to(device)
G_e.train()
D.train()
D1.train()
D2.train()
D3.train()

d.train()
d1.train()
d2.train()
d3.train()

VGG.eval()
print('---------- Networks initialized -------------')
utils.print_network(G_e)
utils.print_network(d)
utils.print_network(D)
utils.print_network(VGG)
print('-----------------------------------------------')

# loss
BCE_loss = nn.BCELoss().to(device)
BCE_loss1 = nn.BCELoss().to(device)
BCE_loss2 = nn.BCELoss().to(device)
BCE_loss3 = nn.BCELoss().to(device)

L1_loss = nn.L1Loss().to(device)

# Adam optimizer
d_optimizer = optim.Adam(d.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
d1_optimizer = optim.Adam(d1.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D1_optimizer = optim.Adam(D1.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
d2_optimizer = optim.Adam(d2.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D2_optimizer = optim.Adam(D2.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
d3_optimizer = optim.Adam(d2.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D3_optimizer = optim.Adam(D2.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer,
                                             milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer,
                                             milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
G1_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer,
                                              milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D1_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer,
                                              milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
G2_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer,
                                              milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D2_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer,
                                              milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
G3_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer,
                                              milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D3_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer,
                                              milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

pre_train_hist = {}
pre_train_hist['Recon_loss'] = []
pre_train_hist['per_epoch_time'] = []
pre_train_hist['total_time'] = []

'''
""" Pre-train reconstruction """
if args.latest_generator_model == '':
    print('Pre-training start!')
    start_time = time.time()
    for epoch in range(args.pre_train_epoch):
        epoch_start_time = time.time()
        Recon_losses = []
        for __, (x, _) in tqdm(enumerate(train_loader_src), total=len(train_loader_src)):
            x = x.to(device)

            # train generator G
            G_optimizer.zero_grad()

            x_feature = VGG(x)
            G_ = G(x)
            G_feature = VGG(G_)

            Recon_loss = 10 * L1_loss(G_feature, x_feature.detach())
            Recon_losses.append(Recon_loss.item())
            pre_train_hist['Recon_loss'].append(Recon_loss.item())

            Recon_loss.backward()
            G_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        pre_train_hist['per_epoch_time'].append(per_epoch_time)
        print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), args.pre_train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

    total_time = time.time() - start_time
    pre_train_hist['total_time'].append(total_time)
    with open(os.path.join(args.name + '_results',  'pre_train_hist.pkl'), 'wb') as f:
        pickle.dump(pre_train_hist, f)

    with torch.no_grad():
        G.eval()
        for n, (x, _) in tqdm(enumerate(train_loader_src), total=len(train_loader_src)):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_train_recon_' + str(n + 1) + '.png')
            result = result[[2, 1, 0], :, :]
            result = result.data.cpu().float() * 0.5 + 0.5
            vutils.save_image(result, path)
            if n == 4:
                break

        for n, (x, _) in tqdm(enumerate(test_loader_src), total=len(test_loader_src)):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_test_recon_' + str(n + 1) + '.png')
            result = result[[2, 1, 0], :, :]
            result = result.data.cpu().float() * 0.5 + 0.5
            vutils.save_image(result, path)
            if n == 4:
                break
else:
    print('Load the latest generator model, no need to pre-train')
'''

train_hist = {}
train_hist['Disc_loss'] = []
train_hist['Gen_loss'] = []
train_hist['Con_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
print('training start!')
start_time = time.time()

real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
fake1 = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
fake2 = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
fake3 = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)

for epoch in range(args.train_epoch):
    epoch_start_time = time.time()
    G_e.eval()
    D_scheduler.step()
    D1_scheduler.step()
    D2_scheduler.step()
    D3_scheduler.step()

    Disc_losses = []
    Gen_losses = []
    Con_losses = []
    for (x, _), (y, _), (y1, _), (y2, _), (y3, _) in tzip(train_loader_src, train_loader_tgt, train_loader_tgt1,
                                                          train_loader_tgt2, train_loader_tgt3):
        e = y[:, :, :, args.input_size:]
        y = y[:, :, :, :args.input_size]
        x, y, e, y1, y2, y3 = x.to(device), y.to(device), e.to(device), y1.to(device), y2.to(device), y3.to(device)

        # train D
        D_optimizer.zero_grad()
        D1_optimizer.zero_grad()
        D2_optimizer.zero_grad()
        D3_optimizer.zero_grad()

        D_real = D(y)
        D_real_loss = BCE_loss(D_real, real)

        G_ = G_e(x)[0]
        G_d = d(G_)[0]
        G_d1 = d(G_)[0]
        G_d2 = d(G_)[0]
        G_d3 = d(G_)[0]

        D_fake = D(G_d)
        D_fake_loss = BCE_loss(D_fake, fake)
        D_fake1 = D(G_d1)
        D_fake_loss1 = BCE_loss(D_fake1, fake1)
        D_fake2 = D(G_d2)
        D_fake_loss2 = BCE_loss(D_fake2, fake2)
        D_fake3 = D(G_d3)
        D_fake_loss3 = BCE_loss(D_fake3, fake3)
        D_fake_loss_all = D_fake_loss + D_fake_loss1 + D_fake_loss2 + D_fake_loss3

        D_edge = D(e)
        D_edge1 = D(e)
        D_edge2 = D(e)
        D_edge3 = D(e)

        D_edge_loss = BCE_loss(D_edge, fake)
        D_edge_loss1 = BCE_loss(D_edge1, fake1)
        D_edge_loss2 = BCE_loss(D_edge2, fake2)
        D_edge_loss3 = BCE_loss(D_edge3, fake3)
        D_edge_loss_all = D_edge_loss + D_edge_loss1 + D_edge_loss2 + D_edge_loss3

        Disc_loss = D_real_loss + D_fake_loss_all + D_edge_loss_all
        Disc_losses.append(Disc_loss.item())
        train_hist['Disc_loss'].append(Disc_loss.item())

        Disc_loss.backward()
        D_optimizer.step()

        # train G
        G_ = G_e(x)[0]
        G_d = d(G_)[0]
        G_d1 = d(G_)[0]
        G_d2 = d(G_)[0]
        G_d3 = d(G_)[0]

        D_fake = D(G_d)
        D_fake_loss = BCE_loss(D_fake, real)
        D_fake1 = D(G_d1)
        D_fake_loss1 = BCE_loss(D_fake1, real)
        D_fake2 = D(G_d2)
        D_fake_loss2 = BCE_loss(D_fake2, real)
        D_fake3 = D(G_d3)
        D_fake_loss3 = BCE_loss(D_fake3, real)
        D_fake_loss_all = D_fake_loss + D_fake_loss1 + D_fake_loss2 + D_fake_loss3

        x_feature = VGG(x)
        G_feature = VGG(G_)
        Con_loss = args.con_lambda * L1_loss(G_feature, x_feature.detach())

        Gen_loss = D_fake_loss + Con_loss
        Gen_losses.append(D_fake_loss.item())
        train_hist['Gen_loss'].append(D_fake_loss.item())
        Con_losses.append(Con_loss.item())
        train_hist['Con_loss'].append(Con_loss.item())

        Gen_loss.backward()
        G_optimizer.step()

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(
        '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % (
        (epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
        torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))

    if epoch > 0 and (epoch % args.save_period == 0 or epoch == args.train_epoch - 1):
        with torch.no_grad():
            G.eval()
            for n, (x, _) in enumerate(train_loader_src_large):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer',
                                    str(epoch + 1) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
                result = result[[2, 1, 0], :, :]
                result = result.data.cpu().float() * 0.5 + 0.5
                vutils.save_image(result, path)
                if n == 4:
                    break

            for n, (x, _) in enumerate(test_loader_src_large):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer',
                                    str(epoch + 1) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
                result = result[[2, 1, 0], :, :]
                result = result.data.cpu().float() * 0.5 + 0.5
                vutils.save_image(result, path)
                if n == 4:
                    break

            torch.save(G.state_dict(), os.path.join(args.name + '_results', 'generator_latest.pkl'))
            torch.save(D.state_dict(), os.path.join(args.name + '_results', 'discriminator_latest.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training results")

torch.save(G.state_dict(), os.path.join(args.name + '_results', 'generator_param.pkl'))
torch.save(D.state_dict(), os.path.join(args.name + '_results', 'discriminator_param.pkl'))
with open(os.path.join(args.name + '_results', 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)
