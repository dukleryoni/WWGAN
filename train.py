from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from wwgan_utils import calculate_fid, calc_wgp, make_log_dir, calc_gradient_penalty, load_from_cfg
import argparse
import time
import json
import pdb

from torch.utils.tensorboard import SummaryWriter

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "/home/yoni/Datasets/img_align_celeba_full"

# Number of workers for dataloader
workers = 4

# Spatial size of training images. All images will be resized to this size via the transformer
image_size = 64 # CelebA

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

def parse_args():
    parser = argparse.ArgumentParser(description='Hyper-parameter tuning')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for D and G for ADAM optimizer')
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from checkpoint path')
    parser.add_argument('--beta1',  default=0.5, type=float,
                        help='beta1 for ADAMs parameters')
    parser.add_argument('--lam',  default=4.0, type=float,
                        help='weight of gradient penalty')
    parser.add_argument('--gam', default=1.0, type=float,
                        help='weight of gradient penalty')
    parser.add_argument('--epoch', default=10,  type=int,
                        help='number of epochs for training ')
    parser.add_argument('--ngpu', default=1,  type=int,
                        help='number of gpus for training')
    parser.add_argument('--batch', default=64,  type=int,
                        help='batch size for training')
    parser.add_argument('--no_wass', help='uses the traditional L2 gradient penalty',
                        action='store_true')
    parser.add_argument('--from_cfg', help='alternatively load args from a config file', type=str)

    return parser.parse_args()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

def train(args):
    print("Starting Training Loop...")
    if args.no_wass:
        print('using traditional L2 penalty (WGAN-GP)')
    else:
        print('Using the WWGAN penalty')
    iters = 0
    min_score = 1.0e6
    errG = torch.cuda.FloatTensor([1], device=device) # since logging occurs before first G-pass
    D_G_z2 = 0
    last_time = time.time()

    model_dir = 'saved_model'
    os.mkdir(args.log_dir + '/' + model_dir)
    # Tensorboard
    writer = SummaryWriter(args.log_dir + '/tf_events')

    for epoch in range(args.epoch):
        # epoch_time.append((time.time() - last_time))
        writer.add_scalar('epoch_time', (time.time() - last_time), iters)
        last_time = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize D(x) -D(G(x)) + LAMBDA* grad-penality(D)
            ###########################
            # Train with all-real batch
            for param in netD.parameters():
                param.requires_grad_(True)

            # Critic updates
            netD.zero_grad()
            # train real batch
            real_gpu = data[0].to(device)
            b_size = real_gpu.size(0)
            output_real = netD(real_gpu).view(-1)
            errD_real = output_real.mean()
            D_x = output_real.mean().item()
            # Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            output_fake = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = output_fake.mean()
            D_G_z1 = errD_fake.item()

            if args.no_wass:
                gradient_penalty = calc_gradient_penalty(netD, real_gpu, fake, args.lam)
            else:
                gradient_penalty, wgrad, L2_grad, sum_grad = calc_wgp(netD, real_gpu, fake, args.lam, args.gam)#, gamma =2, a=0.5, b=0.1)

                writer.add_scalar('Wasserstein gradient norm', wgrad, iters)
                writer.add_scalar('L2 gradient norm', L2_grad, iters)
                writer.add_scalar('Sum gradient norm (normalized)', sum_grad, iters)

            GP = gradient_penalty.item()

            errD = errD_fake - errD_real + gradient_penalty
            errD.backward()
            # Update D
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if iters % critic_iter == critic_iter - 1:
                for param in netD.parameters():
                    param.requires_grad_(False)
                netG.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = -1*output.mean()
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = errG.item()
                # Update G
                optimizerG.step()

            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tGP: %.4f\tD(x): %.4f\tD(G(z)): %.4f , %.4f'
                      % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), -1*errG.item(), GP, D_x, D_G_z1, D_G_z2))

            writer.add_scalar(f'loss/G_losses', errG.item(), iters)
            writer.add_scalar(f'loss/D_losses', errD.item(), iters)
            writer.add_scalar(f'loss/Gradient_penalty_losses', GP, iters)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 1000 == 0) or ((epoch == args.epoch-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                epoch_images = vutils.make_grid(fake, padding=2, normalize=True)
                writer.add_image('fixed_seed_genreated_images', epoch_images, iters)

            iters += 1


        print('calculating FID now')
        score = calculate_fid(netG, nz, data='celebA', batch_size=batch_size)
        writer.add_scalar('FID_score', score, iters)
        if score < min_score:
            min_score = score
            torch.save(netG.state_dict(), args.log_dir+ '/' + model_dir +'/Gen.pth.tar')
        torch.save(netD.state_dict(), args.log_dir + '/' + model_dir + '/Disc.pth.tar') # Todo does this make sense? saving D after every epoch...
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    if args.from_cfg is not None:
        my_config = args.from_cfg
        my_log_dir = make_log_dir(prefix='logs')
        args = load_from_cfg(my_config)
        args.log_dir = my_log_dir
        with open(my_log_dir + '/to_config.json', 'w+') as arg_file:
            json.dump(my_config, arg_file) # save the path to the config not the args.
    else:
        args.log_dir = make_log_dir(prefix='logs')

    beta1 = args.beta1
    batch_size = args.batch
    # Number of critic iterations for WGAN
    critic_iter = 5
    # gradient penality factor term
    LAMBDA = args.lam

    setattr(args, 'critic_iter', critic_iter)
    setattr(args, 'dataloader_workers', workers)
    setattr(args, 'number of channels', nc)
    setattr(args, 'image_size', image_size)
    setattr(args, 'nc', nc)
    setattr(args, 'nz', nz)
    setattr(args, 'ngf', ngf)
    setattr(args, 'ndf', ndf)
    setattr(args, 'ngf', ngf)
    with open(args.log_dir+ '/config.json', 'w+') as arg_file:
        json.dump(vars(args), arg_file)

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers, drop_last=False)

    # Intialize network, optimizer, and set-aside fixed noise for generator evaluation
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    # Create the generator and discriminator
    netG = Generator(args.ngpu).to(device)
    netD = Discriminator(args.ngpu).to(device)

    if args.resume is not None:
        try:
            netG.load_state_dict(torch.load(args.resume + '/Gen.pth.tar'))
            print('Loaded generator from '+ args.resume)
        except:
            print('Failed to load generator from ', args.resume)
        try:
            netD.load_state_dict(torch.load(args.resume + '/Disc.pth.tar'))
            print('loaded discriminator from ' + args.resume)
        except:
            print('Failed to load discriminator ', args.resume)

    else:
        # custom init
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Load models to multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
        netD = nn.DataParallel(netD, list(range(args.ngpu)))

    print(netG, netD)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.9))

    train(args)