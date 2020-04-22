from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import pickle as pkl
from wwgan_utils import get_time_now, calculate_fid, calc_gradient_penalty, calc_wgp
import subprocess
import argparse

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "../img_align_celeba_full"

# Number of workers for dataloader
workers = 4

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Batch size during training
batch_size = 64

# Number of critic iterations for WGAN
critic_iter = 5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# gradient penality factor term
LAMBDA = 10.0

parser = argparse.ArgumentParser(description='Hyper-parameter tuning')
parser.add_argument('--lr', default = 1e-4, type=float,
                    help='learning rate for D and G for ADAM optimizer')
parser.add_argument('--beta1',  default=0.0, type=float,
                    help='beta1 for ADAMs parameters')
parser.add_argument('--epoch', default=1,  type=int,
                    help='number of epochs for training ')

# We can use an image folder dataset the way we have it setup.
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
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

run_time = get_time_now()
log_dir = 'htune'+run_time + '_log'
os.mkdir(log_dir)


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
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
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



def run(args, save_images):

    # number of epochs
    num_epochs = args.epoch

    # Learning rate for optimizers
    lr = args.lr

    # Beta1 hyperparam for Adam optimizers
    beta1 = args.beta1

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Discriminator Code no batch norm due to gradient penalty
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    #criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    one = torch.cuda.FloatTensor(1, device=device)
    mone = one * -1


    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9))


    ######################################################################
    # Training


    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    GP_losses = []
    grad_list = []
    wgrad_list = []
    FID_scores = []
    iters = 0
    min_score = 1.0e6

    print("Starting Training Loop...")
    # For each epoch
    errD = None
    errG = torch.cuda.FloatTensor([1], device=device)
    D_x = None
    D_G_z1 = None
    D_G_z2 = 0
    GP = 0
    grad = 0
    wgrad = 0
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize D(x) -D(G(x)) + LAMBDA* grad-penality(D)
            ###########################
            # Train with all-real batch
            for param in netD.parameters():
                param.requires_grad_(True)

            # Critic updates
            # for _ in range(critic_iter):
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            # Forward pass real batch through D
            output_real = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = output_real.mean()
            # Calculate gradients for D in backward pass
            #errD_real.backward(mone)
            D_x = output_real.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            output_fake = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = output_fake.mean()
            # Calculate the gradients for this batch
            #errD_fake.backward(one)
            D_G_z1 = errD_fake.item()
            # Add the gradients from the all-real and all-fake batches

            #gradient_penalty = calc_gradient_penalty(netD, real_cpu, fake, b_size, nc, image_size, LAMBDA)
            gradient_penalty, wgrad, grad  = calc_wgp(netD, real_cpu, fake, b_size, nc, image_size, LAMBDA)
            wgrad_list.append(wgrad.item())
            grad_list.append(grad.item())
            #gradient_penalty.backward(one)
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
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), -1*errG.item(), GP, D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            GP_losses.append(GP)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                print('lr={}, beta={}'.format(args.lr, args.beta1))
                if save_images:
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                print('saving temporary data')
                temp_data = {'G_losses': G_losses, 'D_losses':D_losses, 'GP_losses' : GP_losses, 'FID_scores':\
                    FID_scores, 'grad_list': grad_list, 'wgrad_list': wgrad_list}
                pkl.dump(temp_data, open(log_dir+ '/temp_data'+str(lr)+str(beta1)+'.p', 'wb'))

            iters += 1

        print('calculating FID now')
        score = calculate_fid(netG, nz, data='celebA', batch_size=batch_size)
        FID_scores.append(score)
        if score < min_score:
            min_score = score
            torch.save(netG.state_dict(), log_dir+ '/early_stop' +str(lr)+str(beta1)+'.pth.tar')
    return min_score


args = parser.parse_args()
learning_rates = [0.5e-4, 2e-4, 8e-4]
betas = [0, 0.1, 0.5]
min_fids = []
for lr in learning_rates:
    for b in betas:
        args = parser.parse_args()
        args.lr = lr
        args.beta1 = b
        args.epoch = 40
        min_fids.append(run(args, save_images = False))
# to shut down gcloud and save some $cash$ for overnight runs
subprocess.call("sudo poweroff", shell=True)


