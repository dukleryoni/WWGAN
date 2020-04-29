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
from wwgan_utils import get_time_now, calculate_fid, calc_gradient_penalty

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "../data"

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# number of FID score evaluations

# Number of training  between FID score evaluations
num_epochs = 100

# Learning rate for optimizers
lr = 1e-4

# Beta1 hyperparam for Adam optimizers
beta1 = 0.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of critic iterations for WGAN
critic_iter = 5

# gradient penality factor term
LAMBDA = 10.0

run_time = get_time_now()
log_dir = run_time + '_log'
os.mkdir(log_dir)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
my_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = dset.CIFAR10(root=dataroot, train=True,
                       download=False, transform=my_transform)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, drop_last=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


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
        preprocess = nn.Sequential(
            # nn.ConvTranspose2d(nz, ngf * 8 * 4 * 4, 4, 1, 0),
            nn.Linear(nz, 4 * 4 * 8 * ngf),
            nn.BatchNorm1d(4 * 4 * 8 * ngf),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 2, stride=2),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(2 * ngf, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        input = input.view(batch_size, -1)
        output = self.preprocess(input)
        output = output.view(batch_size, 8 * ngf, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(batch_size, nc, image_size, image_size)


######################################################################
# Now, we can instantiate the generator and apply the ``weights_init``
# function. Check out the printed model to see how the generator object is
# structured.
#

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netG.apply(weights_init)

# Print the model
print(netG)


# Discriminator Code no batch norm due to gradient penalty

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, 2 * ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * ndf, 4 * ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * ndf, 8 * ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4 * 4 * 8 * ndf, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 8 * ndf)
        output = self.linear(output)
        return output


######################################################################
# Now, as with the generator, we can create the discriminator, apply the
# ``weights_init`` function, and print the model’s structure.
#

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netD.apply(weights_init)

# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
one = torch.cuda.FloatTensor(1, device=device)
mone = one * -1

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

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
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = output.mean()
        # Calculate gradients for D in backward pass
        errD_real.backward(mone)
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = output.mean()
        # Calculate the gradients for this batch
        errD_fake.backward(one)
        D_G_z1 = errD_fake.item()
        # Add the gradients from the all-real and all-fake batches

        gradient_penalty = calc_gradient_penalty(netD, real_cpu, fake, b_size, nc, image_size, LAMBDA)
        gradient_penalty.backward(one)
        GP = gradient_penalty.item()

        errD = errD_fake - errD_real + gradient_penalty
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
            errG = output.mean()
            # Calculate gradients for G
            errG.backward(mone)
            D_G_z2 = errG.item()
            # Update G
            optimizerG.step()

        # Output training stats
        if i % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tGP: %.4f\tD(x): %.4f\tD(G(z)): %.4f , %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), -1 * errG.item(), GP, D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        GP_losses.append(GP)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 1000 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    print('calculating FID now')
    score = calculate_fid(netG, nz, data='cifar10')
    FID_scores.append(score)
    if score < min_score:
        min_score = score
        torch.save(netG.state_dict(), log_dir + '/early_stop.pth.tar')

# Save results
save_data = {'img_list': img_list, 'G_losses': G_losses, 'D_losses': D_losses, 'GP_losses': GP_losses,
             'FID_scores': FID_scores}
pkl.dump(save_data, open(log_dir + '/WGAN_data' + run_time + 'trial' + '.p', 'wb'))
