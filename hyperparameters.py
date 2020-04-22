import torch
import datetime

# ================ #
# Helper functions #
# ================ #

# <editor-fold desc="Helper functions">
# Helper functions {{{1

# Helper function to get time for logging purposes
def get_time_now():
    date_string = str(datetime.datetime.now())
    day = date_string[0:10]
    hour = date_string[11:13]
    minute = date_string[14:16]
    second = date_string[17:19]
    return day + '--' + hour + '-' + minute + '-' + second

# For saving the models
def save_model(dict_of_stuff,  filename='checkpoint.pth.tar'):
    torch.save(dict_of_stuff, filename)

# ====================== #
# Weights initialization #
# ====================== #
# Custom weights initialization called on the generator
# and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

DO_CUSTOM_WEIGHTS_INIT_DISCRIMINATOR = False
DO_CUSTOM_WEIGHTS_INIT_GENERATOR = False

# }}}1
# </editor-fold>

# ============================================================= #
#                                                               #
#                        Hyperparameters                        #
#                                                               #
# ============================================================= #

# <editor-fold desc="Hyperparameters">
# Hyperparameters {{{1

# +++ Experiment Settings +++ #
DATASET_NAME = 'CIFAR10'  # possible options: 'CIFAR10'
GAN_TYPE='WGANGP'  # possible options: 'WGANGP', 'Vanilla', 'DRAGAN'
ARCHG = 'DCGAN'  # possible options: 'DCGAN', 'MLP'
ARCHD = 'DCGAN'  # possible options: 'DCGAN', 'MLP'
USE_WASSERSTEIN_GRADIENT = False
H = 0.1
G_ITERS = 5
D_ITERS = 5
NUM_RUNS = 5  # Number of times to run the experiment for averaging
MAX_OUTER_ITER = 999999999 # 999,999,999
CONTINUE_FROM_SAVE = False

# +++ Specific settings based off gan type +++ #
LAM = 20  # For WGANGP and DRAGAN.
CC = 0.5  # For DRAGAN

# +++ Some architecture settings +++ #
DIM_WGANGP = 64  # For WGANGP with DCGAN
NGF = 64  # For Vanilla/DRAGAN with DCGAN, or MLP
NDF = 64  # For Vanilla/DRAGAN with DCGAN, or MLP
NZ = 128  # size of the latent z vector

# +++ Data Processing Settings +++ #
BATCH_SIZE = 64
DATA_LOADER_NUM_WORKERS = 2  # Potential bug if not set to zero.
                             # See https://github.com/pytorch/pytorch/issues/1355

# +++ Optimization Settings +++ #
LR_D = 1e-4  # 0.0001
LR_G = LR_D
def D_OPTIMIZER(net_dot_parameters):
    return torch.optim.Adam(net_dot_parameters, lr=LR_D, betas=(0.5, 0.9))
def G_OPTIMIZER(net_dot_parameters):
    return torch.optim.Adam(net_dot_parameters, lr=LR_G, betas=(0.5, 0.9))

# +++ Logging Settings +++ #
USE_TENSORBOARD = False # Keep it False for now. Currently a bug if True because the networks have nn.Sequential.
if USE_WASSERSTEIN_GRADIENT:
    EXPERIMENT_NAME = 'Experiment_' + get_time_now() + '___GAN_TYPE-' + GAN_TYPE + '__ARCHG-' + ARCHG + '__ARCHD-' + ARCHD + '___DATASET_NAME-' + DATASET_NAME + '__USE_WASSERSTEIN_GRADIENT-' + str(USE_WASSERSTEIN_GRADIENT) + '__H-' + str(H) + '__G_ITERS-' + str(G_ITERS)
else:
    EXPERIMENT_NAME = 'Experiment_' + get_time_now() + '___GAN_TYPE-' + GAN_TYPE + '__ARCHG-' + ARCHG + '__ARCHD-' + ARCHD + '___DATASET_NAME-' + DATASET_NAME + '__USE_WASSERSTEIN_GRADIENT-' + str(USE_WASSERSTEIN_GRADIENT)
EXPERIMENT_NAME = EXPERIMENT_NAME + '__D_ITERS-' + str(D_ITERS) + '__LAM-' + str(LAM)
OUTFILE_NAME = 'out.txt'
GENERATED_IMAGES_FOLDER_NAME = 'generated_images_folder'
FID_AVERAGES_FILENAME = 'FID_averages.csv'
LOGGING_RATE = 1000  # Also the calculating-fid-rate
MAX_OUTER_ITER = MAX_OUTER_ITER + LOGGING_RATE + 1

# ------------------------------------------------------------------------------- #
# Miscellaneous properties based off the above hyperparameters (no need to touch) #
# ------------------------------------------------------------------------------- #

# +++ Some properties of the dataset +++ #
if DATASET_NAME=='CIFAR10':
    NC = 3
    NUM_IMAGE_PIXELS = 3*32*32
    IMAGE_SIZE_ONE_SIDE = 32
elif DATASET_NAME=='CELEBA':
    NC = 3
    NUM_IMAGE_PIXELS = 3*128*128
    IMAGE_SIZE_ONE_SIDE = 128
elif DATASET_NAME=='CELEBA64':
    NC = 3
    NUM_IMAGE_PIXELS = 3*64*64
    IMAGE_SIZE_ONE_SIDE = 64
elif DATASET_NAME=='CELEBA16':
    NC = 3
    NUM_IMAGE_PIXELS = 3*16*16
    IMAGE_SIZE_ONE_SIDE = 16
else:
    raise ValueError("Bad dataset_name/not yet implemented.")

# }}}1
# </editor-fold>

# ============================================================ #
#                                                              #
#                            Models                            #
#                                                              #
# ============================================================ #

# <editor-fold desc="Models">
# Models {{{1

import torch.nn as nn

# ------------------------------------------ #
#                                            #
# The WGAN-GP DCGAN Architecture for CIFAR10 #
#                                            #
# ------------------------------------------ #
# WGAN-GP DCGAN Architecture for CIFAR10 {{{2
# The architectures here were taken from: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

# Architecture settings
if GAN_TYPE == 'WGANGP':
    DIM = DIM_WGANGP

class Generator_CIFAR10_WGANGP_DCGAN(nn.Module):
    def __init__(self):
        super(Generator_CIFAR10_WGANGP_DCGAN, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, inp):
        output = self.preprocess(inp)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator_CIFAR10_WGANGP_DCGAN(nn.Module):
    def __init__(self):
        super(Discriminator_CIFAR10_WGANGP_DCGAN, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, inp):
        output = self.main(inp)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

# }}}2

# -------------------------- #
#                            #
# WGAN-GP DCGAN for CelebA64 #
#                            #
# -------------------------- #

# WGAN-GP DCGAN for CelebA {{{2
# The architectures here were taken from: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

# CelebA 64x64

# Architecture settings
if GAN_TYPE == 'WGANGP':
    DIM = DIM_WGANGP

class Generator_CELEBA64_WGANGP_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator_CELEBA64_WGANGP_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     NZ, DIM * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(DIM * 8),
            nn.ReLU(True),
            # state size. (DIM*8) x 4 x 4
            nn.ConvTranspose2d(DIM * 8, DIM * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIM * 4),
            nn.ReLU(True),
            # state size. (DIM*4) x 8 x 8
            nn.ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIM * 2),
            nn.ReLU(True),
            # state size. (DIM*2) x 16 x 16
            nn.ConvTranspose2d(DIM * 2,     DIM, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            # state size. (DIM) x 32 x 32
            nn.ConvTranspose2d(    DIM,      NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (NC) x 64 x 64
        )

    def forward(self, inpinp):
        inp = inpinp.unsqueeze(-1).unsqueeze(-1)
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output


class Discriminator_CELEBA64_WGANGP_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator_CELEBA64_WGANGP_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (NC) x 64 x 64
            nn.Conv2d(NC, DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DIM) x 32 x 32
            nn.Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False),
            ### nn.BatchNorm2d(DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DIM*2) x 16 x 16
            nn.Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False),
            ### nn.BatchNorm2d(DIM * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DIM*4) x 8 x 8
            nn.Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False),
            ### nn.BatchNorm2d(DIM * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DIM*8) x 4 x 4
            nn.Conv2d(DIM * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)

# }}}2

# -------------------------- #
#                            #
# WGAN-GP DCGAN for CelebA16 #
#                            #
# -------------------------- #

# CelebA 16x16
# The architectures here were taken from: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

# {{{2

# Architecture settings
if GAN_TYPE == 'WGANGP':
    DIM = DIM_WGANGP

class Generator_CELEBA16_WGANGP_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator_CELEBA16_WGANGP_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     NZ, DIM * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(DIM * 2),
            nn.ReLU(True),
            # state size. (DIM*2) x 4 x 4
            nn.ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            # state size. (DIM) x 8 x 8
            nn.ConvTranspose2d(DIM, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inpinp):
        inp = inpinp.unsqueeze(-1).unsqueeze(-1)
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output


class Discriminator_CELEBA16_WGANGP_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator_CELEBA16_WGANGP_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (NC) x 16 x 16
            nn.Conv2d(NC, DIM, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DIM) x 8 x 8
            nn.Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False),
            ### nn.BatchNorm2d(DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (DIM*2) x 4 x 4
            nn.Conv2d(DIM * 2, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)

# }}}2

# ---------------------------------- #
#                                    #
# Vanilla GAN with DCGAN for CIFAR10 #
#                                    #
# ---------------------------------- #
# Vanilla loss function with DCGAN Architecture for CIFAR10  {{{2
# Code modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
#   in order to work for CIFAR10 3x32x32 images without resizing preprocess

if GAN_TYPE == 'Vanilla':
    # ngpu = NGPU
    nz = NZ
    ngf = NGF
    ndf = NDF
    nc = NC
    ngpu = 1

class Generator_CIFAR10_Vanilla_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator_CIFAR10_Vanilla_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, inpinp):
        inp = inpinp.unsqueeze(-1).unsqueeze(-1)
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output

class Discriminator_CIFAR10_Vanilla_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator_CIFAR10_Vanilla_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)

# --------------------------------------- #
#                                         #
# Vanilla GAN with DCGAN for CelebA (128) #
#                                         #
# --------------------------------------- #
# For CelebA (128)
# Code modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
#   in order to work for CIFAR10 3x32x32 images without resizing preprocess

if GAN_TYPE == 'Vanilla':
    # ngpu = NGPU
    nz = NZ
    ngf = NGF
    ndf = NDF
    nc = NC
    ngpu = 1

class Generator_CELEBA_Vanilla_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator_CELEBA_Vanilla_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, inpinp):
        inp = inpinp.unsqueeze(-1).unsqueeze(-1)
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output


class Discriminator_CELEBA_Vanilla_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator_CELEBA_Vanilla_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)


# --------------------------------------- #
#                                         #
# Vanilla GAN with DCGAN for CelebA64 #
#                                         #
# --------------------------------------- #
# For CelebA64
# Code modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
#   in order to work for CIFAR10 3x32x32 images without resizing preprocess

if GAN_TYPE == 'Vanilla':
    # ngpu = NGPU
    nz = NZ
    ngf = NGF
    ndf = NDF
    nc = NC
    ngpu = 1

class Generator_CELEBA64_Vanilla_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator_CELEBA64_Vanilla_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inpinp):
        inp = inpinp.unsqueeze(-1).unsqueeze(-1)
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output


class Discriminator_CELEBA64_Vanilla_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator_CELEBA64_Vanilla_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)

# }}}2


# ----------------------------------------------- #
#                                                 #
# DRAGAN loss with DCGAN Architecture for CIFAR10 #
#                                                 #
# ----------------------------------------------- #
# DRAGAN with DCGAN Architecture  {{{2
# The only difference between this architecture and Vanilla is the discriminator
#   does not have batch normalization.
# Code modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
#   in order to work for CIFAR10 3x32x32 images without resizing preprocess

if GAN_TYPE == 'DRAGAN':
    # ngpu = NGPU
    nz = NZ
    ngf = NGF
    ndf = NDF
    nc = NC
    ngpu = 1


class Generator_CIFAR10_DRAGAN_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator_CIFAR10_DRAGAN_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, inpinp):
        inp = inpinp.unsqueeze(-1).unsqueeze(-1)
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output

# Removed the batch norm for the discriminator
class Discriminator_CIFAR10_DRAGAN_DCGAN(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator_CIFAR10_DRAGAN_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            ###nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            ###nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)

# }}}2


# ----------------- #
#                   #
# General MLP model #
#                   #
# ----------------- #

# General MLP model {{{2

class Generator_MLP_4Layer(nn.Module):
    def __init__(self):
        super(Generator_MLP_4Layer, self).__init__()
        self.lin1 = nn.Linear(NZ, NGF)
        self.lin2 = nn.Linear(NGF, NGF)
        self.lin3 = nn.Linear(NGF, NGF)
        self.lin4 = nn.Linear(NGF, NUM_IMAGE_PIXELS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = self.act(self.lin1(inp))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.lin4(x)
        return x.view(x.size(0), NC, IMAGE_SIZE_ONE_SIDE, IMAGE_SIZE_ONE_SIDE)

class Discriminator_MLP_4Layer(nn.Module):
    def __init__(self):
        super(Discriminator_MLP_4Layer, self).__init__()
        self.lin1 = nn.Linear(NUM_IMAGE_PIXELS, NDF)
        self.lin2 = nn.Linear(NDF, NDF)
        self.lin3 = nn.Linear(NDF, NDF)
        self.lin4 = nn.Linear(NDF, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inp):
        inp_reshaped = inp.view(inp.size(0), -1)
        x = self.act(self.lin1(inp_reshaped))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.lin4(x)
        return x

# }}}2

# }}}1
# </editor-fold>


# =============================================================== #
#                                                                 #
# Set the Discriminator and the Generator (no need to touch this) #
#                                                                 #
# =============================================================== #
# <editor-fold desc="Setting the Discriminator and Generator">
# Setting the Discriminator and Generator {{{1
# +++ Set the GAN Discriminator and Generator based off above settings +++ #
if DATASET_NAME == 'CIFAR10':
    if GAN_TYPE == 'WGANGP':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CIFAR10_WGANGP_DCGAN
        elif ARCHG == 'MLP':
            GENERATOR = Generator_MLP_4Layer
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CIFAR10_WGANGP_DCGAN
        elif ARCHD == 'MLP':
            DISCRIMINATOR = Discriminator_MLP_4Layer
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")
    elif GAN_TYPE == 'Vanilla':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CIFAR10_Vanilla_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_GENERATOR = True
        elif ARCHG == 'MLP':
            GENERATOR = Generator_MLP_4Layer
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CIFAR10_Vanilla_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_DISCRIMINATOR = True
        elif ARCHD == 'MLP':
            DISCRIMINATOR = Discriminator_MLP_4Layer
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")
    elif GAN_TYPE == 'DRAGAN':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CIFAR10_DRAGAN_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_GENERATOR = True
        elif ARCHG == 'MLP':
            GENERATOR = Generator_MLP_4Layer
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CIFAR10_DRAGAN_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_DISCRIMINATOR = True
        elif ARCHD == 'MLP':
            DISCRIMINATOR = Discriminator_MLP_4Layer
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")
    else:
        raise ValueError("Bad GAN_TYPE name, or not yet implemented.")
elif DATASET_NAME == 'CELEBA':
    if GAN_TYPE == 'Vanilla':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CELEBA_Vanilla_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_GENERATOR = True
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CELEBA_Vanilla_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_DISCRIMINATOR = True
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")
elif DATASET_NAME == 'CELEBA64':
    if GAN_TYPE == 'Vanilla':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CELEBA64_Vanilla_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_GENERATOR = True
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CELEBA64_Vanilla_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_DISCRIMINATOR = True
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")
    elif GAN_TYPE == 'WGANGP':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CELEBA64_WGANGP_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_GENERATOR = True
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CELEBA64_WGANGP_DCGAN
            DO_CUSTOM_WEIGHTS_INIT_DISCRIMINATOR = True
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")

elif DATASET_NAME == 'CELEBA16':
    if GAN_TYPE == 'WGANGP':
        if ARCHG == 'DCGAN':
            GENERATOR = Generator_CELEBA16_WGANGP_DCGAN
        else:
            raise ValueError("Bad ARCHG name, or not yet implemented.")

        if ARCHD == 'DCGAN':
            DISCRIMINATOR = Discriminator_CELEBA16_WGANGP_DCGAN
        else:
            raise ValueError("Bad ARCHD name, or not yet implemented.")

else:
    raise ValueError("Bad DATASET_NAME, or not yet implemented.")

# }}}1
# </editor-fold>

