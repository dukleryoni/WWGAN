import datetime
import time
import torchvision.utils as vutils
import torch
from torch import autograd
import pytorch_fid.fid_score as fid_torch
import torch.nn as nn
import os
import numpy as np
import json

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Helper function to get time for logging purposes
def get_time_now_split(split=True):
    date_string = str(datetime.datetime.now())
    day = date_string[0:10]
    hour = date_string[11:13]
    minute = date_string[14:16]
    second = date_string[17:19]
    dirs = [day[:-3], day, day + '--' + hour + '-' + minute + '-' + second]
    if split:
        return dirs
    return dirs[-1]


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def load_from_cfg(cfg_path):
    print('Loading the arguments from the config file {}'.format(cfg_path))
    arg_dict = json.load(open(cfg_path, 'r+'))
    return AttributeDict(arg_dict)


def make_log_dir(prefix=''):
    dirs = get_time_now_split()
    dirs.insert(0, prefix)
    try:
        os.mkdir('/'.join(dirs))
    except:
        try:
            print('1st of tha month')
            [os.mkdir('/'.join(dirs[0:i])) for i in range(3, 5)]
        except:
            print('Happy new year!')
            [os.mkdir('/'.join(dirs[0:i])) for i in range(2, 5)]
    return '/'.join(dirs)


# Calculates FID score of generator for CelebA
def calculate_fid(generator, nz, data, batch_size, cuda=True):
    if data == 'cifar10':
        fid_stats_path = './fid_stats_cifar10_train.npz'
    elif data == 'celebA':
        fid_stats_path = './fid_stats_celeba.npz'

    #Saves images to be calculated for FID - not necessary why not pass through inception first to save time
    start_t = time.time()
    generated_images_folder_path = '/home/yoni/Datasets/fid_images'
    number_fid = 5000//batch_size
    for idx in range(0, number_fid):
        z_fid = torch.randn(batch_size, nz, 1, 1, device=device)
        g_z_fid = generator(z_fid)
        for idx_fid in range(0, batch_size):
            vutils.save_image(tensor=g_z_fid[idx_fid],
                                         fp=generated_images_folder_path + '/' + 'img' + str(
                                             idx * batch_size + idx_fid) + '.png',
                                         nrow=1,
                                         padding=0)

    # fid_score = fid.calculate_fid_given_paths(paths=[generated_images_folder_path, fid_stats_path],
    #                                                           inception_path='./inception_model/')
    fid_score = fid_torch.calculate_fid_given_paths(paths=[generated_images_folder_path, fid_stats_path], batch_size=batch_size, dims=2048, cuda=cuda)
    finish_t = time.time()
    print('The fid score is {} and was calcualted in {} (seconds)'.format(fid_score, (finish_t - start_t)))
    return fid_score

def shift(X, a, b, dim=0):
    """
    Input:
     - batched images of size (batch, nc, im_height, im_width)
     - a,b values to multiply adjacent coordinates
     - dim dimension along which we apply the shift: dim = 0 horizontal, dim = 1 vertical, dim =2 between channels
     - Applies operation X[i]*a -X[i+1]*b for the dimension chosen
     - returns the calculated shift, which has width and height dimensions 1 less than X
    """
    if dim == 0:
        filt = torch.FloatTensor([a, b, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, a, b])
        conv1 = nn.Conv2d(3, 3, (1, 2), stride=1, bias=False)
        filt = filt.view(3, 3, 1, 2)
        conv1.weight.data = filt

    elif dim == 1:
        filt = torch.FloatTensor([a, b, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, a, b])
        conv1 = nn.Conv2d(3, 3, (2, 1), stride=1, bias=False)
        filt = filt.view(3, 3, 2, 1)
        conv1.weight.data = filt

    elif dim == 2:
        filt = torch.FloatTensor([a, b, 0, a, 0, b, 0, a, b])
        conv1 = nn.Conv2d(3, 3, (1, 1), stride=1, bias=False)
        filt = filt.view(3, 3, 1, 1)
        conv1.weight.data = filt

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    conv1.to(device)
    conv1.requires_grad = False
    return conv1(X)


def shift_diamond_depthwise(X, a, b, c=0, d=0, dim=0, diag=False):
    """
    Input batched images of size (batch, nc, im_height, im_width)
    Applies operation X[i]*a -X[j]*b for the dimension chosen dim = 0 horizontal, dim = 1 vertical, dim =2 betwween channels
    """
    if diag:
        filt = torch.FloatTensor(
            [a, b, c, d] * 3)
        conv1 = nn.Conv2d(3, 3, (2, 2), groups=3, stride=1, bias=False)
        filt = filt.view(3, 1, 2, 2)
        conv1.weight.data = filt

    else:
        if dim == 0:
            filt = torch.FloatTensor([a, b, c]*3)
            conv1 = nn.Conv2d(3, 3, (1, 3), groups=3, stride=1, bias=False)
            filt = filt.view(3, 1, 1, 3)
            conv1.weight.data = filt

        elif dim == 1:
            filt = torch.FloatTensor([a, b, c]*3)
            conv1 = nn.Conv2d(3, 3, (3, 1), stride=1, groups=3, bias=False)
            filt = filt.view(3, 1, 3, 1)
            conv1.weight.data = filt

        elif dim == 2:
            filt = torch.FloatTensor([a, b, 0, a, 0, b, 0, a, b])
            conv1 = nn.Conv2d(3, 3, (1, 1), stride=1, bias=False)
            filt = filt.view(3, 3, 1, 1)
            conv1.weight.data = filt

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    conv1.to(device)
    conv1.requires_grad = False
    return conv1(X)


def buildL(grad_batch, image_batch, batch_size, color_channel=True):
    """
    Calculates grad f L(X) grad f^T ,
    - grad batch is gradient of D with respect to image batch
    - image batch is a normalized batch of images normalized according to graph degree
    - normalization does'nt matter for our case except in edges
    - create L to also make a color channel
    - returns the wasserstein gradient squared
    """
    w_grad_sq = 0
    total_dims = 2
    if color_channel:
        total_dims = 3

    for dim in range(0, total_dims):
        fminusf = shift(grad_batch, 1, -1, dim)
        fminusf_sq = fminusf ** 2
        XplusX = shift(image_batch, 1, 1, dim)                 # By calling sum we are summing over all entries in an image
                                                               #  and all batches, we normalize by batch size
        w_grad_sq += torch.sum((fminusf_sq * (1+ XplusX/2)), (1,2,3))

    w_grad = torch.sqrt(w_grad_sq)
                                                                # must normalize each image as (x+1)/2
    return w_grad                                              # This leads to normalizing XplusX as XplusX/2 + 1


def buildL_diamond(grad_batch, image_batch, color_channel=True):
    """
    Calculates grad f L(X) grad f^T ,
    - grad batch is gradient of D with respect to image batch
    - image batch is a normalized batch of images normalized according to graph degree
    - normalization does'nt matter for our case except in edges
    - create L to also make a color channel
    - returns the wasserstein gradient squared
    """
    w_grad_sq = 0
    total_dims = 2
    if color_channel:
        total_dims = 3

    # immediate neighbors
    for dim in range(0, total_dims):
        fminusf = shift_diamond_depthwise(grad_batch, 1, -1, dim=dim)
        fminusf_sq = fminusf ** 2
        XplusX = shift_diamond_depthwise(image_batch, 1, 1, dim=dim)  # By calling sum we are summing over all entries in an image
        #  and all batches, we normalize by batch size
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    # farther neighbors
    for dim in range(0, 2):
        fminusf = shift_diamond_depthwise(grad_batch, 1, 0, -1, dim=dim) # f_i - f_j
        fminusf_sq = fminusf ** 2

        XplusX = shift_diamond_depthwise(image_batch, 1, 0, 1, dim=dim) # X_i + X_j
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    # # Diagonals
    for (a, b, c, d) in [(1, 0, 0, 1), (0, 1, 1, 0)]:
        fminusf = shift_diamond_depthwise(grad_batch, a, b, -c, -d, diag=True)
        fminusf_sq = fminusf ** 2

        XplusX = shift_diamond_depthwise(image_batch, a, b, c, d, diag=True)
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3)) # Undos normalization taking [-1,1] -> [0,1] as done by Dataset and torchvision.transforms

    # w_grad = torch.sqrt(w_grad_sq)
    return w_grad_sq # return the sqaured version.

# Calculates gradient for WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA):
    # print "real_data: ", real_data.size(), fake_data.size()
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def calc_wgp(netD, real_data, fake_data, LAMBDA, GAMMA=1):
    '''
    same as calc_gradient_penalty but with wasserstein L gradient
    a is the weight of global connections
    b is the weight for image normalization
    '''
    ### set a and b to 0 for simplicity

    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device) # ToDO can deduce this from the input data perhaps, so that we don't have to pass inputs
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]



    w_grad_sq = buildL_diamond(grad_batch=gradients, image_batch=interpolates, color_channel=False) # Using the Eucleadian gradient we compute the Wasserstein gradient

    # Adding normalization factor in the gradient (L2) , sto perform on un-normalized potentials.
    gradients = gradients.view(batch_size, -1)
    L2_grad_sq = torch.sum(torch.mul(gradients, gradients), dim=1)

    w_grad_sq += L2_grad_sq * GAMMA

    gradient_penalty = ((torch.sqrt(w_grad_sq) - 1)**2).mean() * LAMBDA

    with torch.no_grad():
        gradients = gradients.view(batch_size, -1)
        L2_grad = gradients.norm(2, dim=1)
        sum_grad_normalized = 2.0*L2_grad**2 - 2.0*torch.sum(gradients/(np.sqrt(gradients.size(1))), dim=1)**2

    return gradient_penalty, w_grad_sq.detach().mean().item(), L2_grad.detach().mean().item(), torch.sqrt(sum_grad_normalized.detach()).mean().item()