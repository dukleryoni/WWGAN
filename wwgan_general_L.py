import datetime
import time
import torchvision.utils as vutils
import torch
from torch import autograd
import Frechet_Inception_Distance.TTUR.fid as fid
import torch.nn as nn
import  numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# Calculates gradient for WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, batch_size, nc, image_size, LAMBDA):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, real_data.nelement() // batch_size).contiguous().view(batch_size, nc, image_size,
                                                                                           image_size)
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


def shift_diamond(X, a, b, c=0, d=0, dim=0, diag=False):
    """
    Input batched images of size (batch, nc, im_height, im_width)
    Applys operation X[i]*a -X[i+1]*b for the dimension chosen dim = 0 horizontal, dim = 1 vertical, dim =2 betwween channels
    """
    if diag:
        filt = torch.FloatTensor(
            [a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c,
             d])
        conv1 = nn.Conv2d(3, 3, (2, 2), stride=1, bias=False)
        filt = filt.view(3, 3, 2, 2)
        conv1.weight.data = filt

    else:
        if dim == 0:
            filt = torch.FloatTensor([a, b, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c])
            conv1 = nn.Conv2d(3, 3, (1, 3), stride=1, bias=False)
            filt = filt.view(3, 3, 1, 3)
            conv1.weight.data = filt

        elif dim == 1:
            filt = torch.FloatTensor([a, b, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c])
            conv1 = nn.Conv2d(3, 3, (3, 1), stride=1, bias=False)
            filt = filt.view(3, 3, 3, 1)
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
    - returns the Wasserstein Gradient Squared
    """
    w_grad_sq = 0
    total_dims = 2
    if color_channel:
        total_dims = 3

    for dim in range(0, total_dims):
        fminusf = shift(grad_batch, 1, -1, dim)
        fminusf_sq = fminusf ** 2
        XplusX = shift(image_batch, 1, 1, dim)  # By calling sum we are summing over all entries in an image
        #  and all batches, we normalize by batch size
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    w_grad = torch.sqrt(w_grad_sq)
    # must normalize each image as (x+1)/2
    return w_grad  # This leads to normalizing XplusX as XplusX/2 + 1


def buildL_Random(grad_batch, image_batch, batch_size, color_channel=True):
    """
    Calculates grad f L(X) grad f^T ,
    - grad batch is gradient of D with respect to image batch
    - image batch is a normalized batch of images normalized according to graph degree
    - normalization does'nt matter for our case except in edges
    - create L to also make a color channel
    - returns the wasserstein gradient squared
    """

    # Fixed Random Permutation
    np.random.seed(0)

    image_size = image_batch.size()[-1]
    row_perm_tensor = torch.tensor(np.eye(image_size)[np.random.permutation(image_size)]).float().to(device)
    col_perm_tensor = torch.tensor(np.eye(image_size)[np.random.permutation(image_size)]).float().to(device)

    grad_batch = row_perm_tensor.matmul(grad_batch).matmul(col_perm_tensor)
    image_batch = row_perm_tensor.matmul(image_batch).matmul(col_perm_tensor)

    w_grad_sq = 0
    total_dims = 2
    if color_channel:
        total_dims = 3

    for dim in range(0, total_dims):
        fminusf = shift(grad_batch, 1, -1, dim)
        fminusf_sq = fminusf ** 2
        XplusX = shift(image_batch, 1, 1, dim)  # By calling sum we are summing over all entries in an image
        #  and all batches, we normalize by batch size
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    w_grad = torch.sqrt(w_grad_sq)
    # must normalize each image as (x+1)/2
    return w_grad  # This leads to normalizing XplusX as XplusX/2 + 1


def buildL_diamond_Random(grad_batch, image_batch, batch_size, color_channel=True):
    """
    - USES RANDOM PIXEL LOCATIONS FOR CONVOLUTIONS
    Calculates grad f L(X) grad f^T ,
    - grad batch is gradient of D with respect to image batch
    - image batch is a normalized batch of images normalized according to graph degree
    - normalization does'nt matter for our case except in edges
    - create L to also make a color channel
    - returns the wasserstein gradient squared
    """

    # Fixed Random Permutation
    np.random.seed(0)

    image_size = image_batch.size()[-1]
    row_perm_tensor = torch.tensor(np.eye(image_size)[np.random.permutation(image_size)]).float().to(device)
    col_perm_tensor = torch.tensor(np.eye(image_size)[np.random.permutation(image_size)]).float().to(device)

    grad_batch = row_perm_tensor.matmul(grad_batch).matmul(col_perm_tensor)
    image_batch = row_perm_tensor.matmul(image_batch).matmul(col_perm_tensor)
    print(np.random.rand())
    w_grad_sq = 0
    total_dims = 2
    if color_channel:
        total_dims = 3

    # immediate neighbors
    for dim in range(0, total_dims):
        fminusf = shift_diamond(grad_batch, 1, -1, dim)
        fminusf_sq = fminusf ** 2
        XplusX = shift_diamond(image_batch, 1, 1, dim)  # By calling sum we are summing over all entries in an image
        #  and all batches, we normalize by batch size
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    # far neighbors
    for dim in range(0, 2):
        fminusf = shift_diamond(grad_batch, 1, 0, -1, dim)
        fminusf_sq = fminusf ** 2

        XplusX = shift_diamond(image_batch, 1, 0, 1, dim)
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    # Diagonals
    for (a, b, c, d) in [(1, 0, 0, 1), (0, 1, 1, 0)]:
        fminusf = shift_diamond(grad_batch, a, b, -c, -d, True)
        fminusf_sq = fminusf ** 2

        XplusX = shift_diamond(image_batch, a, b, c, d, True)
        w_grad_sq += torch.sum((fminusf_sq * (1 + XplusX / 2)), (1, 2, 3))

    w_grad = torch.sqrt(w_grad_sq)
    # must normalize each image as (x+1)/2
    return w_grad


def calc_wgp_Random(netD, real_data, fake_data, batch_size, nc, image_size, LAMBDA, gamma=1, a=0, b=0):
    '''
    same as calc_gradient_penalty but with wasserstein L gradient
    a is the weight of global connections
    b is the weight of for image normalization
    '''
    ### set a and b to 0 for simplicity



    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, real_data.nelement() // batch_size).contiguous().view(batch_size, nc, image_size,
                                                                                           image_size)
    alpha = alpha.to(device)

    num_pixels = image_size * image_size * nc

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    w_grad = buildL_diamond_Random(gradients, interpolates, batch_size, color_channel=False)

    gradients = gradients.view(gradients.size(0), -1)
    L2_grad_sq = 2 * gradients.norm(2, dim=1) ** 2
    w_grad_sum_sq = (2 / num_pixels) * torch.sum(gradients, 1) ** 2
    full_w_grad = torch.sqrt(
        w_grad ** 2 + a * L2_grad_sq + b * w_grad_sum_sq)  # Here a = alpha/num_pixels and b = beta - 2*alpha / num_pixels

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    # gradient_penalty = ((w_grad*gamma- 1) ** 2).mean() * LAMBDA

    return gradient_penalty, w_grad.data.mean().item(), torch.sqrt(L2_grad_sq.data).mean().item(), torch.sqrt(
        w_grad_sum_sq).mean().item()

