import datetime
import time
import torchvision.utils as vutils
import torch
from torch import autograd
import Frechet_Inception_Distance.TTUR.fid as fid
import torch.nn as nn


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Helper function to get time for logging purposes
def get_time_now():
    date_string = str(datetime.datetime.now())
    day = date_string[0:10]
    hour = date_string[11:13]
    minute = date_string[14:16]
    second = date_string[17:19]
    return day + '--' + hour + '-' + minute + '-' + second


# Calculates FID score of generator for CelebA
def calculate_fid(generator, nz, data, batch_size):
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

    fid_score = fid.calculate_fid_given_paths(paths=[generated_images_folder_path, fid_stats_path],
                                                              inception_path='./inception_model/')
    finish_t = time.time()
    print('The fid score is {} and was calcualted in {} (seconds)'.format(fid_score, (finish_t - start_t)))
    return fid_score


# Calculates gradient for WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, batch_size, nc, image_size, LAMBDA):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size, nc, image_size, image_size)
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


def buildL_diamond(grad_batch, image_batch, batch_size, color_channel=True):
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


def calc_wgp(netD, real_data, fake_data, batch_size, nc, image_size, LAMBDA, gamma =1, a=0, b=0):
    '''
    same as calc_gradient_penalty but with wasserstein L gradient
    a is the weight of global connections
    b is the weight for image normalization
    '''
    ### set a and b to 0 for simplicity



    alpha = torch.rand(batch_size, 1, device=device) # ToDO can deduce this from the input data perhaps, so that we don't have to pass inputs
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size, nc, image_size, image_size)
    alpha = alpha.to(device)

    num_pixels = image_size * image_size * nc

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True) # use .requires_grad=True

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]


    w_grad = buildL_diamond(gradients, interpolates, batch_size, color_channel=False) # From the eucleadian gradient we compute the Wasserstein gradient

    gradients = gradients.view(gradients.size(0), -1)
    L2_grad_sq = 2*gradients.norm(2, dim=1)**2
    w_grad_sum_sq = (2/num_pixels)*torch.sum(gradients,1)**2
    full_w_grad = torch.sqrt(w_grad**2 + a * L2_grad_sq + b * w_grad_sum_sq) # Here a = alpha/num_pixels and b = beta - 2*alpha / num_pixels


    gradient_penalty = ((gradients.norm(2, dim=1)-1) ** 2).mean() * LAMBDA
    #gradient_penalty = ((w_grad*gamma- 1) ** 2).mean() * LAMBDA

    return gradient_penalty, w_grad.data.mean().item(), torch.sqrt(L2_grad_sq.data).mean().item(), torch.sqrt(w_grad_sum_sq).mean().item()