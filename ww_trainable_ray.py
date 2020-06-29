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
from wwgan_utils import calculate_fid, calc_wgp, make_log_dir, calc_gradient_penalty, load_from_json
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from ray import tune
import logging

def ww_trainable_ray(args, reporter):
    # Root directory for dataset
    dataroot = "/home/yoni/Datasets/img_align_celeba_full"

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch,
                                             shuffle=True, num_workers=args.workers, drop_last=False)

    # save text output into file
    out_log_name = args.log_dir+ '/out.txt'
    out_log = open(out_log_name, 'w')

    print("Starting Training Loop...", file=out_log)

    if args.no_wass:
        print('using traditional L2 penalty (WGAN-GP)', file=out_log)
    else:
        print('Using the WWGAN penalty', file=out_log)
    print('The gradient penalty scalar (lambda) used is {}'.format(args.lam), file=out_log)

    iters = 0
    min_score = 1.0e6
    errG = torch.cuda.FloatTensor([1], device=args.device) # since logging occurs before first G-pass
    D_G_z2 = 0
    last_time = time.time()

    model_dir = 'saved_model'
    os.mkdir(args.log_dir + '/' + model_dir)
    # Tensorboard
    writer = SummaryWriter(args.log_dir + '/tf_events')
    netD = args.netD
    netG= args.netG
    optimizerD = args.optimizerD
    optimizerG = args.optimizerG
    fixed_noise = args.fixed_noise

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
            real_gpu = data[0].to(args.device)
            b_size = real_gpu.size(0)
            output_real = netD(real_gpu).view(-1)
            errD_real = output_real.mean()
            D_x = output_real.mean().item()
            # Train with all-fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=args.device)
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
            if iters % args.critic_iter == args.critic_iter - 1:
                for param in netD.parameters():
                    param.requires_grad_(False)
                netG.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                noise = torch.randn(args.batch, args.nz, 1, 1, device=args.device)
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
                         errD.item(), -1*errG.item(), GP, D_x, D_G_z1, D_G_z2), file=out_log)

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


        print('calculating FID now', file=out_log)
        score = calculate_fid(netG, args.nz, data='celebA', batch_size=args.batch, fid_stats_path='/home/yoni/repos/WWGAN/fid_stats_celeba.npz')
        reporter(FID_score=score) # ToDo added ray tuning

        writer.add_scalar('FID_score', score, iters)
        if score < min_score:
            min_score = score
            torch.save(netG.state_dict(), args.log_dir+ '/' + model_dir +'/Gen.pth.tar')
        torch.save(netD.state_dict(), args.log_dir + '/' + model_dir + '/Disc.pth.tar') # Todo does this make sense? saving D after every epoch...
    writer.close()
