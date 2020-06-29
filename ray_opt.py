from load_cfg import Config
from wwgan_utils import make_log_dir
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import train
import json
import argparse
from functools import partial
import ray
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np
from ww_trainable_ray import ww_trainable_ray



def small_parser():
    parser = argparse.ArgumentParser(description='config uploading')
    parser.add_argument('--config', default=None, type=str,
                        help='python file that contains the config of a file')
    return parser.parse_args()

def train_from_cfg(args, reporter):
    # args.log_dir = make_log_dir(prefix='/home/yoni/repos/WWGAN/logs')

    # Getting the log directory from ray.tune
    args.log_dir = tune.get_trial_dir()
    out_log_name = args.log_dir + '/out.txt'
    out_log = open(out_log_name, 'w')

    with open(args.log_dir + '/config.json', 'w+') as arg_file:
        cfg_dict = {k: v for k, v in args.items()}
        json.dump(cfg_dict, arg_file)

    # Intialize network, optimizer, and set-aside fixed noise for generator evaluation
    args.device = torch.device("cuda" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    # Create the generator and discriminator
    args.netG = train.Generator(args.ngpu).to(args.device)
    args.netD = train.Discriminator(args.ngpu).to(args.device)

    if args.resume is not None:
        try:
            args.netG.load_state_dict(torch.load(args.resume + '/Gen.pth.tar'))
            print('Loaded generator from '+ args.resume)
        except:
            print('Failed to load generator from ', args.resume)
        try:
            args.netD.load_state_dict(torch.load(args.resume + '/Disc.pth.tar'))
            print('loaded discriminator from ' + args.resume)
        except:
            print('Failed to load discriminator ', args.resume)

    else:
        # custom init
        args.netG.apply(train.weights_init)
        args.netD.apply(train.weights_init)

    # Load models to multi-gpu if desired
    if (args.device.type == 'cuda') and (args.ngpu > 1):
        args.netG = torch.nn.DataParallel(args.netG, list(range(args.ngpu)))
        args.netD = torch.nn.DataParallel(args.netD, list(range(args.ngpu)))

    print(args.netG, args.netD, file=out_log)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    args.fixed_noise = torch.randn(64, args.nz, 1, 1, device=args.device)

    # Setup Adam optimizers for both G and D
    args.optimizerD = torch.optim.Adam(args.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.9))
    args.optimizerG = torch.optim.Adam(args.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.9))
    # train.train(args)
    ww_trainable_ray(args, reporter)



def train_from_split_cfg(optim_cfg, reporter, full_cfg):
    for k,v in optim_cfg.items():
        full_cfg[k] = v
    train_from_cfg(full_cfg, reporter)


def ray_mini_test(cfg, reporter):
    import numpy as np
    print(cfg)
    for k,v in cfg.items():
        print(k, '---', v)
    score = int(10*np.random.randn(1))
    print('FID', score)
    reporter(FID_score=score)


if __name__ == '__main__':
    parsed = small_parser()
    args = Config.fromfile(parsed.config)
    ray.init(num_cpus=12, num_gpus=6) # request number of GPUs and CPUs and total memory

    # pass full config and incorporate tune params.
    ray_trainable = partial(train_from_split_cfg, full_cfg=args.args)

    bten = np.log(10)
    # search space coming from this distribution
    search_space = {
        "lr": hp.loguniform('lr',  -5*bten, -3.4*bten,),
        "gam": hp.loguniform('gam', -2*bten, 1*bten),
        "lam": hp.loguniform('lam', 1.5*bten, 2*bten),
    }

    hyperopt_search = HyperOptSearch(search_space, metric="FID_score", mode="min")


    # Uncomment this to enable distributed execution
    # `ray.init(address=...)`

    ray_run ='wwgan_lr_gamlam_fix_bug'
    analysis = tune.run(ray_trainable, name=ray_run, num_samples=8, search_alg=hyperopt_search, resources_per_trial={'gpu': 2}, local_dir='/home/yoni/repos/WWGAN/logs/ray_logs')
    print(analysis.dataframe())

