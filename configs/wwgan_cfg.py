# simple toy config
args = dict(
    lr=1e-4,
    critic_iter=5,
    workers=4,
    nc=3,
    image_size=64,
    resume=None,
    beta1 = 0.5,
    lam = 4.0,
    gam = 1.0,
    epoch=40,
    ngpu=2, # ToDo remember to change this according to ray tune parameters of gpu per trial
    batch=128,
    no_wass=False,
    from_json=False,
    nz=100,
    ngf=64,
    ndf=64,
)
