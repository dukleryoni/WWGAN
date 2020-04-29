# WWGAN
A simple implementation of WWGAN and the Wasserstein image space gradient penalty.

Based on [Marvin Cao's wgan-gp](https://github.com/caogang/wgan-gp) and ADD dc_gan

_____
The Wasserstein of Wasserstein loss for generative models, assumes an optimal transport metric as a distance between samples. 
Details can be found from our paper [*Wasserstein of Wasserstein Loss for Learning Generative Models*](http://proceedings.mlr.press/v97/dukler19a/dukler19a.pdf).



## Getting started

### Setup

Clone the repository using 

```bash
$ git clone https://github.com/dukleryoni/WWGAN.git
```

**Download training data**
Download and extract the [CelebA image dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) from google drive. See [`gdown`](https://pypi.org/project/gdown/) for downloading from google drive on command line.

For evaluating the network using the Freceht Inception Distance (FID) score we use the [`TTUR`](https://github.com/bioinf-jku/TTUR) repository.
In the `WWGAN` repository,  we clone `TTUR` and download the pre-computed FID statistics for CelebA:

```bash
$ mkdir Frechet_Inception_Distance
$ cd Frechet_Inception_Distance
$ git clone https://github.com/bioinf-jku/TTUR.git
$ wget http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz # get pre-computed stats for FID
```

Now in the `WWGAN` repo train the model by simply running `python train.py` 








