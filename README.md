# Wasserstein of Wasserstein Loss for Generative Models (WWGAN)
A simple implementation of WWGAN and the Wasserstein image-space gradient penalty (Wasserstein ground metric).

Based on [Marvin Cao's wgan-gp](https://github.com/caogang/wgan-gp) and [Nathan Inkawhich's](https://github.com/inkawhich) example.

_____
The Wasserstein of Wasserstein loss for generative models, uses an optimal transport metric as the distance measure between images. This is then formulated in the WGAN-GP formulation.

Details can be found in our paper [*Wasserstein of Wasserstein Loss for Learning Generative Models*](http://proceedings.mlr.press/v97/dukler19a/dukler19a.pdf).

## Getting started

### Setup

Clone the repository using 

```bash
$ git clone https://github.com/dukleryoni/WWGAN.git

```

#### Prerequisites
The following packages are required to run the repo: PyTorch, torchvision, Scipy, Pillow, TensorFlow, TensorBoard.
For your convinience, you can create the suitable conda environment using `wwgan_env.yml` by running

```bash
$ conda create --name name_of_wwgan_env --clone wwgan_env.yml
```

#### Download training data
Download and extract the [CelebA image dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) from google drive. See [`gdown`](https://pypi.org/project/gdown/) for downloading from google drive on command line.

In `train.py` , line 34, change `dataroot` to the correct path to the downloaded CelebA directory.

#### Evaluation using FID
For evaluating the network using the Freceht Inception Distance (FID) score we use the [`TTUR`](https://github.com/bioinf-jku/TTUR) repository.
In the `WWGAN` repository,  we clone `TTUR` and download the pre-computed FID statistics for CelebA:

```bash
$ mkdir Frechet_Inception_Distance
$ cd Frechet_Inception_Distance
$ git clone https://github.com/bioinf-jku/TTUR.git
$ wget http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz # get pre-computed stats for FID
```

####Run and inspect results
Now in the `WWGAN` repo train the model by simply running `python train.py` 

The user can specify hyperparamters for diffferent runs, (e.g. `--ngpu 2` for number of GPUs for training).

After training the user can inspect simple properties of the generator using the Jupyter notebook `Analyze_generated_images.ipynb`.

### Wasserstein ground metric gradient penalty
The code for computing the Wasserstein gradient penalty is given in `wwgan_utils.py` and invloves the most new implementation details. One can compute the Wsserstein ground metic by calling the  `calc_wgp()` function.