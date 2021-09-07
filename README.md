
## GENERATIVE MODEL PROJECT: IDGAN

<br><br>
This is an implementation of [IDGAN](https://arxiv.org/abs/2001.04296). It trains on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) "Align&Cropped Images" dataset. The gan structure was taken from the ["Official PyTorch implementation"](https://github.com/1Konny/idgan).

Modify *configs.yml* to change configurations and hyperparameters.

First preprocess the dataset

    python preprocessing.py


Second train vae

    python train_vae.py


Third train idgan

    python train_gan.py


