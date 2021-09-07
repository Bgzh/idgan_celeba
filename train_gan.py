import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from time import time

from models.vae import BetaVAE
from models.gan import Generator, Discriminator
import utils

configs = utils.load_configs()
IMG_SIZE = configs['gan']['IMG_SIZE']
CELEBA_PATH = utils.get_celeba_path(IMG_SIZE, configs)
BATCH_SIZE = configs['gan']['BATCH_SIZE']
LR_GAN = configs['gan']['LR']
REG_PARAM = configs['gan']['REG_PARAM']
W_INFO = configs['gan']['W_INFO']
BETA = configs['vae']['BETA']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = utils.CelebA(CELEBA_PATH, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        drop_last=True,
                                        )

vae = BetaVAE(beta=BETA).to(device)
vae = utils.load_last_cp(vae, 'v')

generator = Generator(256+20, IMG_SIZE).to(device)
discriminator = Discriminator(256+20, IMG_SIZE).to(device)
generator = utils.load_last_cp(generator, 'g')
discriminator = utils.load_last_cp(discriminator, 'd')

g_optimizer = optim.RMSprop(generator.parameters(), lr=LR_GAN)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=LR_GAN)

def sample(fname, folder='samples_gan'):
    generator.eval()
    vae.eval()
    for imgs in dataloader:
        imgs = imgs[:32].to(device)
        break
    with torch.no_grad():
        z = torch.randn(32, 256).to(device)
        c_mu, c_logvar = vae.encode(imgs)
        c = vae.reparameterize(c_mu, c_logvar)
        z_ = torch.cat([z, c], 1)
        recon_imgs = generator(z_)
    samples = torch.cat([imgs.cpu(), recon_imgs.cpu()])
    save_image(samples, nrow=8, fp=f'{folder}/{fname}.jpg')

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

history = utils.load_history('gan')
if history is None:
    history = {'dloss': [], 'gloss': [], 'total_time': 0.}

MAX_ITERS = configs['gan']['MAX_ITERS']
CHECKPOINT_EVERY = configs['gan']['CHECKPOINT_EVERY']
SAMPLE_EVERY = configs['gan']['SAMPLE_EVERY']
PLOT_EVERY = configs['gan']['PLOT_EVERY']
D_STEPS = configs['gan']['D_STEPS']
it = len(history['dloss'])
finished = False
pbar = tqdm(total=MAX_ITERS)
pbar.update(it)
pbar.start_t = time() - history['total_time']
plt.ion()
fig, ax1 = plt.subplots()
line1, = ax1.plot(history['gloss'], label='gloss', c='b')
ax1.tick_params(axis='y', colors='b')
ax2 = plt.twinx(ax1)
line2, = ax2.plot(history['dloss'], label='dloss', c='r')
ax2.tick_params(axis='y', colors='r')
lines = [line1, line2]
plt.legend(lines, [line.get_label() for line in lines])
fig.canvas.draw()
fig.canvas.flush_events()
last_time = time()
while not finished:
    for imgs_real in dataloader:
        it += 1
        z = torch.randn(BATCH_SIZE, 256).to(device)
        imgs_real = imgs_real.to(device)
        
        # train discriminator
        generator.requires_grad_(False)
        vae.requires_grad_(False)
        discriminator.requires_grad_(True)
        generator.train()
        vae.train()
        discriminator.train()
        d_optimizer.zero_grad()
        
        imgs_real.requires_grad_()
        
        d_real = discriminator(imgs_real)
        target = torch.ones_like(d_real)
        dloss_real = F.binary_cross_entropy_with_logits(d_real, target)
        dloss_real.backward(retain_graph=True)
        reg = REG_PARAM * compute_grad2(d_real, imgs_real).mean()
        reg.backward()
        
        with torch.no_grad():
            c_mu, c_logvar = vae.encode(imgs_real)
            c = vae.reparameterize(c_mu, c_logvar)
            z_ = torch.cat([z, c], 1)
            imgs_fake = generator(z_)
        
        d_fake = discriminator(imgs_fake)
        target.fill_(0.)
        dloss_fake = F.binary_cross_entropy_with_logits(d_fake, target)
        dloss_fake.backward()
        
        d_optimizer.step()
        discriminator.requires_grad_(False)
        
        dloss = dloss_real + dloss_fake + reg
        history['dloss'].append(dloss.item())        
        if it % D_STEPS == 0:
            z = torch.randn(BATCH_SIZE, 256).to(device)
            
            # train generator
            generator.requires_grad_(True)
            vae.requires_grad_(True)
            vae.zero_grad()
            g_optimizer.zero_grad()
            
            z_ = torch.cat([z, c], 1)
            imgs_fake = generator(z_)
            d_fake = discriminator(imgs_fake)
            
            target.fill_(1.)
            gloss_d = F.binary_cross_entropy_with_logits(d_fake, target)
            
            ch_mu, ch_logvar = vae.encode(imgs_fake)
            gloss_info = (np.log(2*np.pi) + ch_logvar + (c-ch_mu).pow(2)/(ch_logvar.exp()+1e-8)).div(2).sum(1).mean()
            
            gloss = gloss_d + W_INFO * gloss_info
            gloss.backward()
            g_optimizer.step()
            
            for _ in range(D_STEPS):
                history['gloss'].append(gloss.item())
        if it % PLOT_EVERY == 0:
            ax1.set_xlim(0, len(history['dloss']))
            ax1.set_ylim(*utils.get_ylim(history['gloss']))
            ax2.set_ylim(*utils.get_ylim(history['dloss']))
            line1.set_data(range(1, it+1), history['gloss'])
            line2.set_data(range(1, it+1), history['dloss'])
            fig.canvas.draw()
            fig.canvas.flush_events()
        if it % CHECKPOINT_EVERY == 0:
            utils.save_cp(discriminator, 'd')
            utils.save_cp(generator, 'g')
            history['total_time'] += time() - last_time
            last_time = time()
            utils.save_history(history, 'gan')
        if it % SAMPLE_EVERY == 0:
            sample(f'iters_{it}')
        pbar.update()
        if it == MAX_ITERS:
            finished = True
            break
