import torch
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time

from models.vae import BetaVAE
import utils
from utils import load_configs, get_celeba_path, CelebA

utils.mkdirs()

configs = load_configs()
CELEBA_PATH = get_celeba_path(64, configs)
BATCH_SIZE = configs['vae']['BATCH_SIZE']
LR_VAE = configs['vae']['LR']
BETA = configs['vae']['BETA']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CelebA(CELEBA_PATH, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        drop_last=True,
                                        )

model = BetaVAE(beta=BETA).to(device)
model = utils.load_last_cp(model, 'v')
optimizer = optim.Adam(model.parameters(), lr=LR_VAE)

def sample(fname, folder='samples_vae'):
    model.eval()
    for imgs in dataloader:
        imgs = imgs[:32].to(device)
        break
    with torch.no_grad():
        recon_imgs, *_ = model(imgs)
    samples = torch.cat([imgs.cpu(), recon_imgs.cpu()])
    save_image(samples, nrow=8, fp=f'{folder}/{fname}.jpg')
    model.train()

history = utils.load_history('vae')
if history is None:
    history = {'recon': [], 'kld': [], 'total_time': 0.}

MAX_ITERS = configs['vae']['MAX_ITERS']
CHECKPOINT_EVERY = configs['vae']['CHECKPOINT_EVERY']
SAMPLE_EVERY = configs['vae']['SAMPLE_EVERY']
PLOT_EVERY = configs['vae']['PLOT_EVERY']
it = len(history['recon'])
finished = False
pbar = tqdm(total=MAX_ITERS)
pbar.update(it)
pbar.start_t = time() - history['total_time']
plt.ion()
fig, ax1 = plt.subplots()
line1, = ax1.plot(history['recon'], label='recon', c='b')
ax1.tick_params(axis='y', colors='b')
ax2 = plt.twinx(ax1)
line2, = ax2.plot(history['kld'], label='kld', c='r')
ax2.tick_params(axis='y', colors='r')
lines = [line1, line2]
plt.legend(lines, [line.get_label() for line in lines])
fig.canvas.draw()
fig.canvas.flush_events()
last_time = time()
while not finished:
    model.train()
    for imgs in dataloader:
        it += 1
        imgs = imgs.to(device)
        recon_imgs, mu, logvar = model(imgs)
        loss, mse, kld = model.loss(recon_imgs, imgs, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history['recon'].append(mse.item())
        history['kld'].append(kld.item())

        if it % PLOT_EVERY == 0:
            ax1.set_xlim(0, len(history['kld']))
            ax1.set_ylim(*utils.get_ylim(history['recon']))
            ax2.set_ylim(*utils.get_ylim(history['kld']))
            line1.set_data(range(1, it+1), history['recon'])
            line2.set_data(range(1, it+1), history['kld'])
            fig.canvas.draw()
            fig.canvas.flush_events()

        if it % CHECKPOINT_EVERY == 0:
            utils.save_cp(model, 'v')
            history['total_time'] += time() - last_time
            last_time = time()
            utils.save_history(history, 'vae')
        if it % SAMPLE_EVERY == 0:
            sample(f'iters_{it}')
        pbar.update()
        if it == MAX_ITERS:
            finished = True
            break


