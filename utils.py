import yaml
import os
import PIL
import torch
from typing import Callable, Optional
import json
import numpy as np


def mkdirs():
    dirs = ['checkpoints', 'history', 'samples_gan', 'samples_vae']
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)

def load_configs():
    with open('configs.yml') as f:
        configs = yaml.safe_load(f)
    return configs

def get_celeba_path(img_size, configs):
    assert(img_size in (64, 128, 256))
    path = os.path.join(configs['data']['CELEBA_PREPROCESSED_ROOT'], f"CelebA_{img_size}")
    return path

class CelebA(torch.utils.data.Dataset):
    def __init__(self, root_path:str, transform:Optional[Callable]=None):
        super().__init__()
        self.transform = transform
        self.root_path = root_path
        self.img_paths = os.listdir(root_path)
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx:int):
        path = os.path.join(self.root_path, self.img_paths[idx])
        img = PIL.Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
def get_last_cpidx(model_type):
    '''
    get the index of the last checkpoint
    model_type in ('v', 'g', 'd')
    '''
    dlist = os.listdir('checkpoints')
    dlist = list(filter(lambda x: x.startswith(model_type), dlist))
    if not dlist:
        last = -1
    else:
        last_name = sorted(dlist)[-1]
        last = int(last_name.split('.')[0][1:])
    return last

def load_last_cp(model, model_type):
    '''
    model_type in ('v', 'g', 'd')
    '''
    ind = get_last_cpidx(model_type)
    p = os.path.join('checkpoints', f'{model_type}{ind:05d}.pt')
    if os.path.isfile(p):
        model.load_state_dict(torch.load(p))
        if model_type == 'v':
            model_name = 'vae'
        elif model_type == 'g':
            model_name = 'generator'
        elif model_type == 'd':
            model_name = 'discriminator'
        print(f'{model_name} checkpoints No. {ind} successfully loaded')
    return model

def save_cp(model, model_type):
    '''
    model_type in ('v', 'g', 'd')
    '''
    ind = get_last_cpidx(model_type) + 1
    p = os.path.join('checkpoints', f'{model_type}{ind:05d}.pt')
    torch.save(model.state_dict(), p)

def save_history(history, train_type):
    '''
    train_type in ('vae', 'gan')
    '''
    p = os.path.join('history', f'{train_type}_history.json')
    with open(p, 'w') as f:
        json.dump(history, f)
    
def load_history(train_type):
    '''
    train_type in ('vae', 'gan')
    '''
    p = os.path.join('history', f'{train_type}_history.json')
    if os.path.isfile(p):
        with open(p) as f:
            history = json.load(f)
    else:
        history = None
    return history

def get_ylim(y):
    ymin = np.quantile(y, 0.1)
    ymax = np.quantile(y, 0.9)
    yspan = ymax - ymin
    ymin -= 0.2 * yspan
    ymax += 0.2 * yspan
    return ymin, ymax