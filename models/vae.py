import torch
from torch import nn
from torch.nn import functional as F

def make_conv_block(in_channels, out_channels):
    layers = (nn.Conv2d(in_channels, out_channels, 4, 2, 1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
             )
    return layers

def make_transcov_block(in_channels, out_channels):
    layers = (nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
             )
    return layers

def normal_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(mean=0, std=0.02)
        if m.bias.data is not None:
            m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, z_size=20):
        super().__init__()
        
        self.encoder = nn.Sequential(  #input 3x64x64
            *make_conv_block(3, 32),  #32
            *make_conv_block(32, 32),  #16
            *make_conv_block(32, 64),  #8
            *make_conv_block(64, 64),  #64x4x4
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(256, z_size)
        self.logvar = nn.Linear(256, z_size)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, z_size=20):
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4*4*64),
            nn.ReLU(inplace=True),
        )
        
        self.conv = nn.Sequential(
            *make_transcov_block(64, 64),
            *make_transcov_block(64, 32),
            *make_transcov_block(32, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
    
    def forward(self, z):
        h = self.dense(z).view(-1, 64, 4, 4)
        return torch.tanh(self.conv(h))


class BetaVAE(nn.Module):
    def __init__(self, z_size=20, beta=1):
        super().__init__()
        self.z_size = z_size
        self.beta = beta
        self.encoder = Encoder(z_size)
        self.decoder = Decoder(z_size)
        self.apply(normal_init)
        
        self.id_mode = False
        
    def encode(self, x):
        if self.id_mode:
            #x = x.add(1).div(2)
            if (x.size(2) > 64) or (x.size(3) > 64):
                x = F.adaptive_avg_pool2d(x, (64, 64))
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        MSE = 0.5 * F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (MSE + KLD * self.beta) / x.shape[0], MSE/x.shape[0], KLD/x.shape[0]
    