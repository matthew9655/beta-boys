"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
# import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from random import randrange
import matplotlib.pyplot as plt
import numpy as np

from utils import cuda, grid2gif
from model import BetaVAE_B, BetaVAE_H
from dataset import return_data


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, epochs=2000, latent_dim = 10, gamma=1000, C_max=25, C_stop_iter=1e5, lr=200,
    dset_dir='', dataset='', batch_size=64, model='beta'):
        self.use_cuda = torch.cuda.is_available()
        self.epochs = epochs
        self.global_iter = 0
        self.cur_batch = 0

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.C_max = C_max
        self.C_stop_iter = C_stop_iter
        self.lr = lr
        self.recon_indices = []

        if dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif dataset.lower() == 'celeba':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        else:
            raise NotImplementedError
        
        if model == 'beta':
            net = BetaVAE_H
        else:
            net = BetaVAE_B

        self.model = model

        self.net = cuda(net(self.latent_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,betas=(0.9, 0.999))

        self.gather_step = 2
        self.display_step = 1

        self.dset_dir = dset_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader, dset_size = return_data(dset_dir=dset_dir, dset=self.dataset, batch_size=batch_size, image_size=64, num_workers=0)
        self.gather = DataGather()

        # Setting up indices for image sampling
        while len(self.recon_indices) < 5:
            index = randrange(self.batch_size)
            if index not in self.recon_indices:
                self.recon_indices.append(index)
        # Setting up file directory for plotting
        if not os.path.exists("plot/"):
            os.mkdir("plot")
            os.mkdir("plot/recon")
            os.mkdir("plot/latent")
            os.mkdir("plot/training_plots")

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.epochs)
        pbar.update(self.global_iter)
        C_validator = 0
        while not out:
            self.global_iter += 1
            pbar.update(1)
            for x in self.data_loader:
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.cur_batch == 0 and self.model != 'beta' and C_validator < self.C_max:
                    current_C = C.data[0]
                    if C_validator == 0  or current_C - C_validator > 0.1:
                        self.plot_recon(x_recon, current_C)
                        pbar.write('C = {}'.format(current_C))
                        pbar.write('plot saved at plot/training_plots')
                        C_validator = current_C + 0.1


                # Updating which batch we are doing right now
                self.cur_batch += 1

            # Resetting batch size numbers after one epoch is done
            self.cur_batch = 0 
            
            if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
+                        self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))

            if self.global_iter >= self.epochs:
                out = True
                break

        pbar.write("[Training Finished]")
        pbar.close()

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def plot_recon(self, x_recon, C):
            recon = F.sigmoid(x_recon.squeeze(1))
            recon = recon.detach().cpu().numpy()
            for index in range(len(self.recon_indices)):
                path = os.path.join("plot/training_plots/", "C_{}_Image_{}.png".format(C, self.recon_indices[index]))
                plt.imsave(path, recon[index], cmap='gray')
