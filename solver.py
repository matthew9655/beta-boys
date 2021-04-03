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
from model import BetaVAE_B
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
    dset_dir='', dataset='', batch_size=64):
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
        elif dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        net = BetaVAE_B

        self.net = cuda(net(self.latent_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,betas=(0.9, 0.999))

        # self.viz_name = "main"
        # self.viz_port = 8097
        # self.viz_on = True
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        # if self.viz_on:
        #     self.viz = visdom.Visdom(port=self.viz_port)

        # self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        # if not os.path.exists(self.ckpt_dir):
        #     os.makedirs(self.ckpt_dir, exist_ok=True)
        # self.ckpt_name = args.ckpt_name

        # self.save_output = True
        # self.output_dir = os.path.join("", self.viz_name) # TODO: Fill directory name in quotation
        # if not os.path.exists(self.output_dir):
        #     os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = 2
        self.display_step = 1
        # self.save_step = args.save_step

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

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.epochs)
        pbar.update(self.global_iter)
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

                # if self.viz_on and self.global_iter%self.gather_step == 0:
                #     self.gather.insert(iter=self.global_iter,
                #                        mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                #                        recon_loss=recon_loss.data, total_kld=total_kld.data,
                #                        dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                self.plot(x_recon)
                # Updating which batch we are doing right now
                self.cur_batch += 1

            # Resetting batch size numbers after one epoch is done
            self.cur_batch = 0 

            if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
+                        self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))
                    # pbar.write('iter: {}, elbo: {}'.format(
                    #     self.global_iter, beta_vae_loss[0]))

                    # var = logvar.exp().mean(0).data
                    # var_str = ''
                    # for j, var_j in enumerate(var):
                    #     var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    # pbar.write(var_str)

                    # if self.objective == 'B':
                    #     pbar.write('C:{:.3f}'.format(C.data[0])) 

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

    def plot(self, x_recon):
        if self.cur_batch == 0:
            for index in self.recon_indices:
                path = os.path.join("plot/", "Image (Batch 1): " + str(index) + " " + "Epoch: " + str(self.global_iter) + ".png")
                plt.imsave(path, np.resize(x_recon.detach().numpy(), (self.batch_size, 64, 64))[index], cmap='gray')