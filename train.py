
import numpy as np
import torch

from solver import Solver
from utils import str2bool
from sprites_data import Sprites

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
    
    
if __name__ == "__main__":
    #dataset settings
    dset_dir = 'data'
    dataset = 'dsprites'

    #hyperparamters
    epochs =  200
    batch_size = 64
    latent_dim = 10
    gamma = 1000
    C_max = 0
    C_stop_iter = 0
    lr =  1e-5

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    net = Solver(dset_dir=dset_dir, dataset=dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim,
    gamma=gamma, C_max=C_max, C_stop_iter=C_stop_iter, lr=lr)

    
    net.train()
    
    

