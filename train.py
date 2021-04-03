
import numpy as np
import torch

from solver import Solver
from utils import str2bool
from sprites_data import Sprites
from model import BetaVAE_B

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def save_model(model, model_name):
    torch.save(model.encoder.state_dict(), './saved_model/{}_encoder.pt'.format(model_name))
    torch.save(model.decoder.state_dict(), './saved_model/{}_decoder.pt'.format(model_name))

def load_model(model, model_name):
    model.encoder.load_state_dict(torch.load('./saved_model/{}_encoder.pt'.format(model_name)))
    model.decoder.load_state_dict(torch.load('./saved_model/{}_decoder.pt'.format(model_name)))
    model.encoder.eval()
    model.decoder.eval()

    
    
if __name__ == "__main__":
    #dataset settings
    dset_dir = 'data'
    dataset = 'dsprites'

    #hyperparamters
    epochs =  2
    batch_size = 5000
    latent_dim = 10
    gamma = 100
    C_max = 0
    C_stop_iter = 0
    lr =  5e-4

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    net = Solver(dset_dir=dset_dir, dataset=dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim,
    gamma=gamma, C_max=C_max, C_stop_iter=C_stop_iter, lr=lr)

    
    net.train()
    save_model(net.net, 'test')

    model = BetaVAE_B(z_dim=latent_dim)
    load_model(model, 'test')

    print(model)
    
    

