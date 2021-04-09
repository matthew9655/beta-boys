
import numpy as np
import torch

from solver import Solver
from utils import str2bool
from sprites_data import Sprites
from model import BetaVAE_B, BetaVAE_H
from plots import latent_visual, recon

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

def convert_to_model_format(images):
    reshaped = np.reshape(images, (images.shape[0], 1, 64, 64))
    return torch.from_numpy(reshaped)


if __name__ == "__main__":
    #dataset settings
    dset_dir = 'data'
    dataset = 'dsprites'

    model = 'beta'

    #hyperparamters
    epochs = 5000
    batch_size = 512
    latent_dim = 10
    gamma = 4
    C_max = 0
    C_stop_iter = 0
    lr =  1e-4

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    net = Solver(dset_dir=dset_dir, dataset=dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim,
    gamma=gamma, C_max=C_max, C_stop_iter=C_stop_iter, lr=lr, model=model)

    
    # net.train()
    # save_model(net.net, '5000_epochs_beta_4')

    # model = BetaVAE_B(z_dim=latent_dim)
    model = BetaVAE_H(z_dim=latent_dim, nc=1)
    load_model(model, '5000_epochs_beta_4')
    data = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='bytes')
    rand = np.random.randint(0, 300000, 10)
    data = torch.from_numpy(data['imgs'][rand]).unsqueeze(1).float()
    
    latent_visual(model, data, latent_dim=latent_dim)

    # # recon code
    # recon(model, data, latent_dim)
    




    
    

