import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps

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

def latent_visual(model, images, latent_dim):
    with torch.no_grad():
        num_images = images.shape[0]
        distributions = model._encode(images)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        _, dim_wise_kl, _ = kl_divergence(mu, logvar)
        print('dimension wise kl')
        print(dim_wise_kl)
        z = reparametrize(mu, logvar)
        print('----')
        num_latent_dims = latent_dim

        num_latent_traverse = 10
        latent_traverse_arr = torch.linspace(-3, 3, 10)
        
        zs = torch.zeros(num_images * num_latent_traverse * num_latent_dims, num_latent_dims)
        for i in range(num_images):
            image_z = z[i]
            for j in range(num_latent_dims):
                for k in range(num_latent_traverse):
                    temp_image_z = torch.clone(image_z)
                    temp_image_z[j] = latent_traverse_arr[k]
                    zs[(100 * i) + (10 * j) + k] = temp_image_z
        x_hat = model._decode(zs)
        x_hat = F.sigmoid(x_hat).detach().to('cpu').numpy()
        x_hat = np.reshape(x_hat, (num_images * num_latent_traverse * num_latent_dims, 64, 64))
        for i in range(num_images):
            for j in range(num_latent_dims):
                for k in range(num_latent_traverse):
                    index = (100 * i) + (10 * j) + k
                    plt.imsave('plot/latent/latent_visual_image_{}_dim_{}_traverse_{}.png'.format(i, j, k), 
                    x_hat[index, :], cmap='gray')

def recon(model, images, latent_dim):
    with torch.no_grad():
        num_images = images.shape[0]
        distributions = model._encode(images)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        z = reparametrize(mu, logvar)
        x_recon = model._decode(z).view(images.size())
        x_recon = F.sigmoid(x_recon)

        # convert to numpy
        images = images.numpy()
        x_recon = x_recon.numpy()
        images = np.reshape(images, (num_images, 64, 64))
        x_recon = np.reshape(x_recon, (num_images, 64, 64))


        for i in range(num_images):
            plt.imsave('plot/recon/original_{}.png'.format(i), images[i], cmap='gray')
            plt.imsave('plot/recon/recon_{}.png'.format(i), x_recon[i], cmap='gray')






if __name__ == "__main__":
    pass
    



