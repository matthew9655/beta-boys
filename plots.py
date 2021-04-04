import torch
import numpy as np
import matplotlib.pyplot as plt


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps

def latent_visual(model, images, latent_dim):
    with torch.no_grad():
        num_images = images.shape[0]
        distributions = model._encode(images)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        z = reparametrize(mu, logvar)
        num_latent_dims = latent_dim

        num_latent_traverse = 10
        latent_traverse_arr = torch.linspace(-3, 3, 10)
        
        zs = torch.zeros(num_images * num_latent_traverse, num_latent_dims)
        for i in range(num_images):
            image_z = z[i]
            for j in range(num_latent_dims):
                for k in range(num_latent_traverse):
                    temp_image_z = torch.clone(image_z)
                    temp_image_z[j] = latent_traverse_arr[k]
                    zs[(10 * i) + k] = temp_image_z
        
        x_hat = model._decode(zs).detach().to('cpu').numpy()
        x_hat = np.reshape(x_hat, (num_images * num_latent_traverse, 64, 64))
        for i in range(num_images):
            for j in range(num_latent_traverse):
                index = (10 * i) + j
                plt.imsave('plot/latent/latent_visual_image: {}_traverse{:.3f}.png'.format(i, latent_traverse_arr[j]), 
                x_hat[index, :], cmap='gray')


if __name__ == "__main__":
    pass
    



