from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np

def show_images_grid(imgs_, num_images=25):
        ncols = int(np.ceil(num_images**0.5))
        nrows = int(np.ceil(num_images / ncols))
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i < num_images:
                ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        plt.show()

class Sprites():
    def __init__(self):
        dataset_zip = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def sample_images(self, num_samples=5000):
        latents_sampled = self.sample_latent(size=num_samples)

        # Select images
        indices_sampled = self.latent_to_index(latents_sampled)
        return self.imgs[indices_sampled]

if __name__ == "__main__":
        s = Sprites()
        imgs_sampled = s.sample_images()
        show_images_grid(imgs_sampled)

        


