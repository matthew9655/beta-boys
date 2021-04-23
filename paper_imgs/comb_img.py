from PIL import Image

new_image = Image.new('RGB',(640, 64), (250,250,250))

img = 'recon_'
i = 0
for i in range(10):
    img_name = img + str(i) + '.png'
    opened = Image.open(img_name)
    new_image.paste(opened, (i*64, 0))
    i+=1

new_image.save("celeba_gamma_recon.jpg","JPEG")
