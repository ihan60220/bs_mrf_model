import imageio

images = []

for i in range(10, 110, 10):
    images.append(imageio.imread(f"../results/reconstruction/bs_recon_{i}.png"))

imageio.mimsave(f'../results/reconstruction/bs_recon.gif', images)