import imageio

images = []

for i in range(10, 80, 10):
    images.append(imageio.imread(f"results/reconstruction/reconstruction/bs_recon_{i}.png"))

imageio.mimsave(f'results/reconstruction/bs_recon_10_to_70.gif', images, fps=2)