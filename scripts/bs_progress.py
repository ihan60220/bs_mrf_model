import imageio

images = []

# append
for i in range(10, 80, 10):
    images.append(imageio.imread(f"../results/reconstruction/bs_recon_{i}.png"))

imageio.mimsave(f'../results/reconstruction/bs_recon_{10}_to_{70}.gif', images)