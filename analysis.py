import numpy as np
from glob import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

dataset = list(glob('*.npz'))
source_d = 'known_mnist.npz'
target_d = [d for d in dataset if d != 'known_mnist.npz']


# dataset = ['mnist', 'svhn', 'usps']

source = np.load(source_d)
source_feat = source['features']
source_lbl = source['labels']


targets = [np.load(d)['features'] for d in target_d]

pca = PCA(n_components=512)
pca.fit(source_feat)
# emb = pca.transform(source_feat)
emb = source_feat
mean_vec = ((emb.T @ np.eye(5)[source_lbl]) / source_lbl.sum(axis=0)).T
print(mean_vec.shape)
fig, axes = plt.subplots(figsize=(12.8, 4.8))
t = np.arange(0, 512)

print(emb.shape)
dist = emb - mean_vec[distance_matrix(emb, mean_vec).argmin(axis=1)]
std = dist.std(axis=0)
# plt.hist((dist / std).std(axis=0), label='known_mnist', log=True)
# axes.plot(t, (dist / std).std(axis=0), label='known_mnist')
axes.plot(t, np.log(np.abs(dist)).mean(axis=0), label='known_mnist', linewidth = 0.5)

for i, x in enumerate(targets):
    # x = pca.transform(x)
    x = pca.transform(x)
    dist = x - mean_vec[distance_matrix(x, mean_vec).argmin(axis=1)]
    # plt.hist((dist / std).std(axis=0), label=target_d[i], log=True)
    # axes.plot(t, (dist / std).std(axis=0), label=target_d[i], linewidth = 0.5)
    axes.plot(t, np.log(np.abs(dist)).mean(axis=0), label=target_d[i], linewidth = 0.5)

# axes.set_xlim(300, 500)
# axes.set_ylim(0, 10)
fig.legend()
axes.set_ylabel('log abs distance to nearest class mean')
# axes.set_xlabel('principal components')
fig.tight_layout()
plt.savefig('pca_variance.png')
