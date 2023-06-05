
import umap
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# digits = load_digits()


# reducer = umap.UMAP(random_state=42)
# embedding = reducer.fit_transform(digits.data)
# print(embedding.shape)

# plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('UMAP projection of the Digits dataset')
# plt.show()

import torch

m = torch.randn(64, 64)
print(m.min().item(), m.max().item())