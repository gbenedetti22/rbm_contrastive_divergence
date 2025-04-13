import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from mnist_data_loader import MnistDataloader
import matplotlib.pyplot as plt

from rbm import RBM

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train).reshape((-1, 784))
x_test = np.array(x_test).reshape((-1, 784))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rbm = RBM(n_visible=784, n_hidden=128, learning_rate=0.01)

# Addestriamo la RBM — esempio 10 epoche
print("Training start")
rbm.train(x_train, epochs=10)
print("Training end")

sample = rbm.generate(n_steps=2000)
plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.title("Sample generato dalla RBM")
plt.axis('off')
plt.show()

# # Impostiamo il layout per visualizzare 10 coppie di immagini
# plt.figure(figsize=(20, 10))
#
# for i in range(10):
#     # Prendiamo un'immagine a caso dal test set
#     index = random.randint(0, len(x_test) - 1)
#     sample = x_test[index]
#
#     # Propaghiamo nel modello
#     x_pred = rbm.reconstruct(sample)
#
#     # Visualizziamo immagine originale
#     plt.subplot(2, 10, i + 1)
#     plt.title("Originale")
#     plt.imshow(sample.reshape(28, 28), cmap='gray')
#     plt.axis('off')
#
#     # Visualizziamo immagine ricostruita
#     plt.subplot(2, 10, i + 11)
#     plt.title("Ricostruita")
#     plt.imshow(x_pred.reshape(28, 28), cmap='gray')
#     plt.axis('off')
#
# plt.tight_layout()
# plt.show()