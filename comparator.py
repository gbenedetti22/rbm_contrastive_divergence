import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from mnist_data_loader import MnistDataloader
from rbm import RBM

# Caricamento dataset
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
x_train = np.array(x_train).reshape((-1, 784))
x_test = np.array(x_test).reshape((-1, 784))

# Normalizzazione tra 0 e 1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Addestramento RBM custom
print("Addestramento RBM custom...")
rbm_custom = RBM(n_visible=784, n_hidden=128, learning_rate=0.01)
rbm_custom.train(x_train, epochs=10)

# Addestramento RBM scikit-learn
print("Addestramento RBM sklearn...")
rbm_sklearn = BernoulliRBM(n_components=128, learning_rate=0.01, n_iter=10, verbose=True)
rbm_sklearn.fit(x_train)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reconstruct_sklearn(rbm, v, n_steps=1):
    v_sample = v.reshape(1, -1)

    for _ in range(n_steps):
        v_sample = rbm.gibbs(v_sample)

    return v_sample.flatten()

N = 10
indices = np.random.randint(0, len(x_test), N)

# Creiamo due figure, ognuna per 5 immagini
for group in range(0, N, 5):
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(indices[group:group + 5]):
        original = x_test[idx]

        # Ricostruzione RBM custom
        recon_custom = rbm_custom.reconstruct(original)

        # Ricostruzione RBM sklearn
        recon_sklearn = reconstruct_sklearn(rbm_sklearn, original, n_steps=10)

        # Calcolo errori
        error_custom = np.mean((original - recon_custom) ** 2)
        error_sklearn = np.mean((original - recon_sklearn) ** 2)

        # Plot immagini originali e ricostruite
        # Riga 1: Originali
        plt.subplot(3, 5, i + 1)
        plt.imshow(original.reshape(28, 28), cmap="gray")
        plt.title(f"Originale\nLabel {y_test[idx]}")
        plt.axis('off')

        # Riga 2: Custom RBM
        plt.subplot(3, 5, i + 6)
        plt.imshow(recon_custom.reshape(28, 28), cmap="gray")
        plt.title(f"Custom RBM\nMSE: {error_custom:.4f}")
        plt.axis('off')

        # Riga 3: Sklearn RBM
        plt.subplot(3, 5, i + 11)
        plt.imshow(recon_sklearn.reshape(28, 28), cmap="gray")
        plt.title(f"Sklearn RBM\nMSE: {error_sklearn:.4f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()