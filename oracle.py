import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler

from mnist_data_loader import MnistDataloader


# Funzione per visualizzare i pesi della RBM
def visualize_rbm_weights(rbm, title="RBM Weights", figsize=(12, 6)):
    plt.figure(figsize=figsize)

    n_components = rbm.n_components
    n_cols = 5
    n_rows = (n_components + n_cols - 1) // n_cols

    for i in range(n_components):
        plt.subplot(n_rows, n_cols, i + 1)
        # Ottieni il componente i-esimo
        weight = rbm.components_[i].reshape(28, 28)

        # Normalizza per una migliore visualizzazione
        weight = (weight - weight.min()) / (weight.max() - weight.min())

        plt.imshow(weight, cmap='gray')
        plt.title(f'Componente {i + 1}')
        plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Funzione per visualizzare ricostruzioni
def visualize_reconstructions(rbm, X_test, n_samples=5, figsize=(12, 4)):
    plt.figure(figsize=figsize)

    for i in range(n_samples):
        # Immagine originale
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title('Originale')
        plt.axis('off')

        # Immagine ricostruita
        plt.subplot(2, n_samples, n_samples + i + 1)
        # Ricostruzione: trasforma avanti e indietro
        reconstruction = rbm.gibbs(X_test[i])
        plt.imshow(reconstruction.reshape(28, 28), cmap='gray')
        plt.title('Ricostruita')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Funzione per confrontare l'errore di ricostruzione tra due RBM
def compare_reconstruction_error(rbm1, rbm2, X_test, name1="RBM Personalizzata", name2="RBM Scikit"):
    # Calcola errore di ricostruzione per rbm1 (personalizzata)
    if hasattr(rbm1, 'reconstruct'):
        reconstructions1 = np.array([rbm1.reconstruct(x) for x in X_test])
    else:
        # Fallback per la tua implementazione
        def reconstruct(x):
            h_prob = 1 / (1 + np.exp(-(np.dot(x, rbm1.W) + rbm1.b)))
            v_prob = 1 / (1 + np.exp(-(np.dot(h_prob, rbm1.W.T) + rbm1.a)))
            return v_prob

        reconstructions1 = np.array([reconstruct(x) for x in X_test])

    error1 = np.mean((X_test - reconstructions1) ** 2)

    # Calcola errore di ricostruzione per rbm2 (scikit-learn)
    reconstructions2 = rbm2.gibbs(X_test)
    error2 = np.mean((X_test - reconstructions2) ** 2)

    print(f"Errore di ricostruzione {name1}: {error1:.6f}")
    print(f"Errore di ricostruzione {name2}: {error2:.6f}")

    # Visualizza gli errori
    plt.figure(figsize=(8, 5))
    plt.bar([name1, name2], [error1, error2], color=['blue', 'orange'])
    plt.ylabel('MSE')
    plt.title('Confronto Errore di Ricostruzione')
    plt.show()


mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
x_train = np.array(x_train).reshape((-1, 784))
x_test = np.array(x_test).reshape((-1, 784))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Inizializza e addestra la RBM di sklearn-contrib
rbm_sklearn = BernoulliRBM(
    n_components=128,
    learning_rate=0.1,
    n_iter=10,
    verbose=True,
)

print("Addestramento RBM sklearn...")
rbm_sklearn.fit(x_train)

# Visualizza i pesi appresi
visualize_rbm_weights(rbm_sklearn, "Pesi RBM Scikit-Learn")

# Visualizza alcune ricostruzioni
visualize_reconstructions(rbm_sklearn, x_test)

# Per confrontare con la tua RBM, avrai bisogno di adattare il confronto
# alla tua implementazione specifica

# Esempio di come potresti usare la funzione di confronto
# (decommentare e adattare quando hai la tua RBM)
"""
# Inizializza la tua RBM
my_rbm = RBM(n_visible=784, n_hidden=100, learning_rate=0.01)

# Addestra la tua RBM (supposizione della sintassi)
my_rbm.train(X_train, epochs=20)

# Confronta le due RBM
compare_reconstruction_error(my_rbm, rbm_sklearn, X_test)
"""
