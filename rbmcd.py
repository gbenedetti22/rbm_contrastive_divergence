import numpy as np
from matplotlib import pyplot as plt

class RBMCD:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.__n_visible = n_visible
        self.__n_hidden = n_hidden
        self.__lr = learning_rate

        # Inizializzazione pesi e bias
        self.__W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.__a = np.zeros(n_visible)  # Bias visibili
        self.__b = np.zeros(n_hidden)   # Bias nascosti

        self.__rng = np.random.default_rng()

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Calcola la probabilità dei neuroni nascosti
    def __prop_up(self, v):
        return self.__sigmoid(np.dot(v, self.__W) + self.__b)

    # Calcola la probabilità dei neuroni visibili
    def __prop_down(self, h):
        return self.__sigmoid(np.dot(h, self.__W.T) + self.__a)

    # Funzione per campionare valori binari (0 o 1) da una probabilità
    def __sample_prob(self, probs):
        return self.__rng.binomial(1, probs)

    # Contrastive Divergence CD-1
    def __contrastive_divergence(self, v0, k=1):
        h_prob_0 = self.__prop_up(v0)
        h_state = self.__sample_prob(h_prob_0)

        v_state, h_prob, v_prob = None, None, None
        for step in range(k):
            # dream phase
            v_prob = self.__prop_down(h_state)
            v_state = self.__sample_prob(v_prob)

            # wake phase
            h_prob = self.__prop_up(v_state)
            h_state = self.__sample_prob(h_prob)

        self.__W += self.__lr * (np.outer(v0, h_prob_0) - np.outer(v_state, h_prob))
        self.__a += self.__lr * (v0 - v_state)
        self.__b += self.__lr * (h_prob_0 - h_prob)

        error = np.mean((v0 - v_prob) ** 2)
        return error

    def __visualize_weights(self, epoch=None, figsize=(10, 5)):
        """
        Visualizza i pesi della RBM come immagini 28x28
        Ogni riga rappresenta i pesi di un neurone nascosto
        """
        plt.figure(figsize=figsize)

        # Determina quanti neuroni nascosti visualizzare
        n_hidden_to_show = min(self.__n_hidden, 20)  # Mostra al massimo 20 neuroni

        # Determina la disposizione della griglia
        n_cols = 5
        n_rows = (n_hidden_to_show + n_cols - 1) // n_cols

        for i in range(n_hidden_to_show):
            plt.subplot(n_rows, n_cols, i + 1)

            # Prendi i pesi del neurone nascosto i-esimo e rimodellali in 28x28
            weights = self.__W[:, i].reshape(28, 28)

            # Normalizza i pesi per una migliore visualizzazione
            weights = (weights - weights.min()) / (weights.max() - weights.min())

            plt.imshow(weights, cmap='gray')
            plt.axis('off')
            plt.title(f'Neurone {i + 1}')

        title = "Pesi RBM"
        if epoch is not None:
            title += f" - Epoca {epoch}"
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def train(self, x, epochs=10, k=1):
        error = 0

        for epoch in range(epochs):
            for v in x:
                error += self.__contrastive_divergence(v, k=k)
            error /= len(x)

            # self.__visualize_weights(epoch)
            print(f"Epoch {epoch}, Error (Mean): {error:.4f}")

            error = 0

        return error

    def reconstruct(self, x):
        h_prob = self.__prop_up(x)

        return self.__prop_down(h_prob)

    def encode(self, x):
        return self.__prop_up(x)

    def save(self, filename="rbm"):
        np.savez(filename,
                 W=self.__W,
                 a=self.__a,
                 b=self.__b,
                 n_visible=self.__n_visible,
                 n_hidden=self.__n_hidden,
                 learning_rate=self.__lr)
        print(f"Model saved: {filename}.npz")

    def load(self, filename="rbm.npz"):
        data = np.load(filename)

        self.__W = data['W']
        self.__a = data['a']
        self.__b = data['b']
        self.__n_visible = int(data['n_visible'])
        self.__n_hidden = int(data['n_hidden'])
        self.__lr = float(data['learning_rate'])
        print(f"Model loaded from {filename}")